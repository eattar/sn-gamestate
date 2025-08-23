from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import torch

# TODO: verify the correct import path for ImageLevelModule in your repo
try:
    from sn_gamestate.visualization.pitch import draw_pitch  # used in future viz
except Exception:
    pass

try:
    from sn_gamestate.calibration.tvcalib import Camera  # re-use Camera JSON IO + projection
except Exception:
    # Fallback path to baseline camera if needed
    from sn_calibration_baseline.camera import Camera  # type: ignore

from sn_gamestate.models.unified_st import UnifiedSTModel, decode_detections


class UnifiedSTModule:
    """
    Pipeline wrapper for the unified spatio-temporal model.

    Expected input (per sample):
    - 'image': a stack of T frames with shape (T, H, W, 3), RGB uint8
    - 'metadata': contains image_id and optional supervision

    Output:
    - detection.bbox_ltrb (DataFrame index = detection rows)
    - detection.bbox_pitch (projected using predicted camera)
    - image.parameters (per image_id, one row)
    - image.lines (optional: placeholder for future pitch polylines)
    """
    input_columns = {
        "image": ["frames"],  # ndarray (T, H, W, 3)
    }
    output_columns = {
        "image": ["parameters"],
        "detection": ["bbox_ltrb", "bbox_pitch"],
    }

    def __init__(self, image_width: int = 1920, image_height: int = 1080, batch_size: int = 1, device: str = "cuda", T: int = 5, with_distortion: bool = True, **kwargs):
        self.image_width = image_width
        self.image_height = image_height
        self.batch_size = batch_size
        self.device = device
        self.T = T
        self.model = UnifiedSTModel(with_distortion=with_distortion).to(self.device).eval()

    def preprocess(self, image: np.ndarray, detections: pd.DataFrame, metadata: pd.Series) -> Any:
        """
        image: expected to be a dict with key 'frames' -> np.ndarray (T, H, W, 3) RGB
        """
        frames = image["frames"]  # (T, H, W, 3)
        frames = frames.astype(np.float32) / 255.0
        frames_t = torch.from_numpy(frames).permute(0, 3, 1, 2)  # (T, C, H, W)
        frames_t = frames_t.unsqueeze(0).to(self.device)  # (1, T, C, H, W)
        return frames_t

    def _calib_to_json(self, calib_out: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        # Convert tensors to scalars; wrap into sn Camera JSON schema
        def t2s(x): return float(x.detach().cpu().item())
        cam_dict = {
            "pan_degrees": np.degrees(t2s(calib_out["pan"])),
            "tilt_degrees": np.degrees(t2s(calib_out["tilt"])),
            "roll_degrees": np.degrees(t2s(calib_out["roll"])),
            "aov_degrees": np.degrees(t2s(calib_out["aov"])),
            "position_meters": [t2s(calib_out["c_x"]), t2s(calib_out["c_y"]), t2s(calib_out["c_z"])],
        }
        if "k1" in calib_out:
            cam_dict["k1"] = t2s(calib_out["k1"])
        if "k2" in calib_out:
            cam_dict["k2"] = t2s(calib_out["k2"])
        return cam_dict

    def process(self, batch: Any, detections: pd.DataFrame, metadatas: pd.DataFrame):
        with torch.no_grad():
            out = self.model(batch)
        det_boxes = decode_detections(out["det"], conf_thresh=0.3)

        # Build camera JSON and a Camera object for projection
        cam_json = self._calib_to_json({k: v[0:1, ...] if v.ndim > 1 else v.unsqueeze(0) for k, v in out["calib"].items()})
        sn_cam = Camera(iwidth=self.image_width, iheight=self.image_height)
        try:
            sn_cam.from_json_parameters(cam_json)
        except Exception:
            # Some Camera paths expect extra fields; fall back on default principal point, etc.
            pass

        # Compose detections DataFrame outputs
        image_id = metadatas.index[0] if len(metadatas.index) > 0 else 0
        det_index = []
        det_ltrb = []
        det_pitch = []

        def get_bbox_pitch(cam):
            def _get_bbox(bbox_ltrb):
                l, t, r, b = bbox_ltrb
                bl = [l, b]
                br = [r, b]
                bm = [l + (r - l) / 2.0, b]
                pbl_x, pbl_y, _ = cam.unproject_point_on_planeZ0(bl)
                pbr_x, pbr_y, _ = cam.unproject_point_on_planeZ0(br)
                pbm_x, pbm_y, _ = cam.unproject_point_on_planeZ0(bm)
                return {
                    "left_bottom": [pbl_x, pbl_y],
                    "right_bottom": [pbr_x, pbr_y],
                    "bottom_middle": [pbm_x, pbm_y],
                }
            return _get_bbox

        for i, box in enumerate(det_boxes):
            det_index.append((image_id, i))
            det_ltrb.append(box)
            det_pitch.append(get_bbox_pitch(sn_cam)(box))

        df_det = pd.DataFrame(
            {"bbox_ltrb": det_ltrb, "bbox_pitch": det_pitch},
            index=pd.MultiIndex.from_tuples(det_index, names=["image_id", "row_id"]),
        )
        df_img = pd.DataFrame({"parameters": [sn_cam.to_json_parameters()]}, index=metadatas.index)
        return df_det[["bbox_ltrb", "bbox_pitch"]], df_img