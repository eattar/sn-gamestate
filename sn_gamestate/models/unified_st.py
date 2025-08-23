from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Minimal backbone: 2D convs + shallow temporal fusion by 1D conv over time.
# Replace with a 3D conv or VideoSwin later.
class ConvEncoder2DTemporal(nn.Module):
    def __init__(self, in_ch=3, base_ch=64, num_stages=3, temporal_kernel=3):
        super().__init__()
        self.spatial = nn.ModuleList()
        ch = in_ch
        for i in range(num_stages):
            out_ch = base_ch * (2 ** i)
            self.spatial.append(
                nn.Sequential(
                    nn.Conv2d(ch, out_ch, kernel_size=3, padding=1, stride=2),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                )
            )
            ch = out_ch
        # temporal conv over features: expects input as (B, C, T, H, W)
        self.temporal = nn.Conv3d(ch, ch, kernel_size=(temporal_kernel, 1, 1),
                                  padding=(temporal_kernel // 2, 0, 0), groups=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        for stage in self.spatial:
            x = stage(x)
        _, C2, H2, W2 = x.shape
        x = x.view(B, T, C2, H2, W2).permute(0, 2, 1, 3, 4)  # (B, C2, T, H2, W2)
        x = self.temporal(x)  # (B, C2, T, H2, W2)
        # simple temporal pooling
        x = x.mean(dim=2)  # (B, C2, H2, W2)
        return x


class DetectionHead(nn.Module):
    # Anchor-free head (CenterNet-like). Placeholder.
    def __init__(self, in_ch: int, num_classes: int = 2):
        super().__init__()
        self.hm = nn.Conv2d(in_ch, num_classes, 1)
        self.wh = nn.Conv2d(in_ch, 2, 1)  # width/height
        self.reg = nn.Conv2d(in_ch, 2, 1)  # center offset

    def forward(self, f: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"heatmap": self.hm(f), "wh": self.wh(f), "reg": self.reg(f)}


class PitchSegHead(nn.Module):
    # Predicts line/point heatmaps. Placeholder.
    def __init__(self, in_ch: int, out_ch: int = 1):
        super().__init__()
        self.out = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.out(f)


class CalibrationHead(nn.Module):
    # Simple MLP over spatial pooled features to regress camera params.
    # Paramization: pan, tilt, roll, aov, c_x, c_y, c_z, optional k1,k2
    def __init__(self, in_ch: int, with_distortion: bool = True):
        super().__init__()
        self.with_distortion = with_distortion
        dims = 7 + (2 if with_distortion else 0)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Conv2d(in_ch, 256, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, dims, 1),
        )

    def forward(self, f: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.pool(f)
        x = self.mlp(x).flatten(1)  # (B, dims)
        idx = 0
        pan, tilt, roll, aov = x[:, idx:idx+1], x[:, idx+1:idx+2], x[:, idx+2:idx+3], x[:, idx+3:idx+4]
        idx += 4
        c_x, c_y, c_z = x[:, idx:idx+1], x[:, idx+1:idx+2], x[:, idx+2:idx+3]
        idx += 3
        out = {
            "pan": pan, "tilt": tilt, "roll": roll, "aov": aov,
            "c_x": c_x, "c_y": c_y, "c_z": c_z
        }
        if self.with_distortion:
            out["k1"] = x[:, idx:idx+1]
            out["k2"] = x[:, idx+1:idx+2]
        return out


class UnifiedSTModel(nn.Module):
    def __init__(self, num_det_classes: int = 2, pitch_out_ch: int = 1, with_distortion: bool = True):
        super().__init__()
        self.backbone = ConvEncoder2DTemporal(in_ch=3, base_ch=64, num_stages=3)
        self.det_head = DetectionHead(in_ch=64 * (2 ** 2), num_classes=num_det_classes)
        self.pitch_head = PitchSegHead(in_ch=64 * (2 ** 2), out_ch=pitch_out_ch)
        self.calib_head = CalibrationHead(in_ch=64 * (2 ** 2), with_distortion=with_distortion)

    def forward(self, frames_btchw: torch.Tensor) -> Dict[str, torch.Tensor]:
        # frames: (B, T, C, H, W)
        f = self.backbone(frames_btchw)
        det = self.det_head(f)
        pitch = self.pitch_head(f)
        calib = self.calib_head(f)
        return {
            "det": det,
            "pitch": pitch,
            "calib": calib,
        }

# Decoding stubs for detection and conversion to LTRB will be implemented later.
def decode_detections(det_out: Dict[str, torch.Tensor], conf_thresh: float = 0.3) -> List[List[float]]:
    # TODO: implement top-K center decode + size/offset to boxes
    return []