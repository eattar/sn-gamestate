# Unified Spatio-Temporal Backbone (UnifiedST)

*Ready for iterative development and incremental integration*

Goal: Replace YOLOv8 + TVCalib with a single shared encoder that jointly predicts:
- Player/ball detections
- Pitch structure (lines/points) for supervision and visualization
- Camera parameters (pan/tilt/roll, intrinsics, position) with differentiable geometric consistency

Phases:
1) Static-stage scaffold with shared 2D encoder and 3 heads; training against existing labels and distillation targets.
2) Temporal-stage with video windows: 2D backbone + temporal convs, or 3D backbone (e.g., R(2+1)D / X3D / VideoSwin).
3) End-to-end geometry-aware training using tvcalib's differentiable projector to penalize reprojection error of predicted pitch/lines vs. masks/points.

Losses:
- Detection: focal/heatmap + L1/IoU for boxes (CenterNet/FCOS style).
- Pitch: BCE + Dice for line/point segmentation.
- Calibration: L1 on parameters (if GT exists), plus reprojection consistency loss using tvcalib's projector.
- Optional: temporal smoothing loss across windowed frames for cams and tracks.

Outputs (per image id):
- detection.bbox_ltrb: list of predicted boxes (ltrb)
- detection.bbox_pitch: same boxes projected onto pitch plane via predicted camera
- image.parameters: camera parameters in the same JSON shape as current modules
- image.lines (optional): line polylines or masks for visualization

Notes:
- We import tvcalib's Camera/SNProjectiveCamera to avoid code duplication and keep geometry consistent with existing TVCalib modules.
- Start with a light backbone (ResNet-18 + temporal conv or a small 3D CNN), then scale.

Open TODOs:
- Implement real heads, NMS/center decoding, and proper camera parameterization.
- Wire training script and datamodule against your datasets.
- Distill from current TVCalib/YOLO outputs as auxiliary supervision.