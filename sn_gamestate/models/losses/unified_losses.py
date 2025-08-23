from typing import Dict

import torch
import torch.nn.functional as F

def detection_loss(det_out: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
    # TODO: implement CenterNet-style focal + L1 losses
    return torch.tensor(0.0, device=next(iter(det_out.values())).device)

def pitch_loss(pitch_out: torch.Tensor, pitch_target: torch.Tensor) -> torch.Tensor:
    # BCE + Dice placeholder
    bce = F.binary_cross_entropy_with_logits(pitch_out, pitch_target)
    probs = torch.sigmoid(pitch_out)
    inter = (probs * pitch_target).sum()
    denom = probs.sum() + pitch_target.sum() + 1e-6
    dice = 1 - (2 * inter / denom)
    return bce + dice

def calibration_loss(calib_out: Dict[str, torch.Tensor], calib_target: Dict[str, torch.Tensor]) -> torch.Tensor:
    # L1 on parameters; later add reprojection loss via tvcalib projector
    loss = 0.0
    for k in calib_target.keys():
        loss = loss + F.l1_loss(calib_out[k], calib_target[k])
    return loss