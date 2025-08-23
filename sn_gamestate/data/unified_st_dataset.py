from typing import Dict, Any, Tuple, List
import numpy as np
import torch
from torch.utils.data import Dataset

class UnifiedSTDataset(Dataset):
    """
    Minimal dataset wrapper that returns:
    - frames: np.ndarray (T, H, W, 3) RGB
    - labels: dict with detection/pitch/calibration targets (optional for bootstrapping)
    """
    def __init__(self, index: List[Dict[str, Any]], T: int = 5):
        self.index = index
        self.T = T

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        rec = self.index[i]
        frames = rec["frames"]  # expected np.ndarray (T, H, W, 3)
        labels = rec.get("labels", {})
        return {"frames": frames, "labels": labels}