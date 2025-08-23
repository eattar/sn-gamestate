"""
Training module for the Unified Spatio-Temporal Backbone.
"""

from .train_unified_backbone import (
    train_epoch,
    validate_epoch,
    MultiTaskLoss,
    SoccerNetDataset
)

__all__ = [
    'train_epoch',
    'validate_epoch', 
    'MultiTaskLoss',
    'SoccerNetDataset'
]
