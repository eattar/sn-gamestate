#!/usr/bin/env python3
"""
Training script for the Unified Spatio-Temporal Backbone.
This script trains the unified backbone on SoccerNet data for joint detection, pitch detection, and calibration.
"""

import os
import sys
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
from tqdm import tqdm
import wandb

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent.parent))

from sn_gamestate.calibration.unified_backbone import SpatioTemporalBackbone
from sn_gamestate.calibration.camera import Camera

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SoccerNetDataset(torch.utils.data.Dataset):
    """
    Dataset class for SoccerNet data that provides temporal sequences.
    This is a simplified version - you'll need to adapt it to your actual data format.
    """
    
    def __init__(self, data_root: str, split: str = "train", temporal_frames: int = 5):
        self.data_root = Path(data_root)
        self.split = split
        self.temporal_frames = temporal_frames
        
        # Load data annotations
        self.annotations = self._load_annotations()
        
        # Get unique video sequences
        self.video_sequences = self._get_video_sequences()
        
        logger.info(f"Loaded {len(self.video_sequences)} video sequences for {split} split")
    
    def _load_annotations(self):
        """Load annotations from the dataset."""
        # This is a placeholder - you'll need to implement based on your data format
        # For now, we'll create dummy data
        return pd.DataFrame({
            'video_id': ['video_001'] * 100,
            'frame_id': list(range(100)),
            'bbox_ltwh': [[100, 100, 50, 100]] * 100,
            'pitch_lines': [[[100, 100, 200, 100]]] * 100,
            'camera_params': [{'pan': 0, 'tilt': 0, 'roll': 0, 'x_focal': 1000, 'y_focal': 1000, 'principal_point': [320, 240], 'height': 10}] * 100
        })
    
    def _get_video_sequences(self):
        """Get unique video sequences."""
        return self.annotations['video_id'].unique().tolist()
    
    def __len__(self):
        return len(self.video_sequences)
    
    def __getitem__(self, idx):
        video_id = self.video_sequences[idx]
        video_frames = self.annotations[self.annotations['video_id'] == video_id]
        
        # Get temporal sequence of frames
        if len(video_frames) < self.temporal_frames:
            # Pad with last frame if not enough frames
            last_frame = video_frames.iloc[-1]
            while len(video_frames) < self.temporal_frames:
                video_frames = pd.concat([video_frames, pd.DataFrame([last_frame])], ignore_index=True)
        
        # Select frames for temporal sequence
        frame_indices = np.linspace(0, len(video_frames) - 1, self.temporal_frames, dtype=int)
        selected_frames = video_frames.iloc[frame_indices]
        
        # Load images (placeholder - you'll need to implement actual image loading)
        images = []
        for _, frame in selected_frames.iterrows():
            # Create dummy image (replace with actual image loading)
            img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            images.append(img)
        
        # Prepare targets
        targets = {
            'detection': self._prepare_detection_targets(selected_frames),
            'pitch': self._prepare_pitch_targets(selected_frames),
            'calibration': self._prepare_calibration_targets(selected_frames)
        }
        
        return torch.stack([torch.from_numpy(img).float() / 255.0 for img in images]), targets
    
    def _prepare_detection_targets(self, frames):
        """Prepare detection targets."""
        # This is a placeholder - implement based on your detection format
        return torch.zeros(self.temporal_frames, 5, 480, 640)
    
    def _prepare_pitch_targets(self, frames):
        """Prepare pitch detection targets."""
        # This is a placeholder - implement based on your pitch format
        return torch.zeros(self.temporal_frames, 29, 480, 640)
    
    def _prepare_calibration_targets(self, frames):
        """Prepare calibration targets."""
        # This is a placeholder - implement based on your calibration format
        return torch.zeros(self.temporal_frames, 8)


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss function for the unified backbone.
    Combines detection, pitch detection, and calibration losses.
    """
    
    def __init__(self, 
                 detection_weight: float = 1.0,
                 pitch_weight: float = 1.0,
                 calibration_weight: float = 1.0):
        super().__init__()
        self.detection_weight = detection_weight
        self.pitch_weight = pitch_weight
        self.calibration_weight = calibration_weight
        
        # Loss functions
        self.detection_loss = nn.MSELoss()
        self.pitch_loss = nn.CrossEntropyLoss()
        self.calibration_loss = nn.MSELoss()
    
    def forward(self, predictions, targets):
        """
        Compute multi-task loss.
        
        Args:
            predictions: Dictionary with 'detection', 'pitch', 'calibration' keys
            targets: Dictionary with corresponding target values
        
        Returns:
            Total loss and individual losses
        """
        # Detection loss (MSE for bbox regression + confidence)
        detection_loss = self.detection_loss(predictions['detection'], targets['detection'])
        
        # Pitch loss (Cross-entropy for semantic segmentation)
        pitch_loss = self.pitch_loss(predictions['pitch'], targets['pitch'].long())
        
        # Calibration loss (MSE for camera parameters)
        calibration_loss = self.calibration_loss(predictions['calibration'], targets['calibration'])
        
        # Total loss
        total_loss = (self.detection_weight * detection_loss + 
                     self.pitch_weight * pitch_loss + 
                     self.calibration_weight * calibration_loss)
        
        return total_loss, {
            'detection': detection_loss.item(),
            'pitch': pitch_loss.item(),
            'calibration': calibration_loss.item(),
            'total': total_loss.item()
        }


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    loss_components = {'detection': 0, 'pitch': 0, 'calibration': 0}
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, (images, targets) in enumerate(progress_bar):
        # Move to device
        images = images.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(images)
        
        # Compute loss
        loss, loss_breakdown = criterion(predictions, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        for k, v in loss_breakdown.items():
            if k in loss_components:
                loss_components[k] += v
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Det': f'{loss_breakdown["detection"]:.4f}',
            'Pitch': f'{loss_breakdown["pitch"]:.4f}',
            'Calib': f'{loss_breakdown["calibration"]:.4f}'
        })
    
    # Calculate average losses
    avg_loss = total_loss / len(dataloader)
    avg_components = {k: v / len(dataloader) for k, v in loss_components.items()}
    
    return avg_loss, avg_components


def validate_epoch(model, dataloader, criterion, device, epoch):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0
    loss_components = {'detection': 0, 'pitch': 0, 'calibration': 0}
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc=f'Validation {epoch}')
        
        for batch_idx, (images, targets) in enumerate(progress_bar):
            # Move to device
            images = images.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}
            
            # Forward pass
            predictions = model(images)
            
            # Compute loss
            loss, loss_breakdown = criterion(predictions, targets)
            
            # Update metrics
            total_loss += loss.item()
            for k, v in loss_breakdown.items():
                if k in loss_components:
                    loss_components[k] += v
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Det': f'{loss_breakdown["detection"]:.4f}',
                'Pitch': f'{loss_breakdown["pitch"]:.4f}',
                'Calib': f'{loss_breakdown["calibration"]:.4f}'
            })
    
    # Calculate average losses
    avg_loss = total_loss / len(dataloader)
    avg_components = {k: v / len(dataloader) for k, v in loss_components.items()}
    
    return avg_loss, avg_components


def main():
    parser = argparse.ArgumentParser(description='Train Unified Spatio-Temporal Backbone')
    parser.add_argument('--config', type=str, default='configs/training_config.yaml',
                       help='Path to training configuration file')
    parser.add_argument('--data-root', type=str, required=True,
                       help='Path to SoccerNet dataset root')
    parser.add_argument('--output-dir', type=str, default='outputs/training',
                       help='Output directory for training artifacts')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--no-wandb', action='store_true',
                       help='Disable wandb logging')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Default configuration
        config = {
            'model': {
                'backbone_type': 'resnet50',
                'temporal_frames': 5,
                'use_attention': True
            },
            'training': {
                'batch_size': 4,
                'num_epochs': 100,
                'learning_rate': 1e-4,
                'weight_decay': 1e-4,
                'detection_weight': 1.0,
                'pitch_weight': 1.0,
                'calibration_weight': 1.0
            },
            'data': {
                'train_split': 'train',
                'val_split': 'val',
                'num_workers': 4
            }
        }
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup wandb
    if not args.no_wandb:
        wandb.init(
            project="soccernet-unified-backbone",
            config=config,
            name=f"unified-backbone-{config['model']['backbone_type']}"
        )
    
    # Create datasets
    train_dataset = SoccerNetDataset(
        data_root=args.data_root,
        split=config['data']['train_split'],
        temporal_frames=config['model']['temporal_frames']
    )
    
    val_dataset = SoccerNetDataset(
        data_root=args.data_root,
        split=config['data']['val_split'],
        temporal_frames=config['model']['temporal_frames']
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    # Create model
    model = SpatioTemporalBackbone(
        backbone_type=config['model']['backbone_type'],
        temporal_frames=config['model']['temporal_frames'],
        use_attention=config['model']['use_attention']
    )
    
    # Move to device
    model = model.to(device)
    
    # Create loss function
    criterion = MultiTaskLoss(
        detection_weight=config['training']['detection_weight'],
        pitch_weight=config['training']['pitch_weight'],
        calibration_weight=config['training']['calibration_weight']
    )
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Create scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        logger.info(f'Resuming from checkpoint: {args.resume}')
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f'Resumed from epoch {start_epoch}')
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, config['training']['num_epochs']):
        logger.info(f'Starting epoch {epoch}')
        
        # Train
        train_loss, train_components = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_components = validate_epoch(
            model, val_loader, criterion, device, epoch
        )
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Log metrics
        if not args.no_wandb:
            wandb.log({
                'epoch': epoch,
                'train/loss': train_loss,
                'train/detection_loss': train_components['detection'],
                'train/pitch_loss': train_components['pitch'],
                'train/calibration_loss': train_components['calibration'],
                'val/loss': val_loss,
                'val/detection_loss': val_components['detection'],
                'val/pitch_loss': val_components['pitch'],
                'val/calibration_loss': val_components['calibration'],
                'learning_rate': optimizer.param_groups[0]['lr']
            })
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'config': config,
            'train_loss': train_loss,
            'val_loss': val_loss
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, output_dir / 'latest_checkpoint.pth')
        
        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, output_dir / 'best_checkpoint.pth')
            logger.info(f'New best validation loss: {best_val_loss:.4f}')
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save(checkpoint, output_dir / f'checkpoint_epoch_{epoch}.pth')
        
        logger.info(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    logger.info('Training completed!')
    
    # Save final model
    final_model_path = output_dir / 'final_model.pth'
    torch.save(model.state_dict(), final_model_path)
    logger.info(f'Final model saved to: {final_model_path}')


if __name__ == '__main__':
    main()
