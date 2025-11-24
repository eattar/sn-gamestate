#!/usr/bin/env python3
"""
Fine-tune YOLO on soccer ball dataset.
Run this after reviewing and correcting annotations.
"""

import argparse
from pathlib import Path
import shutil
from ultralytics import YOLO
import yaml


def prepare_dataset(annotations_dir: Path, train_split: float = 0.8):
    """Split dataset into train/val sets"""
    
    images_dir = annotations_dir / "images"
    labels_dir = annotations_dir / "labels"
    
    # Get all images
    image_files = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    
    if len(image_files) == 0:
        raise ValueError(f"No images found in {images_dir}")
    
    # Split into train/val
    split_idx = int(len(image_files) * train_split)
    train_images = image_files[:split_idx]
    val_images = image_files[split_idx:]
    
    print(f"Dataset split: {len(train_images)} train, {len(val_images)} val")
    
    # Create train/val directories
    train_images_dir = annotations_dir / "train" / "images"
    train_labels_dir = annotations_dir / "train" / "labels"
    val_images_dir = annotations_dir / "val" / "images"
    val_labels_dir = annotations_dir / "val" / "labels"
    
    for d in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Copy files to train/val
    for img in train_images:
        shutil.copy(img, train_images_dir / img.name)
        label = labels_dir / f"{img.stem}.txt"
        if label.exists():
            shutil.copy(label, train_labels_dir / label.name)
    
    for img in val_images:
        shutil.copy(img, val_images_dir / img.name)
        label = labels_dir / f"{img.stem}.txt"
        if label.exists():
            shutil.copy(label, val_labels_dir / label.name)
    
    # Create updated dataset.yaml
    dataset_config = {
        'path': str(annotations_dir.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'names': {0: 'ball'},
        'nc': 1
    }
    
    yaml_path = annotations_dir / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print(f"Created dataset config: {yaml_path}")
    return yaml_path


def train_yolo(dataset_yaml: Path, 
               base_model: str = 'yolov8n.pt',
               epochs: int = 100,
               imgsz: int = 1280,
               batch: int = -1,
               output_name: str = 'soccer_ball_soccernet_v3',
               optimizer: str = 'SGD'):
    """
    Fine-tune YOLO on soccer ball dataset.
    Uses optimized configuration from kmouts/FootAndBall research.
    
    Args:
        dataset_yaml: Path to dataset configuration
        base_model: Base YOLO model (yolov8n.pt recommended for speed)
        epochs: Training epochs (100 with early stopping at 5)
        imgsz: Image size (1280 for small ball detection)
        batch: Batch size (-1 for auto-detection)
        output_name: Output model name
        optimizer: Optimizer (SGD recommended over Adam for this task)
    """
    
    print("\n" + "="*60)
    print("Fine-Tuning YOLO on Soccer Balls")
    print("Configuration from kmouts/FootAndBall research")
    print("="*60)
    print(f"Base model: {base_model}")
    print(f"Epochs: {epochs}")
    print(f"Image size: {imgsz} (large for small balls)")
    print(f"Batch size: {batch} (-1 = auto-detect)")
    print(f"Optimizer: {optimizer}")
    print("="*60 + "\n")
    
    # Load pretrained model
    model = YOLO(base_model)
    
    # Train with optimized configuration
    results = model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=5,  # Early stopping (kmouts config)
        save=True,
        device='cuda',  # Use GPU (change to 'cpu' if no GPU)
        
        # Optimization (kmouts config: SGD instead of Adam)
        optimizer=optimizer,
        lr0=0.001,  # Lower learning rate for fine-tuning
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        
        # kmouts specific settings
        pretrained=True,
        val=True,
        rect=True,  # Rectangular training
        cache=False,  # Don't cache (large dataset)
        workers=12,
        
        # Detection parameters
        conf=0.01,  # Low confidence during training
        iou=0.5,
        
        # Augmentation (optimized for soccer balls)
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5,  # Reduced rotation for broadcast
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,  # Full mosaic (kmouts uses 1.0)
        
        # Output
        project='runs/train',
        name=output_name,
        exist_ok=True,
        
        # Verbose
        verbose=True
    )
    
    # Get best model path
    best_model_path = Path('runs/train') / output_name / 'weights' / 'best.pt'
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Best model saved to: {best_model_path}")
    print(f"\nTo use the fine-tuned model:")
    print(f"  python run_player_ball_actions.py \\")
    print(f"    --ball-model {best_model_path} \\")
    print(f"    --ball-confidence 0.20 \\")
    print(f"    ...")
    print("="*60)
    
    return best_model_path


def main():
    parser = argparse.ArgumentParser(
        description='Fine-tune YOLO on soccer ball dataset'
    )
    parser.add_argument('--dataset-dir', type=str, required=True,
                       help='Directory containing annotations (from auto_annotate_balls.py)')
    parser.add_argument('--base-model', type=str, default='yolov8x.pt',
                       help='Base YOLO model to fine-tune (default: yolov8x.pt)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Training epochs (default: 50)')
    parser.add_argument('--imgsz', type=int, default=1280,
                       help='Image size for training (default: 1280)')
    parser.add_argument('--batch', type=int, default=8,
                       help='Batch size (default: 8, reduce if GPU memory error)')
    parser.add_argument('--train-split', type=float, default=0.8,
                       help='Train/val split ratio (default: 0.8)')
    parser.add_argument('--output-name', type=str, default='soccer_ball_yolo',
                       help='Output model name (default: soccer_ball_yolo)')
    
    args = parser.parse_args()
    
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        print(f"Error: Dataset directory not found: {dataset_dir}")
        return
    
    # Check if annotations have been reviewed
    reviewed_file = dataset_dir / "annotations_reviewed.json"
    if not reviewed_file.exists():
        print("⚠️  Warning: annotations_reviewed.json not found.")
        print("   It's recommended to review annotations first:")
        print(f"   python review_annotations.py --annotations {dataset_dir}/annotations.json")
        response = input("\nContinue with unreviewed annotations? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Prepare dataset
    dataset_yaml = prepare_dataset(dataset_dir, args.train_split)
    
    # Train
    best_model = train_yolo(
        dataset_yaml=dataset_yaml,
        base_model=args.base_model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        output_name=args.output_name
    )


if __name__ == '__main__':
    main()
