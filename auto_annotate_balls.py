#!/usr/bin/env python3
"""
Auto-annotate soccer balls using pretrained YOLO for fine-tuning dataset creation.
This generates initial annotations that need manual review/correction.
"""

import argparse
import json
from pathlib import Path
import cv2
from ultralytics import YOLO
from tqdm import tqdm
import shutil


def auto_annotate_frames(frames_dir: Path, output_dir: Path, 
                         model_name: str = 'yolov8x.pt',
                         confidence: float = 0.15,
                         sample_interval: int = 1):
    """
    Auto-annotate ball positions in frames using YOLO.
    
    Args:
        frames_dir: Directory containing frame images
        output_dir: Output directory for annotations
        model_name: YOLO model to use for auto-annotation
        confidence: Detection confidence threshold (lower = more detections)
        sample_interval: Annotate every Nth frame (1 = all frames)
    """
    
    # Create output directories
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Load YOLO model
    print(f"Loading YOLO model: {model_name}")
    model = YOLO(model_name)
    
    # Find all image files
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_files = []
    for ext in image_extensions:
        image_files.extend(frames_dir.glob(f"*{ext}"))
        image_files.extend(frames_dir.glob(f"**/*{ext}"))
    
    image_files = sorted(set(image_files))
    
    # Sample frames if interval > 1
    if sample_interval > 1:
        image_files = image_files[::sample_interval]
    
    print(f"Found {len(image_files)} frames to annotate")
    
    # Statistics
    stats = {
        'total_frames': 0,
        'frames_with_ball': 0,
        'total_detections': 0,
        'avg_confidence': 0.0
    }
    
    annotations = []
    confidence_sum = 0.0
    
    # Process each frame
    for img_path in tqdm(image_files, desc="Auto-annotating"):
        # Read image to get dimensions
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        img_height, img_width = img.shape[:2]
        
        # Run YOLO detection
        results = model(str(img_path), classes=[32], conf=confidence, verbose=False)
        
        frame_annotations = []
        
        if results and len(results) > 0 and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                # Get box coordinates in YOLO format (center_x, center_y, width, height)
                # All normalized to [0, 1]
                xywh = box.xywh[0].cpu().numpy()
                conf = float(box.conf[0])
                
                # Normalize coordinates
                center_x = float(xywh[0]) / img_width
                center_y = float(xywh[1]) / img_height
                width = float(xywh[2]) / img_width
                height = float(xywh[3]) / img_height
                
                frame_annotations.append({
                    'class': 0,  # Ball class (YOLO format)
                    'center_x': center_x,
                    'center_y': center_y,
                    'width': width,
                    'height': height,
                    'confidence': conf,
                    'verified': False  # Needs manual review
                })
                
                confidence_sum += conf
                stats['total_detections'] += 1
        
        # Copy image to output directory
        output_img_name = f"frame_{stats['total_frames']:06d}{img_path.suffix}"
        output_img_path = images_dir / output_img_name
        shutil.copy(img_path, output_img_path)
        
        # Save YOLO format label file
        label_file = labels_dir / f"frame_{stats['total_frames']:06d}.txt"
        with open(label_file, 'w') as f:
            for ann in frame_annotations:
                # YOLO format: class center_x center_y width height
                f.write(f"{ann['class']} {ann['center_x']} {ann['center_y']} "
                       f"{ann['width']} {ann['height']}\n")
        
        annotations.append({
            'frame_id': stats['total_frames'],
            'image': str(output_img_path),
            'label_file': str(label_file),
            'original_path': str(img_path),
            'width': img_width,
            'height': img_height,
            'detections': frame_annotations
        })
        
        stats['total_frames'] += 1
        if len(frame_annotations) > 0:
            stats['frames_with_ball'] += 1
    
    # Calculate statistics
    if stats['total_detections'] > 0:
        stats['avg_confidence'] = confidence_sum / stats['total_detections']
    
    # Save detailed annotations (for review tool)
    annotations_file = output_dir / "annotations.json"
    with open(annotations_file, 'w') as f:
        json.dump({
            'annotations': annotations,
            'stats': stats
        }, f, indent=2)
    
    # Create dataset.yaml for YOLO training
    dataset_yaml = output_dir / "dataset.yaml"
    with open(dataset_yaml, 'w') as f:
        f.write(f"""# Soccer Ball Detection Dataset
path: {output_dir.absolute()}
train: images
val: images  # Will split later

# Classes
names:
  0: ball

# Training parameters
nc: 1  # number of classes
""")
    
    # Print summary
    print("\n" + "="*60)
    print("Auto-Annotation Complete")
    print("="*60)
    print(f"Total frames: {stats['total_frames']}")
    print(f"Frames with ball detected: {stats['frames_with_ball']} "
          f"({100*stats['frames_with_ball']/stats['total_frames']:.1f}%)")
    print(f"Total detections: {stats['total_detections']}")
    print(f"Average confidence: {stats['avg_confidence']:.3f}")
    print(f"\nOutput directory: {output_dir}")
    print(f"Annotations file: {annotations_file}")
    print(f"Dataset config: {dataset_yaml}")
    print("\n⚠️  IMPORTANT: These annotations need manual review!")
    print("   Run: python review_annotations.py --annotations annotations.json")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Auto-annotate soccer balls using YOLO for fine-tuning dataset'
    )
    parser.add_argument('--frames-dir', type=str, required=True,
                       help='Directory containing frame images')
    parser.add_argument('--output-dir', type=str, default='ball_dataset',
                       help='Output directory for annotations (default: ball_dataset)')
    parser.add_argument('--model', type=str, default='yolov8x.pt',
                       help='YOLO model for auto-annotation (default: yolov8x.pt)')
    parser.add_argument('--confidence', type=float, default=0.15,
                       help='Detection confidence threshold (default: 0.15)')
    parser.add_argument('--sample-interval', type=int, default=1,
                       help='Annotate every Nth frame (default: 1 = all frames)')
    
    args = parser.parse_args()
    
    frames_dir = Path(args.frames_dir)
    output_dir = Path(args.output_dir)
    
    if not frames_dir.exists():
        print(f"Error: Frames directory not found: {frames_dir}")
        return
    
    auto_annotate_frames(
        frames_dir=frames_dir,
        output_dir=output_dir,
        model_name=args.model,
        confidence=args.confidence,
        sample_interval=args.sample_interval
    )


if __name__ == '__main__':
    main()
