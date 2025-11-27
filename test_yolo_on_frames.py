#!/usr/bin/env python3
"""
Test YOLO Model on Frames

Run a YOLO model on a directory of frames and save visualized results with bounding boxes.
Useful for evaluating model accuracy and detection quality.

Usage:
    python test_yolo_on_frames.py --frames-dir /path/to/frames --model yolov8x.pt --output-dir results
    
    # Test fine-tuned model
    python test_yolo_on_frames.py \
        --frames-dir /path/to/frames \
        --model pretrained_models/yolov8n_soccernet_v3_best.pt \
        --output-dir results_finetuned
    
    # Filter specific classes
    python test_yolo_on_frames.py \
        --frames-dir /path/to/frames \
        --model yolov8x.pt \
        --output-dir results \
        --classes 0 32  # person and sports ball only
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np
from tqdm import tqdm


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Test YOLO model on frames and save visualized results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python test_yolo_on_frames.py --frames-dir frames/ --model yolov8x.pt --output-dir results/
  
  # Test fine-tuned model
  python test_yolo_on_frames.py \\
      --frames-dir frames/ \\
      --model pretrained_models/yolov8n_soccernet_v3_best.pt \\
      --output-dir results_finetuned/
  
  # Filter specific classes (e.g., person=0, sports ball=32)
  python test_yolo_on_frames.py \\
      --frames-dir frames/ \\
      --model yolov8x.pt \\
      --output-dir results/ \\
      --classes 0 32
  
  # Adjust confidence threshold
  python test_yolo_on_frames.py \\
      --frames-dir frames/ \\
      --model yolov8x.pt \\
      --output-dir results/ \\
      --conf 0.5
        """
    )
    
    parser.add_argument('--frames-dir', type=str, required=True,
                       help='Directory containing input frames (jpg/png)')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to YOLO model (.pt file)')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory to save visualized frames with bboxes')
    parser.add_argument('--classes', type=int, nargs='+', default=None,
                       help='Filter specific class IDs (e.g., --classes 0 32 for person and ball)')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold (default: 0.25)')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IoU threshold for NMS (default: 0.45)')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum number of frames to process (default: all)')
    parser.add_argument('--skip-frames', type=int, default=1,
                       help='Process every Nth frame (default: 1 = all frames)')
    parser.add_argument('--show-labels', action='store_true',
                       help='Show class labels and confidence on bboxes')
    parser.add_argument('--save-stats', action='store_true',
                       help='Save detection statistics to JSON file')
    
    return parser.parse_args()


def get_frame_files(frames_dir: Path, max_frames: Optional[int] = None, skip: int = 1) -> List[Path]:
    """Get list of frame files from directory"""
    # Support common image formats
    frame_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        frame_files.extend(sorted(frames_dir.glob(ext)))
    
    # Remove duplicates and sort
    frame_files = sorted(set(frame_files))
    
    # Apply skip
    if skip > 1:
        frame_files = frame_files[::skip]
    
    # Apply max frames limit
    if max_frames is not None:
        frame_files = frame_files[:max_frames]
    
    return frame_files


def draw_detections(image: np.ndarray, boxes, class_names: dict, show_labels: bool = True) -> np.ndarray:
    """Draw bounding boxes on image"""
    img_draw = image.copy()
    
    if boxes is None or len(boxes) == 0:
        return img_draw
    
    # Color palette for different classes
    np.random.seed(42)
    colors = {}
    for cls_id in class_names.keys():
        colors[cls_id] = tuple(map(int, np.random.randint(0, 255, 3)))
    
    for box in boxes:
        # Get box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        cls_name = class_names.get(cls_id, f'class_{cls_id}')
        
        # Get color for this class
        color = colors.get(cls_id, (0, 255, 0))
        
        # Draw bounding box
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)
        
        # Draw label if enabled
        if show_labels:
            label = f'{cls_name} {conf:.2f}'
            
            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            # Draw background rectangle
            cv2.rectangle(
                img_draw,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                color,
                -1
            )
            
            # Draw text
            cv2.putText(
                img_draw,
                label,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
    
    return img_draw


def main():
    args = parse_args()
    
    # Validate paths
    frames_dir = Path(args.frames_dir)
    if not frames_dir.exists():
        print(f"❌ Error: Frames directory not found: {frames_dir}")
        sys.exit(1)
    
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Error: Model file not found: {model_path}")
        sys.exit(1)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("YOLO Model Testing on Frames")
    print("=" * 70)
    print(f"Frames directory: {frames_dir}")
    print(f"Model: {model_path}")
    print(f"Output directory: {output_dir}")
    print(f"Confidence threshold: {args.conf}")
    print(f"IoU threshold: {args.iou}")
    
    # Load YOLO model
    print("\n📦 Loading YOLO model...")
    try:
        from ultralytics import YOLO
        model = YOLO(str(model_path))
        print(f"✓ Model loaded successfully")
        
        # Print model info
        class_names = model.names
        print(f"✓ Model has {len(class_names)} classes:")
        for cls_id, cls_name in sorted(class_names.items())[:10]:
            print(f"    {cls_id}: {cls_name}")
        if len(class_names) > 10:
            print(f"    ... and {len(class_names) - 10} more")
        
        # Show which classes will be detected
        if args.classes is not None:
            print(f"\n🎯 Filtering classes: {args.classes}")
            for cls_id in args.classes:
                cls_name = class_names.get(cls_id, f'unknown_{cls_id}')
                print(f"    {cls_id}: {cls_name}")
        else:
            print(f"\n🎯 Detecting all {len(class_names)} classes")
            
    except ImportError:
        print("❌ Error: ultralytics not installed")
        print("   Install with: pip install ultralytics")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)
    
    # Get frame files
    print(f"\n📂 Scanning frames directory...")
    frame_files = get_frame_files(frames_dir, args.max_frames, args.skip_frames)
    
    if len(frame_files) == 0:
        print(f"❌ Error: No image files found in {frames_dir}")
        sys.exit(1)
    
    print(f"✓ Found {len(frame_files)} frames to process")
    if args.skip_frames > 1:
        print(f"  (Processing every {args.skip_frames} frames)")
    
    # Process frames
    print(f"\n🔍 Running detection on frames...")
    
    stats = {
        'total_frames': len(frame_files),
        'frames_with_detections': 0,
        'total_detections': 0,
        'detections_per_class': {},
        'avg_confidence_per_class': {}
    }
    
    for frame_path in tqdm(frame_files, desc="Processing frames"):
        # Read frame
        img = cv2.imread(str(frame_path))
        if img is None:
            print(f"\n⚠️  Warning: Could not read {frame_path.name}, skipping...")
            continue
        
        # Run detection
        results = model(
            img,
            classes=args.classes,
            conf=args.conf,
            iou=args.iou,
            verbose=False
        )
        
        # Get detections
        boxes = results[0].boxes if len(results) > 0 else None
        
        # Update statistics
        if boxes is not None and len(boxes) > 0:
            stats['frames_with_detections'] += 1
            stats['total_detections'] += len(boxes)
            
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = class_names.get(cls_id, f'class_{cls_id}')
                
                if cls_name not in stats['detections_per_class']:
                    stats['detections_per_class'][cls_name] = 0
                    stats['avg_confidence_per_class'][cls_name] = []
                
                stats['detections_per_class'][cls_name] += 1
                stats['avg_confidence_per_class'][cls_name].append(conf)
        
        # Draw detections
        img_with_boxes = draw_detections(img, boxes, class_names, args.show_labels)
        
        # Save result
        output_path = output_dir / frame_path.name
        cv2.imwrite(str(output_path), img_with_boxes)
    
    # Calculate average confidences
    for cls_name in stats['avg_confidence_per_class']:
        confs = stats['avg_confidence_per_class'][cls_name]
        stats['avg_confidence_per_class'][cls_name] = sum(confs) / len(confs) if confs else 0.0
    
    # Print statistics
    print("\n" + "=" * 70)
    print("Detection Statistics")
    print("=" * 70)
    print(f"Total frames processed: {stats['total_frames']}")
    print(f"Frames with detections: {stats['frames_with_detections']} ({stats['frames_with_detections']/stats['total_frames']*100:.1f}%)")
    print(f"Total detections: {stats['total_detections']}")
    print(f"Average detections per frame: {stats['total_detections']/stats['total_frames']:.2f}")
    
    if stats['detections_per_class']:
        print(f"\nDetections per class:")
        for cls_name, count in sorted(stats['detections_per_class'].items(), key=lambda x: x[1], reverse=True):
            avg_conf = stats['avg_confidence_per_class'][cls_name]
            print(f"  {cls_name}: {count} detections (avg conf: {avg_conf:.3f})")
    
    # Save statistics to JSON if requested
    if args.save_stats:
        import json
        stats_file = output_dir / 'detection_stats.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\n💾 Statistics saved to: {stats_file}")
    
    print(f"\n✅ Done! Visualized frames saved to: {output_dir}")
    print(f"   Total frames: {len(frame_files)}")
    print(f"   Frames with detections: {stats['frames_with_detections']}")


if __name__ == '__main__':
    main()
