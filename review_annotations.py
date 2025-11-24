#!/usr/bin/env python3
"""
Interactive tool to review and correct auto-generated ball annotations.
Allows adding, deleting, and adjusting ball bounding boxes.
"""

import argparse
import json
from pathlib import Path
import cv2
import numpy as np


class AnnotationReviewer:
    def __init__(self, annotations_file: Path):
        """Load annotations for review"""
        with open(annotations_file) as f:
            data = json.load(f)
        
        self.annotations = data['annotations']
        self.stats = data.get('stats', {})
        self.current_idx = 0
        self.modified = False
        self.output_dir = annotations_file.parent
        
        # Mouse callback state
        self.drawing = False
        self.start_point = None
        self.current_box = None
        
        print(f"Loaded {len(self.annotations)} frames for review")
        print("\nControls:")
        print("  SPACE    - Next frame")
        print("  B        - Previous frame")
        print("  D        - Delete all detections in current frame")
        print("  A        - Add detection (click and drag to draw box)")
        print("  Y        - Mark frame as verified (good)")
        print("  S        - Save and exit")
        print("  Q        - Quit without saving")
        print("  H        - Show this help")
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing bounding boxes"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.current_box = None
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.current_box = (self.start_point[0], self.start_point[1], x, y)
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if self.start_point is not None:
                # Add new detection
                x1, y1 = self.start_point
                x2, y2 = x, y
                
                # Ensure x1 < x2 and y1 < y2
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                
                # Get image dimensions
                ann = self.annotations[self.current_idx]
                img_width = ann['width']
                img_height = ann['height']
                
                # Convert to YOLO format (normalized center_x, center_y, width, height)
                center_x = ((x1 + x2) / 2) / img_width
                center_y = ((y1 + y2) / 2) / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height
                
                new_detection = {
                    'class': 0,
                    'center_x': center_x,
                    'center_y': center_y,
                    'width': width,
                    'height': height,
                    'confidence': 1.0,
                    'verified': True
                }
                
                ann['detections'].append(new_detection)
                self.modified = True
                print(f"  Added detection at ({center_x:.3f}, {center_y:.3f})")
            
            self.start_point = None
            self.current_box = None
    
    def draw_annotations(self, img, ann):
        """Draw bounding boxes on image"""
        img_display = img.copy()
        img_height, img_width = img.shape[:2]
        
        # Draw existing detections
        for det in ann['detections']:
            # Convert from YOLO format to pixel coordinates
            center_x = int(det['center_x'] * img_width)
            center_y = int(det['center_y'] * img_height)
            w = int(det['width'] * img_width)
            h = int(det['height'] * img_height)
            
            x1 = center_x - w // 2
            y1 = center_y - h // 2
            x2 = center_x + w // 2
            y2 = center_y + h // 2
            
            # Color: green if verified, yellow if not
            color = (0, 255, 0) if det.get('verified') else (0, 255, 255)
            
            cv2.rectangle(img_display, (x1, y1), (x2, y2), color, 2)
            
            # Show confidence
            conf_text = f"{det['confidence']:.2f}"
            cv2.putText(img_display, conf_text, (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw current box being drawn
        if self.current_box is not None:
            x1, y1, x2, y2 = self.current_box
            cv2.rectangle(img_display, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Add frame info
        info_text = f"Frame {self.current_idx + 1}/{len(self.annotations)} | Detections: {len(ann['detections'])}"
        cv2.putText(img_display, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return img_display
    
    def save_annotations(self):
        """Save corrected annotations"""
        # Update label files
        labels_dir = self.output_dir / "labels"
        
        for ann in self.annotations:
            label_file = Path(ann['label_file'])
            
            # Write YOLO format labels
            with open(label_file, 'w') as f:
                for det in ann['detections']:
                    f.write(f"{det['class']} {det['center_x']} {det['center_y']} "
                           f"{det['width']} {det['height']}\n")
        
        # Save updated annotations JSON
        output_file = self.output_dir / "annotations_reviewed.json"
        
        # Update statistics
        frames_with_ball = sum(1 for ann in self.annotations if len(ann['detections']) > 0)
        total_detections = sum(len(ann['detections']) for ann in self.annotations)
        
        self.stats.update({
            'frames_with_ball': frames_with_ball,
            'total_detections': total_detections,
            'reviewed': True
        })
        
        with open(output_file, 'w') as f:
            json.dump({
                'annotations': self.annotations,
                'stats': self.stats
            }, f, indent=2)
        
        print(f"\n✅ Saved corrected annotations to: {output_file}")
        print(f"   Frames with ball: {frames_with_ball}/{len(self.annotations)}")
        print(f"   Total detections: {total_detections}")
    
    def run(self):
        """Main review loop"""
        cv2.namedWindow('Review Annotations')
        cv2.setMouseCallback('Review Annotations', self.mouse_callback)
        
        while True:
            if self.current_idx >= len(self.annotations):
                print("\n✅ Reached end of annotations")
                break
            
            ann = self.annotations[self.current_idx]
            
            # Load image
            img = cv2.imread(ann['image'])
            if img is None:
                print(f"Warning: Could not load image: {ann['image']}")
                self.current_idx += 1
                continue
            
            # Draw annotations
            img_display = self.draw_annotations(img, ann)
            
            cv2.imshow('Review Annotations', img_display)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # Space - next frame
                self.current_idx += 1
            
            elif key == ord('b'):  # B - previous frame
                self.current_idx = max(0, self.current_idx - 1)
            
            elif key == ord('d'):  # D - delete all detections
                if ann['detections']:
                    print(f"  Deleted {len(ann['detections'])} detection(s)")
                    ann['detections'] = []
                    self.modified = True
            
            elif key == ord('a'):  # A - add detection mode
                print("  Click and drag to draw bounding box")
            
            elif key == ord('y'):  # Y - mark as verified
                for det in ann['detections']:
                    det['verified'] = True
                self.modified = True
                print(f"  Marked frame {self.current_idx} as verified")
                self.current_idx += 1
            
            elif key == ord('s'):  # S - save and exit
                if self.modified:
                    self.save_annotations()
                else:
                    print("\n⚠️  No changes made")
                break
            
            elif key == ord('q'):  # Q - quit without saving
                if self.modified:
                    print("\n⚠️  Exiting without saving changes!")
                break
            
            elif key == ord('h'):  # H - show help
                print("\nControls:")
                print("  SPACE    - Next frame")
                print("  B        - Previous frame")
                print("  D        - Delete all detections in current frame")
                print("  A        - Add detection (click and drag)")
                print("  Y        - Mark frame as verified")
                print("  S        - Save and exit")
                print("  Q        - Quit without saving")
        
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description='Review and correct auto-generated ball annotations'
    )
    parser.add_argument('--annotations', type=str, required=True,
                       help='Path to annotations.json file')
    
    args = parser.parse_args()
    
    annotations_file = Path(args.annotations)
    if not annotations_file.exists():
        print(f"Error: Annotations file not found: {annotations_file}")
        return
    
    reviewer = AnnotationReviewer(annotations_file)
    reviewer.run()


if __name__ == '__main__':
    main()
