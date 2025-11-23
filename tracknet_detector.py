"""
TrackNet Ball Detector for Soccer

TrackNet is a deep learning model specifically designed for ball tracking in sports videos.
It performs significantly better than generic object detectors like YOLO for small ball detection.

Based on: https://github.com/Chang-Chia-Chi/TrackNet-Badminton-Tracking-tensorflow2
Adapted for soccer ball detection.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Optional, List
import torch
import torch.nn as nn


class TrackNet(nn.Module):
    """
    TrackNet architecture for ball detection
    
    Uses a U-Net style architecture with VGG-like encoder
    Input: 3 consecutive frames (HEIGHT x WIDTH x 9)
    Output: Heatmap (HEIGHT x WIDTH) where bright spots indicate ball location
    """
    
    def __init__(self, in_channels=9, out_channels=256):
        super(TrackNet, self).__init__()
        
        # Encoder (VGG-style)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Decoder (Upsampling)
        self.upsample1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        self.upsample2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        self.upsample3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv7 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Encoder
        x1 = self.conv1(x)
        x = self.pool1(x1)
        
        x2 = self.conv2(x)
        x = self.pool2(x2)
        
        x3 = self.conv3(x)
        x = self.pool3(x3)
        
        x = self.conv4(x)
        
        # Decoder
        x = self.upsample1(x)
        x = self.conv5(x)
        
        x = self.upsample2(x)
        x = self.conv6(x)
        
        x = self.upsample3(x)
        x = self.conv7(x)
        
        return x


class TrackNetBallDetector:
    """
    Ball detector using TrackNet
    
    Detects soccer balls in video frames using a temporal approach
    (analyzes 3 consecutive frames to track ball motion)
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        input_height: int = 288,
        input_width: int = 512,
        device: str = 'cuda',
        confidence_threshold: float = 0.5
    ):
        """
        Initialize TrackNet ball detector
        
        Args:
            model_path: Path to trained TrackNet weights (if None, uses pretrained)
            input_height: Input height for model (default: 288)
            input_width: Input width for model (default: 512)
            device: 'cuda' or 'cpu'
            confidence_threshold: Threshold for detection confidence
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.input_height = input_height
        self.input_width = input_width
        self.confidence_threshold = confidence_threshold
        
        # Initialize model
        self.model = TrackNet(in_channels=9, out_channels=256)
        
        if model_path and Path(model_path).exists():
            print(f"Loading TrackNet weights from: {model_path}")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            print("⚠️  No TrackNet weights provided - using untrained model")
            print("   For better results, train TrackNet on soccer ball dataset")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Frame buffer (stores last 3 frames)
        self.frame_buffer = []
        
        print(f"✓ TrackNet initialized on {self.device}")
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess a single frame
        
        Args:
            frame: BGR image (H, W, 3)
            
        Returns:
            Preprocessed frame (input_height, input_width, 3)
        """
        # Resize
        resized = cv2.resize(frame, (self.input_width, self.input_height))
        
        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        return normalized
    
    def detect(self, frame: np.ndarray) -> Optional[Tuple[int, int, float]]:
        """
        Detect ball in current frame
        
        Args:
            frame: BGR image (H, W, 3)
            
        Returns:
            Tuple of (x, y, confidence) if ball detected, None otherwise
            Coordinates are in original frame space
        """
        original_height, original_width = frame.shape[:2]
        
        # Preprocess and add to buffer
        preprocessed = self.preprocess_frame(frame)
        self.frame_buffer.append(preprocessed)
        
        # Keep only last 3 frames
        if len(self.frame_buffer) > 3:
            self.frame_buffer.pop(0)
        
        # Need 3 frames for prediction
        if len(self.frame_buffer) < 3:
            return None
        
        # Stack 3 frames: (H, W, 9)
        input_frames = np.concatenate(self.frame_buffer, axis=2)
        
        # Convert to torch tensor: (1, 9, H, W)
        input_tensor = torch.from_numpy(input_frames).permute(2, 0, 1).unsqueeze(0)
        input_tensor = input_tensor.to(self.device)
        
        # Run inference
        with torch.no_grad():
            output = self.model(input_tensor)
        
        # Get heatmap: (H, W, 256)
        heatmap = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        # Take maximum across channels
        heatmap_max = np.max(heatmap, axis=2)
        
        # Find peak
        max_confidence = np.max(heatmap_max)
        
        if max_confidence < self.confidence_threshold:
            return None
        
        # Get coordinates of maximum
        max_pos = np.unravel_index(np.argmax(heatmap_max), heatmap_max.shape)
        y_norm, x_norm = max_pos
        
        # Convert back to original frame coordinates
        x = int(x_norm * original_width / self.input_width)
        y = int(y_norm * original_height / self.input_height)
        
        return (x, y, float(max_confidence))
    
    def detect_in_region(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        margin: int = 50
    ) -> Optional[Tuple[int, int, float]]:
        """
        Detect ball in a specific region of the frame
        
        Args:
            frame: BGR image
            bbox: Bounding box (x1, y1, x2, y2)
            margin: Margin around bbox to search
            
        Returns:
            Tuple of (x, y, confidence) in full frame coordinates
        """
        x1, y1, x2, y2 = bbox
        height, width = frame.shape[:2]
        
        # Expand bbox with margin
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(width, x2 + margin)
        y2 = min(height, y2 + margin)
        
        # Crop region
        region = frame[y1:y2, x1:x2]
        
        # Detect in region
        result = self.detect(region)
        
        if result is None:
            return None
        
        # Convert coordinates back to full frame
        x_region, y_region, confidence = result
        x_full = x1 + x_region
        y_full = y1 + y_region
        
        return (x_full, y_full, confidence)
    
    def reset_buffer(self):
        """Reset the frame buffer (call when starting a new sequence)"""
        self.frame_buffer = []


def download_pretrained_tracknet(output_path: str = "tracknet_weights.pth"):
    """
    Download pretrained TrackNet weights
    
    Note: This is a placeholder. You'll need to either:
    1. Train TrackNet on soccer ball dataset
    2. Find pretrained weights for soccer/football
    3. Adapt weights from tennis/badminton TrackNet
    """
    print("⚠️  Pretrained TrackNet weights for soccer not yet available")
    print("   Options:")
    print("   1. Train TrackNet on your soccer dataset")
    print("   2. Use transfer learning from badminton/tennis TrackNet")
    print("   3. Continue with YOLO for now")
    return None


if __name__ == "__main__":
    # Test TrackNet
    print("TrackNet Ball Detector - Test")
    print("=" * 60)
    
    detector = TrackNetBallDetector(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test with dummy frames
    dummy_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    
    for i in range(5):
        result = detector.detect(dummy_frame)
        if result:
            x, y, conf = result
            print(f"Frame {i}: Ball detected at ({x}, {y}) with confidence {conf:.3f}")
        else:
            print(f"Frame {i}: No ball detected")
