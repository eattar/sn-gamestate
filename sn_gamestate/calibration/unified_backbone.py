import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import logging
from typing import Any, Dict, Tuple, List

from tracklab.pipeline import ImageLevelModule
from tracklab.utils.collate import default_collate
from sn_calibration_baseline.camera import Camera

log = logging.getLogger(__name__)


class SpatioTemporalBackbone(nn.Module):
    """
    Unified spatio-temporal backbone for joint detection, pitch detection, and calibration.
    This replaces the separate YOLOv8 + TVCalib approach with a single model.
    """
    
    def __init__(self, 
                 input_channels: int = 3,
                 backbone_type: str = "resnet50",
                 num_detection_classes: int = 1,  # person class
                 num_pitch_classes: int = 29,     # pitch line classes
                 temporal_frames: int = 5,
                 use_attention: bool = True):
        super().__init__()
        
        self.backbone_type = backbone_type
        self.temporal_frames = temporal_frames
        self.use_attention = use_attention
        
        # Shared backbone encoder
        if backbone_type == "resnet50":
            from torchvision.models import resnet50, ResNet50_Weights
            backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            self.encoder = nn.Sequential(*list(backbone.children())[:-2])  # Remove avgpool and fc
            encoder_channels = 2048
        elif backbone_type == "efficientnet":
            from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
            backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            self.encoder = backbone.features
            encoder_channels = 1280
        else:
            raise ValueError(f"Unsupported backbone type: {backbone_type}")
        
        # Temporal modeling
        if temporal_frames > 1:
            self.temporal_conv = nn.Conv3d(encoder_channels, encoder_channels, 
                                         kernel_size=(3, 1, 1), padding=(1, 0, 0))
            self.temporal_norm = nn.BatchNorm3d(encoder_channels)
            self.temporal_activation = nn.ReLU(inplace=True)
        
        # Multi-task heads
        # 1. Detection head (bounding boxes + confidence)
        self.detection_head = nn.Sequential(
            nn.Conv2d(encoder_channels, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 5, 1)  # 4 bbox coords + 1 confidence
        )
        
        # 2. Pitch detection head (semantic segmentation)
        self.pitch_head = nn.Sequential(
            nn.Conv2d(encoder_channels, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_pitch_classes, 1)
        )
        
        # 3. Calibration head (camera parameters)
        self.calibration_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(encoder_channels, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 8)  # pan, tilt, roll, x_focal, y_focal, principal_point_x, principal_point_y, height
        )
        
        # Attention mechanism for spatio-temporal fusion
        if use_attention:
            self.spatial_attention = nn.MultiheadAttention(
                embed_dim=encoder_channels, 
                num_heads=8, 
                batch_first=True
            )
            self.temporal_attention = nn.MultiheadAttention(
                embed_dim=encoder_channels, 
                num_heads=8, 
                batch_first=True
            )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the unified backbone.
        
        Args:
            x: Input tensor of shape (B, T, C, H, W) where T is temporal frames
            
        Returns:
            Dictionary containing detection, pitch, and calibration outputs
        """
        batch_size, temporal_frames, channels, height, width = x.shape
        
        # Process each temporal frame through the encoder
        encoded_frames = []
        for t in range(temporal_frames):
            frame_features = self.encoder(x[:, t])  # (B, C, H', W')
            encoded_frames.append(frame_features)
        
        # Stack temporal features
        temporal_features = torch.stack(encoded_frames, dim=1)  # (B, T, C, H', W')
        
        # Apply temporal modeling if multiple frames
        if temporal_frames > 1:
            # Reshape for 3D convolution
            B, T, C, H, W = temporal_features.shape
            temporal_features = temporal_features.view(B, T * C, H, W)
            temporal_features = temporal_features.view(B, C, T, H, W)
            
            # Apply temporal convolution
            temporal_features = self.temporal_conv(temporal_features)
            temporal_features = self.temporal_norm(temporal_features)
            temporal_features = self.temporal_activation(temporal_features)
            
            # Reshape back
            temporal_features = temporal_features.view(B, C, T, H, W)
            temporal_features = temporal_features.view(B, T, C, H, W)
        
        # Apply attention mechanisms
        if self.use_attention:
            # For now, skip attention to get the basic pipeline working
            # TODO: Implement proper attention mechanism
            pass
        
        # Use the last temporal frame for detection and pitch (most recent)
        current_features = temporal_features[:, -1]  # (B, C, H', W')
        
        # Generate outputs
        detection_output = self.detection_head(current_features)
        pitch_output = self.pitch_head(current_features)
        calibration_output = self.calibration_head(current_features)
        
        return {
            'detection': detection_output,
            'pitch': pitch_output,
            'calibration': calibration_output,
            'features': current_features
        }


class UnifiedBackboneModule(ImageLevelModule):
    """
    TrackLab module that uses the unified spatio-temporal backbone.
    This replaces the separate bbox_detector, pitch, and calibration modules.
    """
    
    input_columns = []
    output_columns = {
        "detection": ["bbox_ltwh", "confidence", "bbox_pitch"],
        "pitch": ["lines", "keypoints"],
        "image": ["parameters"]
    }
    collate_fn = default_collate
    
    def __init__(self, 
                 model_path: str = None,
                 backbone_type: str = "resnet50",
                 temporal_frames: int = 5,
                 use_attention: bool = True,
                 detection_threshold: float = 0.5,
                 pitch_threshold: float = 0.3,
                 batch_size: int = 1,
                 device: str = "cuda",
                 **kwargs):
        super().__init__(batch_size=batch_size)
        
        self.backbone_type = backbone_type
        self.temporal_frames = temporal_frames
        self.use_attention = use_attention
        self.detection_threshold = detection_threshold
        self.pitch_threshold = pitch_threshold
        self.device = device
        
        # Initialize the unified backbone
        self.model = SpatioTemporalBackbone(
            backbone_type=backbone_type,
            temporal_frames=temporal_frames,
            use_attention=use_attention
        )
        
        # Load pretrained weights if provided
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=device))
        
        self.model.to(device)
        self.model.eval()
        
        # Frame buffer for temporal processing
        self.frame_buffer = []
        self.max_buffer_size = temporal_frames
    
    def preprocess(self, image: np.ndarray, detections: pd.DataFrame, metadata: pd.Series) -> torch.Tensor:
        """
        Preprocess image for the unified backbone.
        """
        # Convert BGR to RGB if needed
        if image.shape[2] == 3:
            image = image[:, :, ::-1]  # BGR to RGB
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
        
        # Add to frame buffer
        self.frame_buffer.append(image_tensor)
        if len(self.frame_buffer) > self.max_buffer_size:
            self.frame_buffer.pop(0)
        
        # Pad frame buffer if needed
        while len(self.frame_buffer) < self.temporal_frames:
            self.frame_buffer.insert(0, self.frame_buffer[0])
        
        # Stack frames for temporal processing
        temporal_input = torch.cat(self.frame_buffer, dim=0)  # (T, C, H, W)
        temporal_input = temporal_input.unsqueeze(0)  # (1, T, C, H, W)
        
        return temporal_input
    
    def process(self, batch: torch.Tensor, detections: pd.DataFrame, metadatas: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process batch through the unified backbone and return detection and calibration outputs.
        """
        # Ensure batch has the correct shape for temporal processing
        print(f"DEBUG: Original batch shape: {batch.shape}")
        
        if len(batch.shape) == 4:  # (B, C, H, W) - single frame
            # Add temporal dimension
            batch = batch.unsqueeze(1)  # (B, 1, C, H, W)
        elif len(batch.shape) == 5:  # (B, T, C, H, W) - temporal frames
            pass  # Already correct shape
        elif len(batch.shape) == 6:  # (B, ?, ?, C, H, W) - complex batch structure
            # Extract the image dimensions (last 3 dimensions)
            # Assuming the format is [B, ?, ?, C, H, W]
            batch_size = batch.shape[0]
            channels = batch.shape[-3]
            height = batch.shape[-2]
            width = batch.shape[-1]
            
            # Reshape to (B, T, C, H, W) where T = product of middle dimensions
            middle_dims = batch.shape[1:-3]
            temporal_frames = np.prod(middle_dims)
            
            # Reshape the batch
            batch = batch.view(batch_size, temporal_frames, channels, height, width)
            print(f"DEBUG: Reshaped batch to: {batch.shape}")
        else:
            raise ValueError(f"Unexpected batch shape: {batch.shape}. Expected (B, C, H, W), (B, T, C, H, W), or 6D batch")
        
        with torch.no_grad():
            batch = batch.to(self.device)
            print(f"DEBUG: Batch shape before model: {batch.shape}")
            outputs = self.model(batch)
        
        # Process detection outputs
        detection_outputs = self._process_detections(outputs['detection'], detections)
        
        # Process pitch outputs
        pitch_outputs = self._process_pitch(outputs['pitch'], metadatas)
        
        # Process calibration outputs
        calibration_outputs = self._process_calibration(outputs['calibration'], metadatas)
        
        return detection_outputs, calibration_outputs
    
    def _process_detections(self, detection_output: torch.Tensor, detections: pd.DataFrame) -> pd.DataFrame:
        """
        Process detection outputs to extract bounding boxes and confidence scores.
        """
        # detection_output shape: (B, 5, H, W) - 4 bbox coords + 1 confidence
        
        # Apply sigmoid to confidence
        confidence = torch.sigmoid(detection_output[:, 4:5])
        
        # Get bbox coordinates
        bbox_coords = detection_output[:, :4]
        
        # Find detections above threshold
        valid_detections = confidence > self.detection_threshold
        
        # For now, create a simplified detection DataFrame
        # In practice, you'd implement proper anchor-based detection
        detection_data = []
        
        # Create dummy detections for testing
        # This should be replaced with actual detection logic
        for i in range(len(detections)):
            # Create dummy bbox_ltwh (left, top, width, height)
            bbox_ltwh = [100, 100, 50, 100]  # Dummy values
            conf = 0.8  # Dummy confidence
            
            # Create dummy bbox_pitch using the existing function
            # For now, create a simple structure that matches expected format
            bbox_pitch = {
                "x_bottom_left": 0.0,
                "y_bottom_left": 0.0,
                "x_bottom_right": 10.0,
                "y_bottom_right": 0.0,
                "x_bottom_middle": 5.0,
                "y_bottom_middle": 0.0
            }
            
            detection_data.append({
                "bbox_ltwh": bbox_ltwh,
                "confidence": conf,
                "bbox_pitch": bbox_pitch
            })
        
        return pd.DataFrame(detection_data)
    
    def _process_pitch(self, pitch_output: torch.Tensor, metadatas: pd.DataFrame) -> pd.DataFrame:
        """
        Process pitch detection outputs to extract line keypoints.
        """
        # pitch_output shape: (B, num_pitch_classes, H, W)
        
        # Apply softmax to get class probabilities
        pitch_probs = F.softmax(pitch_output, dim=1)
        
        # Get predicted classes
        predicted_classes = torch.argmax(pitch_probs, dim=1)
        
        # For now, create dummy pitch data for testing
        # In practice, you'd implement proper keypoint extraction from the heatmaps
        pitch_data = []
        
        for i in range(len(metadatas)):
            # Create dummy lines and keypoints
            lines = []  # Dummy line data
            keypoints = []  # Dummy keypoint data
            
            pitch_data.append({
                "lines": lines,
                "keypoints": keypoints
            })
        
        return pd.DataFrame(pitch_data, index=metadatas.index)
    
    def _process_calibration(self, calibration_output: torch.Tensor, metadatas: pd.DataFrame) -> pd.DataFrame:
        """
        Process calibration outputs to extract camera parameters.
        """
        # calibration_output shape: (B, 8)
        
        # Extract camera parameters
        pan = calibration_output[:, 0]
        tilt = calibration_output[:, 1]
        roll = calibration_output[:, 2]
        x_focal = calibration_output[:, 3]
        y_focal = calibration_output[:, 4]
        principal_point_x = calibration_output[:, 5]
        principal_point_y = calibration_output[:, 6]
        height = calibration_output[:, 7]
        
        # Convert to camera parameters format
        camera_params = []
        for i in range(calibration_output.shape[0]):
            params = {
                "pan_degrees": pan[i].item(),
                "tilt_degrees": tilt[i].item(),
                "roll_degrees": roll[i].item(),
                "x_focal_length": x_focal[i].item(),
                "y_focal_length": y_focal[i].item(),
                "principal_point": [principal_point_x[i].item(), principal_point_y[i].item()],
                "position_meters": [0, 0, height[i].item()]  # Assuming camera at origin
            }
            camera_params.append(params)
        
        # Create output DataFrame
        output_df = pd.DataFrame([
            pd.Series({"parameters": params}, name=metadatas.iloc[i].name)
            for i, params in enumerate(camera_params)
        ])
        
        return output_df


def get_bbox_pitch(cam):
    """
    Convert bounding box to pitch coordinates using camera parameters.
    """
    def _get_bbox(bbox_ltrb):
        l, t, r, b = bbox_ltrb
        bl = np.array([l, b, 1])
        br = np.array([r, b, 1])
        bm = np.array([l+(r-l)/2, b, 1])
        
        pbl_x, pbl_y, _ = cam.unproject_point_on_planeZ0(bl)
        pbr_x, pbr_y, _ = cam.unproject_point_on_planeZ0(br)
        pbm_x, pbm_y, _ = cam.unproject_point_on_planeZ0(bm)
        
        if np.any(np.isnan([pbl_x, pbl_y, pbr_x, pbr_y, pbm_x, pbm_y])):
            return None
        
        return {
            "x_bottom_left": pbl_x, "y_bottom_left": pbl_y,
            "x_bottom_right": pbr_x, "y_bottom_right": pbr_y,
            "x_bottom_middle": pbm_x, "y_bottom_middle": pbm_y,
        }
    
    return _get_bbox
