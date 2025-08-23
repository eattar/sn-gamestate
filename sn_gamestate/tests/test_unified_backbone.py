#!/usr/bin/env python3
"""
Test script for the Unified Spatio-Temporal Backbone.
This script tests the basic functionality of the unified backbone.
"""

import unittest
import torch
import numpy as np
import tempfile
import os
from pathlib import Path

# Add the project root to the path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from sn_gamestate.calibration.unified_backbone import SpatioTemporalBackbone, UnifiedBackboneModule


class TestSpatioTemporalBackbone(unittest.TestCase):
    """Test cases for the SpatioTemporalBackbone class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 2
        self.temporal_frames = 3
        self.channels = 3
        self.height = 64
        self.width = 64
        
        # Create test input
        self.test_input = torch.randn(
            self.batch_size, 
            self.temporal_frames, 
            self.channels, 
            self.height, 
            self.width
        )
    
    def test_backbone_initialization(self):
        """Test backbone initialization with different configurations."""
        # Test ResNet-50 backbone
        backbone = SpatioTemporalBackbone(
            backbone_type="resnet50",
            temporal_frames=self.temporal_frames,
            use_attention=True
        )
        self.assertIsNotNone(backbone)
        self.assertEqual(backbone.backbone_type, "resnet50")
        self.assertEqual(backbone.temporal_frames, self.temporal_frames)
        self.assertTrue(backbone.use_attention)
        
        # Test EfficientNet backbone
        backbone = SpatioTemporalBackbone(
            backbone_type="efficientnet",
            temporal_frames=self.temporal_frames,
            use_attention=False
        )
        self.assertIsNotNone(backbone)
        self.assertEqual(backbone.backbone_type, "efficientnet")
        self.assertEqual(backbone.temporal_frames, self.temporal_frames)
        self.assertFalse(backbone.use_attention)
    
    def test_invalid_backbone_type(self):
        """Test that invalid backbone types raise an error."""
        with self.assertRaises(ValueError):
            SpatioTemporalBackbone(backbone_type="invalid_backbone")
    
    def test_forward_pass_resnet50(self):
        """Test forward pass with ResNet-50 backbone."""
        backbone = SpatioTemporalBackbone(
            backbone_type="resnet50",
            temporal_frames=self.temporal_frames,
            use_attention=True
        )
        
        # Move to device
        backbone = backbone.to(self.device)
        test_input = self.test_input.to(self.device)
        
        # Forward pass
        with torch.no_grad():
            output = backbone(test_input)
        
        # Check output structure
        self.assertIn('detection', output)
        self.assertIn('pitch', output)
        self.assertIn('calibration', output)
        self.assertIn('features', output)
        
        # Check output shapes
        self.assertEqual(output['detection'].shape[0], self.batch_size)
        self.assertEqual(output['pitch'].shape[0], self.batch_size)
        self.assertEqual(output['calibration'].shape[0], self.batch_size)
        self.assertEqual(output['features'].shape[0], self.batch_size)
    
    def test_forward_pass_efficientnet(self):
        """Test forward pass with EfficientNet backbone."""
        backbone = SpatioTemporalBackbone(
            backbone_type="efficientnet",
            temporal_frames=self.temporal_frames,
            use_attention=False
        )
        
        # Move to device
        backbone = backbone.to(self.device)
        test_input = self.test_input.to(self.device)
        
        # Forward pass
        with torch.no_grad():
            output = backbone(test_input)
        
        # Check output structure
        self.assertIn('detection', output)
        self.assertIn('pitch', output)
        self.assertIn('calibration', output)
        self.assertIn('features', output)
    
    def test_single_frame(self):
        """Test with single temporal frame."""
        backbone = SpatioTemporalBackbone(
            backbone_type="resnet50",
            temporal_frames=1,
            use_attention=True
        )
        
        # Single frame input
        single_frame_input = torch.randn(self.batch_size, 1, self.channels, self.height, self.width)
        
        # Move to device
        backbone = backbone.to(self.device)
        single_frame_input = single_frame_input.to(self.device)
        
        # Forward pass
        with torch.no_grad():
            output = backbone(single_frame_input)
        
        # Check outputs
        self.assertIn('detection', output)
        self.assertIn('pitch', output)
        self.assertIn('calibration', output)
    
    def test_model_save_load(self):
        """Test model saving and loading."""
        backbone = SpatioTemporalBackbone(
            backbone_type="resnet50",
            temporal_frames=self.temporal_frames,
            use_attention=True
        )
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            # Save model
            torch.save(backbone.state_dict(), tmp_path)
            
            # Load model
            new_backbone = SpatioTemporalBackbone(
                backbone_type="resnet50",
                temporal_frames=self.temporal_frames,
                use_attention=True
            )
            new_backbone.load_state_dict(torch.load(tmp_path))
            
            # Test that loaded model produces same output
            backbone.eval()
            new_backbone.eval()
            
            with torch.no_grad():
                output1 = backbone(self.test_input)
                output2 = new_backbone(self.test_input)
            
            # Check that outputs are the same
            for key in output1.keys():
                torch.testing.assert_close(output1[key], output2[key])
                
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestUnifiedBackboneModule(unittest.TestCase):
    """Test cases for the UnifiedBackboneModule class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 1
        self.temporal_frames = 3
        
        # Create test image
        self.test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Create test detections DataFrame
        import pandas as pd
        self.test_detections = pd.DataFrame({
            'bbox_ltwh': [[100, 100, 50, 100]],
            'confidence': [0.9]
        })
        
        # Create test metadata
        self.test_metadata = pd.Series({
            'image_id': 'test_image_001',
            'frame_id': 0
        })
    
    def test_module_initialization(self):
        """Test module initialization."""
        module = UnifiedBackboneModule(
            backbone_type="resnet50",
            temporal_frames=self.temporal_frames,
            use_attention=True,
            batch_size=self.batch_size,
            device=str(self.device)
        )
        
        self.assertIsNotNone(module)
        self.assertEqual(module.backbone_type, "resnet50")
        self.assertEqual(module.temporal_frames, self.temporal_frames)
        self.assertTrue(module.use_attention)
        self.assertEqual(module.batch_size, self.batch_size)
    
    def test_preprocess(self):
        """Test preprocessing functionality."""
        module = UnifiedBackboneModule(
            backbone_type="resnet50",
            temporal_frames=self.temporal_frames,
            use_attention=True,
            batch_size=self.batch_size,
            device=str(self.device)
        )
        
        # Test preprocessing
        processed = module.preprocess(self.test_image, self.test_detections, self.test_metadata)
        
        # Check output shape
        self.assertEqual(processed.shape[0], 1)  # batch size
        self.assertEqual(processed.shape[1], self.temporal_frames)  # temporal frames
        self.assertEqual(processed.shape[2], 3)  # channels
        self.assertEqual(processed.shape[3], 480)  # height
        self.assertEqual(processed.shape[4], 640)  # width
    
    def test_process(self):
        """Test processing functionality."""
        module = UnifiedBackboneModule(
            backbone_type="resnet50",
            temporal_frames=self.temporal_frames,
            use_attention=True,
            batch_size=self.batch_size,
            device=str(self.device)
        )
        
        # Create test batch
        batch = torch.randn(1, self.temporal_frames, 3, 480, 640)
        
        # Test processing
        detection_outputs, calibration_outputs = module.process(
            batch, self.test_detections, self.test_metadata
        )
        
        # Check outputs
        self.assertIsInstance(detection_outputs, pd.DataFrame)
        self.assertIsInstance(calibration_outputs, pd.DataFrame)
    
    def test_frame_buffer(self):
        """Test frame buffer functionality."""
        module = UnifiedBackboneModule(
            backbone_type="resnet50",
            temporal_frames=self.temporal_frames,
            use_attention=True,
            batch_size=self.batch_size,
            device=str(self.device)
        )
        
        # Process multiple images to fill buffer
        for i in range(self.temporal_frames + 2):
            processed = module.preprocess(self.test_image, self.test_detections, self.test_metadata)
            
            # Check that buffer size is maintained
            self.assertLessEqual(len(module.frame_buffer), module.max_buffer_size)
        
        # Check that buffer has correct size
        self.assertEqual(len(module.frame_buffer), module.max_buffer_size)


class TestIntegration(unittest.TestCase):
    """Integration tests for the unified backbone."""
    
    def test_end_to_end_pipeline(self):
        """Test end-to-end pipeline with dummy data."""
        # Create backbone
        backbone = SpatioTemporalBackbone(
            backbone_type="resnet50",
            temporal_frames=3,
            use_attention=True
        )
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, 3, 64, 64)
        
        # Test forward pass
        with torch.no_grad():
            output = backbone(dummy_input)
        
        # Verify all outputs are present
        required_keys = ['detection', 'pitch', 'calibration', 'features']
        for key in required_keys:
            self.assertIn(key, output)
            self.assertIsInstance(output[key], torch.Tensor)
        
        # Verify output shapes are reasonable
        self.assertEqual(output['detection'].shape[0], 1)  # batch size
        self.assertEqual(output['pitch'].shape[0], 1)      # batch size
        self.assertEqual(output['calibration'].shape[0], 1)  # batch size
        self.assertEqual(output['features'].shape[0], 1)     # batch size


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
