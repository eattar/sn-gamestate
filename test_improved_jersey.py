#!/usr/bin/env python3
"""
Test script for the improved jersey recognition system.
This script tests the basic functionality of the new modules.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'sn_gamestate'))

import pandas as pd
import numpy as np
from jersey.improved_jersey_recognition import ImprovedJerseyRecognition
from jersey.enhanced_tracklet_aggregation import EnhancedTrackletAggregation

def test_improved_jersey_recognition():
    """Test the ImprovedJerseyRecognition module."""
    print("Testing ImprovedJerseyRecognition module...")
    
    try:
        # Initialize module (without actual device for testing)
        module = ImprovedJerseyRecognition(
            batch_size=2,
            device='cpu',
            sequence_length=3,
            min_confidence_threshold=0.3
        )
        
        print("✓ Module initialized successfully")
        
        # Test utility functions
        assert module.extract_numbers("Player 10") == "10"
        assert module.extract_numbers("No numbers") is None
        assert module.validate_jersey_number("10") == True
        assert module.validate_jersey_number("100") == False
        assert module.validate_jersey_number("abc") == False
        
        print("✓ Utility functions working correctly")
        
        # Test temporal consistency
        jersey_numbers = ["10", "10", "10"]
        confidences = [0.8, 0.9, 0.7]
        temporal_consistency = module.compute_temporal_consistency(jersey_numbers, confidences)
        assert temporal_consistency == 1.0
        
        jersey_numbers_mixed = ["10", "9", "10"]
        temporal_consistency_mixed = module.compute_temporal_consistency(jersey_numbers_mixed, confidences)
        assert 0.5 < temporal_consistency_mixed < 1.0
        
        print("✓ Temporal consistency computation working")
        
        print("✓ All tests passed for ImprovedJerseyRecognition!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

def test_enhanced_tracklet_aggregation():
    """Test the EnhancedTrackletAggregation module."""
    print("\nTesting EnhancedTrackletAggregation module...")
    
    try:
        # Initialize module
        cfg = {
            'min_tracklet_length': 3,
            'confidence_threshold': 0.5,
            'temporal_decay': 0.9,
            'spatial_threshold': 10.0
        }
        module = EnhancedTrackletAggregation(cfg, device='cpu')
        
        print("✓ Module initialized successfully")
        
        # Test temporal weight computation
        frame_indices = [0, 1, 2]
        temporal_weight = module.compute_temporal_weight(frame_indices)
        assert 0 < temporal_weight < 1
        
        print("✓ Temporal weight computation working")
        
        # Test spatial weight computation
        # Mock bbox data
        class MockBbox:
            def __init__(self, x, y):
                self.center = (x, y)
        
        bboxes = [MockBbox(0, 0), MockBbox(1, 1), MockBbox(2, 2)]
        spatial_weight = module.compute_spatial_weight(bboxes)
        assert 0 < spatial_weight < 1
        
        print("✓ Spatial weight computation working")
        
        print("✓ All tests passed for EnhancedTrackletAggregation!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

def test_integration():
    """Test basic integration between modules."""
    print("\nTesting module integration...")
    
    try:
        # Create sample data
        data = {
            'track_id': [1, 1, 1, 2, 2, 2],
            'jersey_number_detection': ['10', '10', '10', '9', '9', '9'],
            'jersey_number_confidence': [0.8, 0.9, 0.7, 0.6, 0.8, 0.7],
            'bbox_pitch': [{'x_bottom_middle': 0, 'y_bottom_middle': 0}] * 6
        }
        
        detections = pd.DataFrame(data)
        
        # Test that we can create instances and they have the right structure
        jersey_module = ImprovedJerseyRecognition(batch_size=2, device='cpu')
        agg_module = EnhancedTrackletAggregation({}, device='cpu')
        
        assert hasattr(jersey_module, 'process')
        assert hasattr(agg_module, 'process')
        
        print("✓ Module integration test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Improved Jersey Recognition System")
    print("=" * 50)
    
    tests = [
        test_improved_jersey_recognition,
        test_enhanced_tracklet_aggregation,
        test_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The improved jersey recognition system is ready.")
        print("\nTo use the improved system, run:")
        print("uv run tracklab -cn soccernet_improved")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
