# Improved Jersey Recognition Implementation

## Overview

This implementation replaces the baseline frame-wise OCR approach with a **tracklet-level, sequence-aware jersey recognition system** to improve the GS-HOTA score from the baseline 38%.

## Key Improvements

### 1. **Sequence-Aware Processing**
- Instead of processing each frame independently, the system now processes sequences of frames for each tracklet
- Uses temporal information to improve confidence and accuracy
- Configurable sequence length (default: 5 frames)

### 2. **Temporal Consistency**
- Computes temporal consistency scores across frame sequences
- Weights more recent frames higher using exponential decay
- Identifies stable jersey number predictions over time

### 3. **Spatial Consistency**
- Analyzes bbox positions across frames to detect movement patterns
- Higher confidence for players with consistent positioning
- Spatial threshold configurable (default: 10 meters)

### 4. **Enhanced Tracklet Aggregation**
- Sophisticated aggregation beyond simple voting
- Combines temporal, spatial, and confidence information
- Fallback mechanisms for short tracklets

## Architecture

### Core Components

1. **`ImprovedJerseyRecognition`** (`sn_gamestate/jersey/improved_jersey_recognition.py`)
   - Main jersey recognition module
   - Processes frame sequences for each tracklet
   - Integrates with MMOCR for text detection/recognition

2. **`EnhancedTrackletAggregation`** (`sn_gamestate/jersey/enhanced_tracklet_aggregation.py`)
   - Advanced aggregation module
   - Multi-factor weighting system
   - Handles edge cases and short tracklets

### Configuration Files

- **`improved_mmocr.yaml`**: Configuration for improved jersey recognition
- **`enhanced_jersey_aggregation.yaml`**: Configuration for enhanced aggregation
- **`soccernet_improved.yaml`**: Complete pipeline configuration

## Key Features

### Temporal Weighting
```python
# Exponential decay based on frame recency
weight = temporal_decay ** (max_frame - frame_idx)
```

### Multi-Factor Scoring
```python
combined_score = (avg_confidence * 0.4 + 
                 temporal_weight * 0.3 + 
                 spatial_weight * 0.2 + 
                 frequency_weight * 0.1)
```

### Adaptive Processing
- Long tracklets: Enhanced aggregation with temporal/spatial analysis
- Short tracklets: Fallback to simple majority voting
- No track ID: Frame-level processing

## Configuration Parameters

### Improved Jersey Recognition
- `sequence_length`: Number of frames to consider (default: 5)
- `min_confidence_threshold`: Minimum confidence for valid detections (default: 0.3)
- `temporal_weight`: Weight for temporal consistency (default: 0.7)
- `spatial_weight`: Weight for spatial consistency (default: 0.3)

### Enhanced Aggregation
- `min_tracklet_length`: Minimum frames for enhanced processing (default: 3)
- `confidence_threshold`: Final confidence threshold (default: 0.5)
- `temporal_decay`: Temporal decay factor (default: 0.9)
- `spatial_threshold`: Spatial consistency threshold in meters (default: 10.0)

## Usage

### 1. Run with Improved Configuration
```bash
uv run tracklab -cn soccernet_improved
```

### 2. Compare with Baseline
```bash
# Baseline
uv run tracklab -cn soccernet

# Improved version
uv run tracklab -cn soccernet_improved
```

## Expected Improvements

### GS-HOTA Score
- **Baseline**: 38%
- **Target**: >45% (significant improvement expected)

### Jersey Number Accuracy
- Better temporal consistency
- Reduced false positives from single-frame errors
- Improved confidence scoring

### Robustness
- Better handling of occlusions
- More stable predictions across frames
- Reduced impact of individual frame failures

## Technical Details

### Algorithm Flow
1. **Frame Processing**: Extract jersey numbers and confidence from each frame
2. **Sequence Building**: Maintain rolling window of frames for each tracklet
3. **Temporal Analysis**: Compute consistency across frame sequences
4. **Spatial Analysis**: Analyze bbox movement patterns
5. **Weighted Aggregation**: Combine multiple factors for final prediction
6. **Fallback Handling**: Graceful degradation for edge cases

### Memory Management
- Rolling window approach prevents memory bloat
- Automatic cleanup of old frame data
- Efficient data structures for large video processing

## Performance Considerations

### Computational Overhead
- Additional temporal/spatial analysis
- Slightly higher memory usage for sequence storage
- Minimal impact on inference speed

### Scalability
- Designed for batch processing
- Efficient pandas operations
- Configurable batch sizes

## Future Enhancements

### Potential Improvements
1. **Learning-based weighting**: Train optimal weights from data
2. **Multi-scale analysis**: Consider different temporal windows
3. **Attention mechanisms**: Focus on most informative frames
4. **Ensemble methods**: Combine multiple OCR models

### Integration Opportunities
1. **Pose information**: Use keypoint data for better spatial analysis
2. **Team context**: Leverage team affiliation for validation
3. **Game state**: Consider game context for prediction refinement

## Evaluation

### Metrics to Monitor
- GS-HOTA score improvement
- Jersey number accuracy per tracklet
- Confidence score calibration
- False positive reduction

### Validation Strategy
1. **A/B Testing**: Compare with baseline on validation set
2. **Error Analysis**: Identify failure cases
3. **Parameter Tuning**: Optimize configuration parameters
4. **Cross-validation**: Ensure robustness across different videos

## Conclusion

This improved jersey recognition system represents a significant advancement over the baseline frame-wise approach. By leveraging temporal and spatial information at the tracklet level, we expect substantial improvements in GS-HOTA scores and overall system robustness.

The modular design allows for easy experimentation and further refinement, while maintaining compatibility with the existing TrackLab framework.
