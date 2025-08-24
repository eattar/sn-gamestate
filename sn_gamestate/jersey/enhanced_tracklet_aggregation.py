import pandas as pd
import torch
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import logging

from tracklab.pipeline.videolevel_module import VideoLevelModule

log = logging.getLogger(__name__)


class EnhancedTrackletAggregation(VideoLevelModule):
    """
    Enhanced Tracklet Aggregation for Jersey Numbers.
    
    This module provides sophisticated aggregation of jersey numbers across tracklets,
    taking into account temporal consistency, spatial relationships, and confidence scores.
    """
    input_columns = ["track_id", "jersey_number_detection", "jersey_number_confidence", "bbox_pitch"]
    output_columns = ["jersey_number_final", "jersey_number_confidence_final"]

    def __init__(self, cfg, device, tracking_dataset=None):
        super().__init__()
        self.min_tracklet_length = cfg.get('min_tracklet_length', 3)
        self.confidence_threshold = cfg.get('confidence_threshold', 0.5)
        self.temporal_decay = cfg.get('temporal_decay', 0.9)
        self.spatial_threshold = cfg.get('spatial_threshold', 10.0)  # meters
        
    def compute_temporal_weight(self, frame_indices: List[int]) -> float:
        """Compute temporal weight based on frame recency."""
        if not frame_indices:
            return 0.0
        
        # Weight more recent frames higher
        max_frame = max(frame_indices)
        weights = []
        for frame_idx in frame_indices:
            # Exponential decay based on frame distance from most recent
            weight = self.temporal_decay ** (max_frame - frame_idx)
            weights.append(weight)
        
        return np.mean(weights)

    def compute_spatial_weight(self, bboxes: List) -> float:
        """Compute spatial weight based on bbox consistency."""
        if len(bboxes) < 2:
            return 1.0
        
        # Extract center points from bboxes
        centers = []
        for bbox in bboxes:
            if hasattr(bbox, 'center'):
                centers.append(bbox.center)
            elif isinstance(bbox, dict) and 'x_bottom_middle' in bbox:
                centers.append((bbox['x_bottom_middle'], bbox['y_bottom_middle']))
            else:
                centers.append((0, 0))  # Fallback
        
        # Compute spatial variance
        center_array = np.array(centers)
        if len(center_array) < 2:
            return 1.0
        
        # Calculate distance between consecutive centers
        distances = []
        for i in range(1, len(center_array)):
            dist = np.linalg.norm(center_array[i] - center_array[i-1])
            distances.append(dist)
        
        if not distances:
            return 1.0
        
        # Spatial weight based on movement consistency
        # Lower movement = higher weight
        avg_distance = np.mean(distances)
        spatial_weight = max(0, 1 - avg_distance / self.spatial_threshold)
        return spatial_weight

    def aggregate_jersey_numbers(self, tracklet_data: List[Dict]) -> Tuple[Optional[str], float]:
        """Aggregate jersey numbers for a tracklet with enhanced weighting."""
        if not tracklet_data:
            return None, 0.0
        
        # Group by jersey number
        jersey_groups = defaultdict(list)
        for item in tracklet_data:
            jn = item['jersey_number']
            if jn is not None:
                jersey_groups[jn].append(item)
        
        if not jersey_groups:
            return None, 0.0
        
        # Compute weighted scores for each jersey number
        jersey_scores = {}
        for jersey_num, items in jersey_groups.items():
            # Extract data for this jersey number
            confidences = [item['confidence'] for item in items]
            bboxes = [item['bbox'] for item in items]
            frame_indices = [item.get('frame_idx', 0) for item in items]
            
            # Compute weights
            temporal_weight = self.compute_temporal_weight(frame_indices)
            spatial_weight = self.compute_spatial_weight(bboxes)
            
            # Average confidence for this jersey number
            avg_confidence = np.mean(confidences)
            
            # Frequency weight (how often this number appears)
            frequency_weight = len(items) / len(tracklet_data)
            
            # Combined score
            combined_score = (avg_confidence * 0.4 + 
                            temporal_weight * 0.3 + 
                            spatial_weight * 0.2 + 
                            frequency_weight * 0.1)
            
            jersey_scores[jersey_num] = combined_score
        
        # Find the best jersey number
        if not jersey_scores:
            return None, 0.0
        
        best_jersey = max(jersey_scores.items(), key=lambda x: x[1])
        return best_jersey[0], best_jersey[1]

    @torch.no_grad()
    def process(self, detections: pd.DataFrame, metadatas: pd.DataFrame):
        """Process detections to aggregate jersey numbers at tracklet level."""
        # Initialize output columns
        detections["jersey_number_final"] = [np.nan] * len(detections)
        detections["jersey_number_confidence_final"] = [0.0] * len(detections)
        
        if "track_id" not in detections.columns:
            return detections
        
        # Group by track_id
        for track_id in detections.track_id.unique():
            if pd.isna(track_id):
                continue
                
            tracklet = detections[detections.track_id == track_id]
            
            # Skip short tracklets
            if len(tracklet) < self.min_tracklet_length:
                # Use simple voting for short tracklets
                jersey_numbers = tracklet.jersey_number_detection.dropna()
                jn_confidences = tracklet.jersey_number_confidence.dropna()
                
                if len(jersey_numbers) > 0:
                    # Simple majority voting
                    from collections import Counter
                    jersey_counts = Counter(jersey_numbers)
                    most_common_jersey = jersey_counts.most_common(1)[0][0]
                    
                    # Average confidence for this jersey number
                    jersey_confidences = [conf for jn, conf in zip(jersey_numbers, jn_confidences) 
                                        if jn == most_common_jersey]
                    avg_confidence = np.mean(jersey_confidences) if jersey_confidences else 0.0
                    
                    detections.loc[tracklet.index, "jersey_number_final"] = most_common_jersey
                    detections.loc[tracklet.index, "jersey_number_confidence_final"] = avg_confidence
                continue
            
            # Prepare tracklet data for enhanced aggregation
            tracklet_data = []
            for _, row in tracklet.iterrows():
                tracklet_data.append({
                    'jersey_number': row.get('jersey_number_detection'),
                    'confidence': row.get('jersey_number_confidence', 0.0),
                    'bbox': row.get('bbox_pitch'),
                    'frame_idx': row.name  # Use index as frame identifier
                })
            
            # Enhanced aggregation
            final_jersey, final_confidence = self.aggregate_jersey_numbers(tracklet_data)
            
            if final_jersey is not None and final_confidence >= self.confidence_threshold:
                detections.loc[tracklet.index, "jersey_number_final"] = final_jersey
                detections.loc[tracklet.index, "jersey_number_confidence_final"] = final_confidence
            else:
                # Fallback to best individual detection
                best_idx = tracklet.jersey_number_confidence.idxmax()
                best_jersey = tracklet.loc[best_idx, 'jersey_number_detection']
                best_confidence = tracklet.loc[best_idx, 'jersey_number_confidence']
                
                detections.loc[tracklet.index, "jersey_number_final"] = best_jersey
                detections.loc[tracklet.index, "jersey_number_confidence_final"] = best_confidence
        
        return detections
