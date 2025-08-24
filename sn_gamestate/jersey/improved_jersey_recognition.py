import pandas as pd
import torch
import numpy as np
from mmocr.apis import MMOCRInferencer
from mmocr.apis import TextDetInferencer, TextRecInferencer
from mmocr.utils import bbox2poly, crop_img, poly2bbox
import logging
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import cv2

from tracklab.utils.collate import default_collate, Unbatchable
from tracklab.pipeline.detectionlevel_module import DetectionLevelModule

log = logging.getLogger(__name__)


class ImprovedJerseyRecognition(DetectionLevelModule):
    """
    Improved Jersey Recognition - Replace frame-wise OCR with tracklet-level, 
    sequence-aware jersey recognition.
    
    This module processes sequences of frames for each tracklet to improve
    jersey number recognition accuracy and confidence.
    """
    input_columns = ["bbox_ltwh"]  # Only require bbox_ltwh like working version
    output_columns = ["jersey_number_detection", "jersey_number_confidence", "role_detection", "role"]
    collate_fn = default_collate

    def __init__(self, batch_size, device, tracking_dataset=None, 
                 sequence_length=5, min_confidence_threshold=0.3,
                 temporal_weight=0.7, spatial_weight=0.3,
                 use_confidence_boost=True, min_sequence_confidence=0.4,
                 enable_aggressive_boosting=True):
        
        super().__init__(batch_size=batch_size)
        self.device = device
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.min_confidence_threshold = min_confidence_threshold
        self.temporal_weight = temporal_weight
        self.spatial_weight = spatial_weight
        self.use_confidence_boost = use_confidence_boost
        self.min_sequence_confidence = min_sequence_confidence
        self.enable_aggressive_boosting = enable_aggressive_boosting
        
        # Debug logging to verify parameters
        log.info(f"ImprovedJerseyRecognition initialized with:")
        log.info(f"  - sequence_length: {self.sequence_length}")
        log.info(f"  - min_confidence_threshold: {self.min_confidence_threshold}")
        log.info(f"  - temporal_weight: {self.temporal_weight}")
        log.info(f"  - spatial_weight: {self.spatial_weight}")
        log.info(f"  - use_confidence_boost: {self.use_confidence_boost}")
        log.info(f"  - min_sequence_confidence: {self.min_sequence_confidence}")
        log.info(f"  - enable_aggressive_boosting: {self.enable_aggressive_boosting}")
        log.info(f"  - device: {self.device}")
        log.info(f"  - batch_size: {self.batch_size}")
        
        # Initialize MMOCR models - EXACTLY like working baseline
        log.info("Loading MMOCRInferencer...")
        self.ocr = MMOCRInferencer(det='dbnet_resnet18_fpnc_1200e_icdar2015', rec='SAR')
        
        log.info("Loading MMOCR text detection model...")
        self.textdetinferencer = TextDetInferencer(
            'dbnet_resnet18_fpnc_1200e_icdar2015', device=device)
        log.info("Loading MMOCR text recognition model...")
        self.textrecinferencer = TextRecInferencer('SAR', device=device)
        
        # Tracklet sequence storage for temporal aggregation
        self.tracklet_sequences = defaultdict(list)
        
        log.info("ImprovedJerseyRecognition initialization complete")

    def no_jersey_number(self):
        return None, 0

    def extract_numbers(self, text):
        """Extract numbers from text - EXACTLY like working baseline"""
        number = ''
        for char in text:
            if char.isdigit():
                number += char
        return number if number != '' else None

    def choose_best_jersey_number(self, jersey_numbers, jn_confidences):
        """Choose best jersey number - EXACTLY like working baseline"""
        if len(jersey_numbers) == 0:
            return self.no_jersey_number()
        else:
            jn_confidences = np.array(jn_confidences)
            idx_sort = np.argsort(jn_confidences)
            return jersey_numbers[idx_sort[-1]], jn_confidences[idx_sort[-1]]

    def extract_jersey_numbers_from_ocr(self, prediction):
        """Extract jersey numbers from OCR - EXACTLY like working baseline"""
        jersey_numbers = []
        jn_confidences = []
        for txt, conf in zip(prediction['rec_texts'], prediction['rec_scores']):
            jn = self.extract_numbers(txt)
            if jn is not None:
                jersey_numbers.append(jn)
                jn_confidences.append(conf)
        jersey_number, jn_confidence = self.choose_best_jersey_number(jersey_numbers, jn_confidences)
        if jersey_number is not None:
            jersey_number = jersey_number[:2]  # Only two-digit numbers are possible
        return jersey_number, jn_confidence

    @torch.no_grad()
    def preprocess(self, image, detection: pd.Series, metadata: pd.Series):
        """Preprocess - EXACTLY like working baseline"""
        l, t, r, b = detection.bbox.ltrb(
            image_shape=(image.shape[1], image.shape[0]), rounded=True
        )
        crop = image[t:b, l:r]
        if crop.shape[0] == 0 or crop.shape[1] == 0:
            crop = np.zeros((10, 10, 3), dtype=np.uint8)
        crop = Unbatchable([crop])
        batch = {
            "img": crop,
        }
        return batch

    def run_mmocr_inference(self, images_np):
        """Run MMOCR inference - EXACTLY like working baseline"""
        result = {}
        result['det'] = self.textdetinferencer(
            images_np,
            return_datasamples=True,
            batch_size=self.batch_size,
            progress_bar=False,
        )['predictions']

        result['rec'] = []
        for img, det_data_sample in zip(images_np, result['det']):
            det_pred = det_data_sample.pred_instances
            rec_inputs = []
            for polygon in det_pred['polygons']:
                # Roughly convert the polygon to a quadangle with 4 points
                quad = bbox2poly(poly2bbox(polygon)).tolist()
                rec_input = crop_img(img, quad)
                if rec_input.shape[0] == 0 or rec_input.shape[1] == 0:
                    continue
                rec_inputs.append(rec_input)
            result['rec'].append(
                self.textrecinferencer(
                    rec_inputs,
                    return_datasamples=True,
                    batch_size=self.batch_size,
                    progress_bar=False)['predictions'])

        pred_results = [{} for _ in range(len(result['rec']))]
        for i, rec_pred in enumerate(result['rec']):
            result_out = dict(rec_texts=[], rec_scores=[])
            for rec_pred_instance in rec_pred:
                rec_dict_res = self.textrecinferencer.pred2dict(rec_pred_instance)
                result_out['rec_texts'].append(rec_dict_res['text'])
                result_out['rec_scores'].append(rec_dict_res['scores'])
            pred_results[i].update(result_out)

        return pred_results

    def aggregate_tracklet_sequence(self, track_id: int, jersey_numbers: List, confidences: List, frame_indices: List) -> Tuple[Optional[str], float]:
        """Advanced temporal aggregation with aggressive confidence boosting and fallback strategies"""
        if not jersey_numbers or all(jn is None for jn in jersey_numbers):
            return None, 0.0
        
        # Filter out None values and low confidence detections
        valid_detections = [(jn, conf, idx) for jn, conf, idx in zip(jersey_numbers, confidences, frame_indices) 
                           if jn is not None and conf >= self.min_confidence_threshold]
        
        if not valid_detections:
            # Fallback: try with even lower threshold
            fallback_detections = [(jn, conf, idx) for jn, conf, idx in zip(jersey_numbers, confidences, frame_indices) 
                                 if jn is not None and conf >= 0.1]
            if fallback_detections:
                valid_detections = fallback_detections
            else:
                return None, 0.0
        
        # Group by jersey number with frequency analysis
        jn_groups = defaultdict(list)
        for jn, conf, idx in valid_detections:
            jn_groups[jn].append((conf, idx))
        
        # Find the most frequent jersey number
        best_jn = max(jn_groups.keys(), key=lambda x: len(jn_groups[x]))
        best_detections = jn_groups[best_jn]
        
        # Calculate temporal consistency score (frequency-based)
        temporal_score = len(best_detections) / len(jersey_numbers)
        
        # Calculate confidence metrics
        avg_confidence = np.mean([conf for conf, _ in best_detections])
        max_confidence = max([conf for conf, _ in best_detections])
        
        # Enhanced confidence boosting strategies
        boosted_confidence = avg_confidence
        
        if self.use_confidence_boost:
            # Strategy 1: Consistency-based boosting
            if temporal_score >= 0.5:  # 50% consistency threshold
                consistency_boost = min(0.3, temporal_score - 0.3)  # Max 0.3 boost
                boosted_confidence = min(1.0, avg_confidence + consistency_boost)
            
            # Strategy 2: High-confidence single detection boosting
            if max_confidence >= 0.8 and temporal_score >= 0.25:
                high_confidence_boost = min(0.2, max_confidence - 0.7)  # Boost for high confidence
                boosted_confidence = min(1.0, boosted_confidence + high_confidence_boost)
            
            # Strategy 3: Aggressive boosting for very high individual confidences
            if self.enable_aggressive_boosting and max_confidence >= 0.9:
                aggressive_boost = min(0.4, max_confidence - 0.8)  # Aggressive boost
                boosted_confidence = min(1.0, boosted_confidence + aggressive_boost)
        
        # Calculate sequence quality score
        sequence_quality = self._calculate_sequence_quality(jersey_numbers, confidences, frame_indices)
        
        # Enhanced final confidence calculation
        base_confidence = (
            self.temporal_weight * temporal_score + 
            self.spatial_weight * boosted_confidence
        )
        
        # Quality bonus for good sequences
        quality_bonus = 0.1 * sequence_quality
        
        # Consistency bonus for temporal consistency
        consistency_bonus = 0.05 * temporal_score if temporal_score >= 0.25 else 0
        
        final_confidence = base_confidence + quality_bonus + consistency_bonus
        
        # Apply minimum sequence confidence threshold
        if final_confidence < self.min_sequence_confidence:
            # Fallback to best single detection with confidence boost
            best_single = max(valid_detections, key=lambda x: x[1])
            fallback_confidence = min(1.0, best_single[1] * 1.1)  # 10% boost for fallback
            return best_single[0], fallback_confidence
        
        # Ensure confidence is in valid range
        final_confidence = np.clip(final_confidence, 0.0, 1.0)
        
        log.info(f"Tracklet {track_id} aggregation:")
        log.info(f"  - Jersey numbers: {jersey_numbers}")
        log.info(f"  - Confidences: {confidences}")
        log.info(f"  - Frame indices: {frame_indices}")
        log.info(f"  - Best jersey number: {best_jn}")
        log.info(f"  - Temporal score: {temporal_score:.3f}")
        log.info(f"  - Avg confidence: {avg_confidence:.3f}")
        log.info(f"  - Boosted confidence: {boosted_confidence:.3f}")
        log.info(f"  - Sequence quality: {sequence_quality:.3f}")
        log.info(f"  - Base confidence: {base_confidence:.3f}")
        log.info(f"  - Quality bonus: {quality_bonus:.3f}")
        log.info(f"  - Consistency bonus: {consistency_bonus:.3f}")
        log.info(f"  - Final confidence: {final_confidence:.3f}")
        
        return best_jn, final_confidence
    
    def _calculate_sequence_quality(self, jersey_numbers: List, confidences: List, frame_indices: List) -> float:
        """Calculate sequence quality based on consistency and confidence patterns"""
        if len(jersey_numbers) < 2:
            return 0.0
        
        # Calculate confidence stability (lower variance = higher quality)
        valid_confidences = [conf for conf in confidences if conf > 0]
        if len(valid_confidences) < 2:
            return 0.0
        
        confidence_variance = np.var(valid_confidences)
        confidence_stability = max(0, 1.0 - confidence_variance)
        
        # Calculate frame spacing consistency
        frame_spacings = [frame_indices[i+1] - frame_indices[i] for i in range(len(frame_indices)-1)]
        if frame_spacings:
            spacing_variance = np.var(frame_spacings)
            spacing_consistency = max(0, 1.0 - spacing_variance / 10.0)  # Normalize
        else:
            spacing_consistency = 1.0
        
        # Overall sequence quality
        sequence_quality = (confidence_stability * 0.7 + spacing_consistency * 0.3)
        return sequence_quality

    @torch.no_grad()
    def process(self, batch, detections: pd.DataFrame, metadatas: pd.DataFrame):
        """Process batch with improved temporal aggregation and fallback strategies"""
        jersey_number_detection = []
        jersey_number_confidence = []
        
        # Convert batch images to numpy
        images_np = [img.cpu().numpy() for img in batch['img']]
        del batch['img']
        
        # Run MMOCR inference using working baseline method
        predictions = self.run_mmocr_inference(images_np)
        
        # Process each detection
        for i, prediction in enumerate(predictions):
            jn, conf = self.extract_jersey_numbers_from_ocr(prediction)
            
            # Get track_id for this detection
            track_id = detections.iloc[i].get('track_id') if i < len(detections) else None
            
            if track_id is not None:
                # Store in tracklet sequence
                self.tracklet_sequences[track_id].append({
                    'jersey_number': jn,
                    'confidence': conf,
                    'frame_index': i
                })
                
                # Process sequence when we have enough frames
                if len(self.tracklet_sequences[track_id]) >= self.sequence_length:
                    # Extract sequence data
                    sequence_data = self.tracklet_sequences[track_id][-self.sequence_length:]
                    jersey_numbers = [item['jersey_number'] for item in sequence_data]
                    confidences = [item['confidence'] for item in sequence_data]
                    frame_indices = [item['frame_index'] for item in sequence_data]
                    
                    # Aggregate over sequence
                    final_jn, final_conf = self.aggregate_tracklet_sequence(
                        track_id, jersey_numbers, confidences, frame_indices
                    )
                    
                    # Apply post-processing confidence boost for very consistent detections
                    if final_jn is not None and final_conf > 0:
                        consistency_count = jersey_numbers.count(final_jn)
                        if consistency_count >= len(jersey_numbers) * 0.8:  # 80% consistency
                            final_conf = min(1.0, final_conf * 1.1)  # 10% boost
                    
                    jersey_number_detection.append(final_jn)
                    jersey_number_confidence.append(final_conf)
                    
                    # Remove oldest frame to maintain sequence length
                    if len(self.tracklet_sequences[track_id]) > self.sequence_length:
                        self.tracklet_sequences[track_id].pop(0)
                else:
                    # Use frame-level prediction for short sequences
                    jersey_number_detection.append(jn)
                    jersey_number_confidence.append(conf)
            else:
                # No track_id, use frame-level prediction
                jersey_number_detection.append(jn)
                jersey_number_confidence.append(conf)
        
        # Post-process results: apply confidence smoothing
        jersey_number_confidence = self._apply_confidence_smoothing(jersey_number_confidence)
        
        # Set outputs
        detections['jersey_number_detection'] = jersey_number_detection
        detections['jersey_number_confidence'] = jersey_number_confidence
        
        # Add role columns for downstream compatibility
        roles = []
        for jn in jersey_number_detection:
            if jn is not None and jn != '':
                try:
                    jn_int = int(jn)
                    if 1 <= jn_int <= 11:
                        roles.append('player')
                    else:
                        roles.append('player')
                except ValueError:
                    roles.append('player')
            else:
                roles.append('player')
        
        detections['role_detection'] = roles
        detections['role'] = roles
        
        return detections
    
    def _apply_confidence_smoothing(self, confidences: List[float]) -> List[float]:
        """Apply confidence smoothing to reduce noise and improve stability"""
        if len(confidences) < 3:
            return confidences
        
        smoothed = []
        for i in range(len(confidences)):
            if i == 0:
                # First element: average with next
                smoothed.append((confidences[i] + confidences[i+1]) / 2)
            elif i == len(confidences) - 1:
                # Last element: average with previous
                smoothed.append((confidences[i] + confidences[i-1]) / 2)
            else:
                # Middle elements: 3-point average
                smoothed.append((confidences[i-1] + confidences[i] + confidences[i+1]) / 3)
        
        return smoothed
