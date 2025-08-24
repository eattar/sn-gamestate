import pandas as pd
import torch
import numpy as np
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
    output_columns = ["jersey_number_detection", "jersey_number_confidence"]
    collate_fn = default_collate

    def __init__(self, batch_size, device, tracking_dataset=None, 
                 sequence_length=5, min_confidence_threshold=0.3,
                 temporal_weight=0.7, spatial_weight=0.3):
        super().__init__(batch_size=batch_size)
        self.device = device
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.min_confidence_threshold = min_confidence_threshold
        self.temporal_weight = temporal_weight
        self.spatial_weight = spatial_weight
        
        # Debug logging to verify parameters
        log.info(f"ImprovedJerseyRecognition initialized with:")
        log.info(f"  - sequence_length: {self.sequence_length}")
        log.info(f"  - min_confidence_threshold: {self.min_confidence_threshold}")
        log.info(f"  - temporal_weight: {self.temporal_weight}")
        log.info(f"  - spatial_weight: {self.spatial_weight}")
        log.info(f"  - device: {self.device}")
        log.info(f"  - batch_size: {self.batch_size}")
        
        # Initialize MMOCR models - match working version exactly
        try:
            from mmocr.apis import MMOCRInferencer
            log.info("Loading MMOCRInferencer...")
            # Don't pass device to MMOCRInferencer like working version
            self.ocr = MMOCRInferencer(det='dbnet_resnet18_fpnc_1200e_icdar2015', rec='SAR')
            self.use_mmocr_inferencer = True
            log.info("MMOCRInferencer loaded successfully")
        except Exception as e:
            log.warning(f"Failed to load MMOCRInferencer: {e}, falling back to separate inferencers")
            self.use_mmocr_inferencer = False
        
        # Always create separate inferencers like working version
        log.info("Loading MMOCR text detection model...")
        self.textdetinferencer = TextDetInferencer(
            'dbnet_resnet18_fpnc_1200e_icdar2015', device=device)
        log.info("Loading MMOCR text recognition model...")
        self.textrecinferencer = TextRecInferencer('SAR', device=device)
        log.info("MMOCR models loaded successfully")
        
        # Tracklet sequence storage
        self.tracklet_sequences = defaultdict(list)
        self.tracklet_jersey_history = defaultdict(list)
        
    def no_jersey_number(self):
        return None, 0.0

    def extract_numbers(self, text: str) -> Optional[str]:
        """Extract numeric characters from text."""
        number = ''
        for char in text:
            if char.isdigit():
                number += char
        return number if number != '' else None

    def validate_jersey_number(self, number: str) -> bool:
        """Validate if the extracted number is a valid jersey number."""
        if not number:
            return False
        # Jersey numbers are typically 1-99
        try:
            num = int(number)
            return 1 <= num <= 99
        except ValueError:
            return False

    def compute_temporal_consistency(self, jersey_numbers: List[str], 
                                   confidences: List[float]) -> float:
        """Compute temporal consistency score for a sequence of jersey numbers."""
        if not jersey_numbers:
            return 0.0
        
        # Count unique jersey numbers
        unique_numbers = set(jersey_numbers)
        if len(unique_numbers) == 1:
            # Perfect consistency
            return 1.0
        elif len(unique_numbers) == 2:
            # Check if one number appears much more frequently
            counts = {}
            for num in jersey_numbers:
                counts[num] = counts.get(num, 0) + 1
            
            max_count = max(counts.values())
            total_count = len(jersey_numbers)
            consistency = max_count / total_count
            return consistency
        else:
            # Low consistency
            return 0.2

    def compute_spatial_consistency(self, bboxes: List, confidences: List[float]) -> float:
        """Compute spatial consistency based on bbox positions and confidences."""
        if len(bboxes) < 2:
            return 1.0
        
        # Compute variance in bbox positions (normalized)
        centers = []
        for bbox in bboxes:
            if hasattr(bbox, 'center'):
                centers.append(bbox.center)
            else:
                # Fallback for different bbox formats
                centers.append((0, 0))  # Default center
        
        if len(centers) < 2:
            return 1.0
        
        # Simple spatial consistency based on center positions
        # Lower variance = higher consistency
        center_array = np.array(centers)
        variance = np.var(center_array, axis=0).sum()
        # Normalize variance to [0, 1] range
        spatial_consistency = max(0, 1 - variance / 1000)
        return spatial_consistency

    def aggregate_tracklet_jersey_simple(self, tracklet_id: int, 
                                       jersey_numbers: List[str], 
                                       confidences: List[float]) -> Tuple[str, float]:
        """Simplified aggregation of jersey numbers across a tracklet sequence (no spatial analysis)."""
        if not jersey_numbers:
            return self.no_jersey_number()
        
        # Filter out low confidence detections
        valid_indices = [i for i, conf in enumerate(confidences) 
                        if conf >= self.min_confidence_threshold]
        
        if not valid_indices:
            return self.no_jersey_number()
        
        filtered_numbers = [jersey_numbers[i] for i in valid_indices]
        filtered_confidences = [confidences[i] for i in valid_indices]
        
        # Compute temporal consistency
        temporal_consistency = self.compute_temporal_consistency(
            filtered_numbers, filtered_confidences)
        
        # Find the most frequent jersey number
        number_counts = defaultdict(int)
        for num in filtered_numbers:
            if self.validate_jersey_number(num):
                number_counts[num] += 1
        
        if not number_counts:
            return self.no_jersey_number()
        
        # Get the most frequent number
        most_frequent_number = max(number_counts.items(), key=lambda x: x[1])[0]
        
        # Compute final confidence based on frequency and temporal consistency
        frequency_score = number_counts[most_frequent_number] / len(filtered_numbers)
        final_confidence = (frequency_score * 0.7 + temporal_consistency * 0.3)
        
        # Ensure confidence is in [0, 1] range
        final_confidence = np.clip(final_confidence, 0.0, 1.0)
        
        return most_frequent_number, final_confidence

    def aggregate_tracklet_jersey(self, tracklet_id: int, 
                                jersey_numbers: List[str], 
                                confidences: List[float],
                                bboxes: List) -> Tuple[str, float]:
        """Aggregate jersey numbers across a tracklet sequence."""
        if not jersey_numbers:
            return self.no_jersey_number()
        
        # Filter out low confidence detections
        valid_indices = [i for i, conf in enumerate(confidences) 
                        if conf >= self.min_confidence_threshold]
        
        if not valid_indices:
            return self.no_jersey_number()
        
        filtered_numbers = [jersey_numbers[i] for i in valid_indices]
        filtered_confidences = [confidences[i] for i in valid_indices]
        filtered_bboxes = [bboxes[i] for i in valid_indices]
        
        # Compute consistency scores
        temporal_consistency = self.compute_temporal_consistency(
            filtered_numbers, filtered_confidences)
        spatial_consistency = self.compute_spatial_consistency(
            filtered_bboxes, filtered_confidences)
        
        # Weighted combination of consistency scores
        overall_consistency = (self.temporal_weight * temporal_consistency + 
                             self.spatial_weight * spatial_consistency)
        
        # Find the most frequent jersey number
        number_counts = defaultdict(int)
        for num in filtered_numbers:
            if self.validate_jersey_number(num):
                number_counts[num] += 1
        
        if not number_counts:
            return self.no_jersey_number()
        
        # Get the most frequent number
        most_frequent_number = max(number_counts.items(), key=lambda x: x[1])[0]
        
        # Compute final confidence based on frequency and consistency
        frequency_score = number_counts[most_frequent_number] / len(filtered_numbers)
        final_confidence = (frequency_score * 0.6 + overall_consistency * 0.4)
        
        # Ensure confidence is in [0, 1] range
        final_confidence = np.clip(final_confidence, 0.0, 1.0)
        
        return most_frequent_number, final_confidence

    @torch.no_grad()
    def preprocess(self, image, detection: pd.Series, metadata: pd.Series):
        """Preprocess detection for OCR."""
        l, t, r, b = detection.bbox.ltrb(
            image_shape=(image.shape[1], image.shape[0]), rounded=True
        )
        crop = image[t:b, l:r]
        
        # Debug logging for image cropping
        log.info(f"Original image shape: {image.shape}")
        log.info(f"Crop coordinates: l={l}, t={t}, r={r}, b={b}")
        log.info(f"Cropped image shape: {crop.shape}")
        
        if crop.shape[0] == 0 or crop.shape[1] == 0:
            log.warning(f"Empty crop detected, using fallback")
            crop = np.zeros((10, 10, 3), dtype=np.uint8)
        
        crop = Unbatchable([crop])
        batch = {
            "img": crop,
        }
        return batch

    def run_mmocr_inference(self, images_np: List[np.ndarray]) -> List[Dict]:
        """Run MMOCR inference on a batch of images."""
        if hasattr(self, 'use_mmocr_inferencer') and self.use_mmocr_inferencer:
            # Use MMOCRInferencer like the working version
            log.info("Using MMOCRInferencer for inference")
            try:
                predictions = self.ocr(images_np)['predictions']
                # Convert to standard format
                pred_results = []
                for pred in predictions:
                    result_out = dict(rec_texts=[], rec_scores=[])
                    if 'rec_texts' in pred and 'rec_scores' in pred:
                        result_out['rec_texts'] = pred['rec_texts']
                        result_out['rec_scores'] = pred['rec_scores']
                    pred_results.append(result_out)
                return pred_results
            except Exception as e:
                log.warning(f"MMOCRInferencer failed: {e}, falling back to separate inferencers")
                self.use_mmocr_inferencer = False
        
        # Fallback to separate inferencers
        log.info("Using separate TextDetInferencer and TextRecInferencer")
        
        # Text detection
        det_results = self.textdetinferencer(
            images_np,
            return_datasamples=True,
            batch_size=self.batch_size,
            progress_bar=False,
        )['predictions']

        # Debug logging for text detection
        log.info(f"Text detection found {len(det_results)} results")
        for i, det_data_sample in enumerate(det_results):
            det_pred = det_data_sample.pred_instances
            log.info(f"Image {i}: {len(det_pred['polygons'])} text regions detected")

        # Text recognition
        rec_results = []
        for img, det_data_sample in zip(images_np, det_results):
            det_pred = det_data_sample.pred_instances
            rec_inputs = []
            
            for polygon in det_pred['polygons']:
                quad = bbox2poly(poly2bbox(polygon)).tolist()
                rec_input = crop_img(img, quad)
                if rec_input.shape[0] == 0 or rec_input.shape[1] == 0:
                    continue
                rec_inputs.append(rec_input)
            
            log.info(f"Processing {len(rec_inputs)} text regions for recognition")
            
            if rec_inputs:
                rec_result = self.textrecinferencer(
                    rec_inputs,
                    return_datasamples=True,
                    batch_size=self.batch_size,
                    progress_bar=False)['predictions']
                rec_results.append(rec_result)
            else:
                rec_results.append([])

        # Convert to standard format
        pred_results = []
        for rec_pred in rec_results:
            result_out = dict(rec_texts=[], rec_scores=[])
            for rec_pred_instance in rec_pred:
                rec_dict_res = self.textrecinferencer.pred2dict(rec_pred_instance)
                result_out['rec_texts'].append(rec_dict_res['text'])
                result_out['rec_scores'].append(rec_dict_res['scores'])
            pred_results.append(result_out)

        return pred_results

    def extract_jersey_numbers_from_ocr(self, prediction: Dict) -> Tuple[Optional[str], float]:
        """Extract jersey number and confidence from OCR prediction."""
        jersey_numbers = []
        jn_confidences = []
        
        # Debug logging to see what OCR is detecting
        if len(prediction['rec_texts']) > 0:
            log.info(f"OCR detected text: {prediction['rec_texts']}")
            log.info(f"OCR confidence scores: {prediction['rec_scores']}")
        
        for txt, conf in zip(prediction['rec_texts'], prediction['rec_scores']):
            jn = self.extract_numbers(txt)
            log.info(f"Text: '{txt}' -> Extracted number: {jn}")
            # Temporarily remove strict validation to match working version
            if jn is not None:  # Remove validation temporarily
                log.info(f"Valid jersey number: {jn} with confidence {conf}")
                jersey_numbers.append(jn)
                jn_confidences.append(conf)
            else:
                log.info(f"Invalid jersey number: {jn} (validation failed)")
        
        if not jersey_numbers:
            log.info("No valid jersey numbers found")
            return self.no_jersey_number()
        
        # Return the highest confidence valid jersey number
        best_idx = np.argmax(jn_confidences)
        best_jn = jersey_numbers[best_idx]
        best_conf = jn_confidences[best_idx]
        
        # Limit to 2 digits like the working version
        if best_jn is not None:
            best_jn = best_jn[:2]
        
        log.info(f"Best jersey number: {best_jn} with confidence {best_conf}")
        return best_jn, best_conf

    @torch.no_grad()
    def process(self, batch, detections: pd.DataFrame, metadatas: pd.DataFrame):
        """Process a batch of detections with improved jersey recognition."""
        # Extract track IDs - handle case where they might not be available
        if 'track_id' in detections.columns:
            track_ids = detections['track_id'].values
            # Convert track IDs to integers if they're floats
            track_ids = [int(tid) if tid is not None and not pd.isna(tid) else None for tid in track_ids]
        else:
            log.warning("track_id not found in detections, using frame-level processing only")
            track_ids = [None] * len(detections)
        
        # Run OCR inference
        images_np = [img.cpu().numpy() for img in batch['img']]
        predictions = self.run_mmocr_inference(images_np)
        
        # Process each detection
        jersey_number_detection = []
        jersey_number_confidence = []
        
        for i, (prediction, track_id) in enumerate(zip(predictions, track_ids)):
            # Extract frame-level jersey number
            jn, conf = self.extract_jersey_numbers_from_ocr(prediction)
            
            # Store in tracklet sequence if we have a track ID
            if track_id is not None and not pd.isna(track_id):
                # Initialize tracklet sequence if it doesn't exist
                if track_id not in self.tracklet_sequences:
                    self.tracklet_sequences[track_id] = []
                
                # Store the frame-level result in tracklet sequence
                self.tracklet_sequences[track_id].append({
                    'jersey_number': jn,
                    'confidence': conf,
                    'frame_idx': i
                })
                
                # Debug logging for tracklet 1 to see what's being stored
                if track_id == 1:
                    log.info(f"Frame {i}: Stored jersey number {jn} with confidence {conf} for tracklet {track_id}")
                    log.info(f"  - Tracklet {track_id} now has {len(self.tracklet_sequences[track_id])} frames")
                    log.info(f"  - Current sequence: {[item['jersey_number'] for item in self.tracklet_sequences[track_id]]}")
                
                # Only process when we have exactly the required number of frames
                if len(self.tracklet_sequences[track_id]) == self.sequence_length:
                    sequence_data = self.tracklet_sequences[track_id]
                    jn_numbers = [item['jersey_number'] for item in sequence_data]
                    jn_confs = [item['confidence'] for item in sequence_data]
                    
                    # Debug logging for tracklet processing
                    if track_id == 1:  # Log for first tracklet only to avoid spam
                        log.info(f"Processing tracklet {track_id} with {len(sequence_data)} frames")
                        log.info(f"  - sequence_length threshold: {self.sequence_length}")
                        log.info(f"  - min_confidence_threshold: {self.min_confidence_threshold}")
                        log.info(f"  - temporal_weight: {self.temporal_weight}")
                        log.info(f"  - spatial_weight: {self.spatial_weight}")
                        log.info(f"  - Jersey numbers detected: {jn_numbers}")
                        log.info(f"  - Confidences: {jn_confs}")
                        log.info(f"  - Frame indices: {[item['frame_idx'] for item in sequence_data]}")
                    
                    # Aggregate at tracklet level
                    tracklet_jn, tracklet_conf = self.aggregate_tracklet_jersey_simple(
                        track_id, jn_numbers, jn_confs)
                    
                    if track_id == 1:  # Log results for first tracklet
                        log.info(f"  - Final jersey number: {tracklet_jn}")
                        log.info(f"  - Final confidence: {tracklet_conf}")
                    
                    # Use tracklet-level prediction for this frame
                    jersey_number_detection.append(tracklet_jn)
                    jersey_number_confidence.append(tracklet_conf)
                    
                    # Remove the oldest frame to maintain sequence length
                    self.tracklet_sequences[track_id].pop(0)
                    
                elif len(self.tracklet_sequences[track_id]) > self.sequence_length:
                    # This shouldn't happen, but if it does, remove oldest frame
                    self.tracklet_sequences[track_id].pop(0)
                    # Use frame-level prediction for now
                    jersey_number_detection.append(jn)
                    jersey_number_confidence.append(conf)
                else:
                    # Use frame-level prediction for now
                    if track_id == 1:  # Log for first tracklet
                        log.info(f"Tracklet {track_id} has only {len(self.tracklet_sequences[track_id])} frames, using frame-level prediction")
                        log.info(f"  - Current sequence: {[item['jersey_number'] for item in self.tracklet_sequences[track_id]]}")
                        log.info(f"  - Need {self.sequence_length} frames for tracklet processing")
                    jersey_number_detection.append(jn)
                    jersey_number_confidence.append(conf)
            else:
                # No track ID, use frame-level prediction
                jersey_number_detection.append(jn)
                jersey_number_confidence.append(conf)
        
        # Update detections
        detections['jersey_number_detection'] = jersey_number_detection
        detections['jersey_number_confidence'] = jersey_number_confidence
        
        return detections

    def reset_tracklet_sequences(self):
        """Reset tracklet sequences (useful for new video)."""
        self.tracklet_sequences.clear()
        self.tracklet_jersey_history.clear()
