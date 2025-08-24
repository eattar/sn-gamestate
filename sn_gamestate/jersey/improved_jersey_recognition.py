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
        """Aggregate jersey numbers over tracklet sequence using temporal consistency"""
        if not jersey_numbers or all(jn is None for jn in jersey_numbers):
            return None, 0.0
        
        # Filter out None values and low confidence detections
        valid_detections = [(jn, conf, idx) for jn, conf, idx in zip(jersey_numbers, confidences, frame_indices) 
                           if jn is not None and conf >= self.min_confidence_threshold]
        
        if not valid_detections:
            return None, 0.0
        
        # Group by jersey number
        jn_groups = defaultdict(list)
        for jn, conf, idx in valid_detections:
            jn_groups[jn].append((conf, idx))
        
        # Find the most frequent jersey number
        best_jn = max(jn_groups.keys(), key=lambda x: len(jn_groups[x]))
        best_detections = jn_groups[best_jn]
        
        # Calculate temporal consistency score
        temporal_score = len(best_detections) / len(jersey_numbers)
        
        # Calculate average confidence for the best jersey number
        avg_confidence = np.mean([conf for conf, _ in best_detections])
        
        # Combine temporal consistency and confidence
        final_confidence = (self.temporal_weight * temporal_score + 
                          self.spatial_weight * avg_confidence)
        
        log.info(f"Tracklet {track_id} aggregation:")
        log.info(f"  - Jersey numbers: {jersey_numbers}")
        log.info(f"  - Confidences: {confidences}")
        log.info(f"  - Frame indices: {frame_indices}")
        log.info(f"  - Best jersey number: {best_jn}")
        log.info(f"  - Final confidence: {final_confidence:.3f}")
        
        return best_jn, final_confidence

    @torch.no_grad()
    def process(self, batch, detections: pd.DataFrame, metadatas: pd.DataFrame):
        """Process batch with temporal aggregation"""
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
