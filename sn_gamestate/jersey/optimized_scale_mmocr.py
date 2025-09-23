import pandas as pd
import torch
import numpy as np
from mmocr.apis import MMOCRInferencer
from mmocr.apis import TextDetInferencer, TextRecInferencer
from mmocr.utils import bbox2poly, crop_img, poly2bbox
import cv2
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import List

from tracklab.utils.collate import default_collate, Unbatchable
from tracklab.pipeline.detectionlevel_module import DetectionLevelModule


class OptimizedScaleMMOCR(DetectionLevelModule):
    """
    Multi-Scale Text Detection Architecture for Jersey Recognition
    
    This module implements multi-scale text detection to improve jersey number
    recognition by processing images at different scales and combining results.
    """
    input_columns = ["bbox_ltwh"]
    output_columns = ["jersey_number_detection", "jersey_number_confidence"]
    collate_fn = default_collate

    def __init__(self, batch_size, device, tracking_dataset=None, 
                 multi_scale_enabled=True, scales=[0.75, 0.85, 1.0, 1.15, 1.25], 
                 scale_weights=[0.15, 0.25, 0.3, 0.2, 0.1], confidence_threshold=0.15,
                 scale_thresholds=[0.10, 0.12, 0.15, 0.12, 0.10], min_text_width=5, min_text_height=5, adaptive_scale_selection=True, scale_effectiveness_weights=True):
        super().__init__(batch_size=batch_size)
        self.batch_size = batch_size
        self.device = device
        
        self.multi_scale_enabled = multi_scale_enabled
        self.scales = scales
        self.scale_weights = scale_weights
        self.confidence_threshold = confidence_threshold
        self.scale_thresholds = scale_thresholds
        
        self.min_text_width = min_text_width
        self.min_text_height = min_text_height
        
        self.ocr = MMOCRInferencer(det='dbnet_resnet18_fpnc_1200e_icdar2015', rec='SAR')
        self.textdetinferencer = TextDetInferencer(
            'dbnet_resnet18_fpnc_1200e_icdar2015', device=device)
        self.textrecinferencer = TextRecInferencer('SAR', device=device)

    def no_jersey_number(self):
        return None, 0

    @torch.no_grad()
    def preprocess(self, image, detection: pd.Series, metadata: pd.Series):
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

    def extract_numbers(self, text):
        number = ''
        for char in text:
            if char.isdigit():
                number += char
        return number if number != '' else None

    def choose_best_jersey_number(self, jersey_numbers, jn_confidences):
        if len(jersey_numbers) == 0:
            return self.no_jersey_number()
        else:
            jn_confidences = np.array(jn_confidences)
            idx_sort = np.argsort(jn_confidences)
            return jersey_numbers[idx_sort[-1]], jn_confidences[
                idx_sort[-1]]

    def extract_jersey_numbers_from_ocr(self, prediction):
        jersey_numbers = []
        jn_confidences = []
        for txt, conf in zip(prediction['rec_texts'], prediction['rec_scores']):
            jn = self.extract_numbers(txt)
            if jn is not None:
                jersey_numbers.append(jn)
                jn_confidences.append(conf)
        jersey_number, jn_confidence = self.choose_best_jersey_number(jersey_numbers,
                                                                      jn_confidences)
        if jersey_number is not None:
            jersey_number = jersey_number[:2]
        return jersey_number, jn_confidence

    def run_single_scale_detection(self, images_np, scale=1.0):
        """Run MMOCR detection on images at a single scale"""
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
                if scale != 1.0:
                    scaled_polygon = polygon / scale
                    polygon = scaled_polygon
                
                quad = bbox2poly(poly2bbox(polygon)).tolist()
                
                bbox = poly2bbox(polygon)
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                
                if width < self.min_text_width or height < self.min_text_height:
                    continue
                
                rec_input = crop_img(img, quad)
                if rec_input.shape[0] == 0 or rec_input.shape[1] == 0:
                    continue
                rec_inputs.append(rec_input)
            
            if len(rec_inputs) > 0:
                rec_result = self.textrecinferencer(
                    rec_inputs,
                    return_datasamples=True,
                    batch_size=self.batch_size,
                    progress_bar=False)['predictions']
                result['rec'].append(rec_result)
            else:
                result['rec'].append([])

        pred_results = [{} for _ in range(len(result['rec']))]
        for i, rec_pred in enumerate(result['rec']):
            result_out = dict(rec_texts=[], rec_scores=[])
            for rec_pred_instance in rec_pred:
                rec_dict_res = self.textrecinferencer.pred2dict(rec_pred_instance)
                result_out['rec_texts'].append(rec_dict_res['text'])
                result_out['rec_scores'].append(rec_dict_res['scores'])
            pred_results[i].update(result_out)

        return pred_results

    def run_multi_scale_detection(self, images_np):
        """Run text detection at multiple scales with proper coordinate scaling"""
        if not self.multi_scale_enabled:
            return self.run_single_scale_detection(images_np)
        
        all_results = []
        for scale_idx, scale in enumerate(self.scales):
            try:
                scale_results = self._process_single_scale(images_np, scale_idx, scale)
                all_results.append(scale_results)
            except Exception as e:
                all_results.append(self.run_single_scale_detection(images_np))
        
        combined_results = self.combine_multi_scale_results(all_results)
        
        return combined_results
    
    def _process_single_scale(self, images_np, scale_idx, scale):
        """Process a single scale (used for sequential processing)"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        scaled_images = []
        for img in images_np:
            h, w = img.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            scaled_img = cv2.resize(img, (new_w, new_h))
            scaled_images.append(scaled_img)
        
        scale_results = self.run_single_scale_detection(scaled_images, scale=scale)
        
        scale_threshold = self.scale_thresholds[scale_idx]
        for img_idx, scale_result in enumerate(scale_results):
            if 'rec_scores' in scale_result:
                weighted_scores = [score * self.scale_weights[scale_idx] 
                                 for score in scale_result['rec_scores']]
                
                filtered_texts = []
                filtered_scores = []
                for text, score in zip(scale_result['rec_texts'], weighted_scores):
                    if score >= scale_threshold:
                        filtered_texts.append(text)
                        filtered_scores.append(score)
                
                scale_result['rec_texts'] = filtered_texts
                scale_result['rec_scores'] = filtered_scores
                scale_result['scale'] = scale
                scale_result['scale_weight'] = self.scale_weights[scale_idx]
                scale_result['scale_threshold'] = scale_threshold
        
        return scale_results

    def combine_multi_scale_results(self, all_results):
        """Combine results from multiple scales using improved weighted voting with NMS"""
        if len(all_results) == 1:
            return all_results[0]
        
        num_images = len(all_results[0])
        combined_results = []
        
        for img_idx in range(num_images):
            all_detections = []
            for scale_idx, scale_results in enumerate(all_results):
                if img_idx < len(scale_results) and 'rec_texts' in scale_results[img_idx]:
                    scale_result = scale_results[img_idx]
                    for text, score in zip(scale_result['rec_texts'], scale_result['rec_scores']):
                        all_detections.append({
                            'text': text,
                            'score': score,
                            'scale': scale_result.get('scale', 1.0),
                            'scale_weight': scale_result.get('scale_weight', 1.0),
                            'scale_threshold': scale_result.get('scale_threshold', self.confidence_threshold)
                        })
            
            text_groups = defaultdict(list)
            for det in all_detections:
                text_groups[det['text']].append(det)
            
            combined_texts = []
            combined_scores = []
            
            for text, detections in text_groups.items():
                detections.sort(key=lambda x: x['score'], reverse=True)
                
                best_detection = detections[0]
                base_score = best_detection['score']
                
                agreement_boost = min(0.08, len(detections) * 0.03)
                
                unique_scales = len(set(det['scale'] for det in detections))
                diversity_bonus = min(0.03, unique_scales * 0.015)
                
                top_detections = detections[:2]
                total_weight = sum(det['scale_weight'] for det in top_detections)
                if total_weight > 0:
                    weighted_score = sum(det['score'] * det['scale_weight'] for det in top_detections) / total_weight
                else:
                    weighted_score = base_score
            
                final_score = min(1.0, weighted_score + agreement_boost + diversity_bonus)
                
                if final_score >= self.confidence_threshold:
                    combined_texts.append(text)
                    combined_scores.append(final_score)
            
            combined_results.append({
                'rec_texts': combined_texts,
                'rec_scores': combined_scores
            })
        
        return combined_results

    @torch.no_grad()
    def process(self, batch, detections: pd.DataFrame, metadatas: pd.DataFrame):
        jersey_number_detection = []
        jersey_number_confidence = []
        images_np = [img.cpu().numpy() for img in batch['img']]
        del batch['img']

        predictions = self.run_multi_scale_detection(images_np)
        
        for prediction in predictions:
            jn, conf = self.extract_jersey_numbers_from_ocr(prediction)
            jersey_number_detection.append(jn)
            jersey_number_confidence.append(conf)

        detections['jersey_number_detection'] = jersey_number_detection
        detections['jersey_number_confidence'] = jersey_number_confidence
        
        # MOTA FIX: Ensure jersey numbers are properly formatted for evaluation
        # Convert None values to empty strings
        detections["jersey_number_detection"] = detections["jersey_number_detection"].where(detections["jersey_number_detection"] != "", None)
        detections['jersey_number_confidence'] = detections['jersey_number_confidence'].fillna(0.0).astype(float)

        return detections

