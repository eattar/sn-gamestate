#!/usr/bin/env python3
"""
Ball Action Spotting Integration with SN-GameState
===================================================

This script integrates ball-action-spotting with sn-gamestate player tracking
to output ball actions performed by a specific player (identified by team and jersey number).

Workflow:
    1. Run SN-GameState tracking on image frames (native SoccerNetGS format)
    2. Convert image frames to video for ball-action-spotting
    3. Run ball-action detection on the video
    4. Match ball actions to specific player's detections
    5. Output results as JSON

Usage:
    python run_player_ball_actions.py --game SNGS-001 --team left --jersey 10

Requirements:
    - SN-GameState installed and configured
    - Ball-action-spotting models in ../ball-action-spotting/data/ball_action/experiments/
    - SoccerNetGS dataset with image frames
    - ffmpeg installed (for frame-to-video conversion)

Output:
    JSON file with format:
    {
        "player": {"team": "left", "jersey": 10},
        "actions": [
            {"action": "PASS", "time": "2:14", "frame": 3350, "confidence": 0.876},
            {"action": "DRIVE", "time": "5:32", "frame": 8300, "confidence": 0.823}
        ]
    }
"""

import argparse
import json
import sys
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import pickle
import gzip
from tqdm import tqdm

# Add ball-action-spotting to path
BALL_ACTION_DIR = Path(__file__).parent.parent / "ball-action-spotting"
sys.path.insert(0, str(BALL_ACTION_DIR))

from src.predictors import MultiDimStackerPredictor
from src.utils import get_best_model_path, get_video_info
from src.ball_action import constants as ball_constants


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Extract ball actions for a specific player by team and jersey number",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Step 1: Run tracklab to generate tracker state (do this first!)
  uv run tracklab -cn soccernet dataset.eval_set=valid dataset.vids_dict.valid=[SNGS-001]
  
  # Step 2: Run ball-action integration with the tracker state
  python run_player_ball_actions.py \
    --game SNGS-001 \
    --split valid \
    --team left \
    --jersey 10 \
    --state-cache outputs/sn-gamestate/YYYY-MM-DD/HH-MM-SS/states/sn-gamestate.pklz
  
  # With custom output file
  python run_player_ball_actions.py \
    --game SNGS-001 \
    --split valid \
    --team right \
    --jersey 7 \
    --state-cache tracking_state.pklz \
    --output player7_actions.json
  
  # Specify frames directory manually
  python run_player_ball_actions.py \
    --game SNGS-001 \
    --split valid \
    --team left \
    --jersey 10 \
    --state-cache tracking_state.pklz \
    --frames-dir /path/to/SoccerNetGS/valid/SNGS-001
        """
    )
    
    parser.add_argument('--game', required=True, type=str,
                        help='Game name (e.g., SNGS-001) or path to video file')
    parser.add_argument('--split', type=str, default='valid',
                        choices=['train', 'valid', 'test', 'challenge'],
                        help='Dataset split (default: valid)')
    parser.add_argument('--team', required=True, choices=['left', 'right'],
                        help='Player team (left or right)')
    parser.add_argument('--jersey', required=True, type=int,
                        help='Player jersey number')
    parser.add_argument('--output', type=str,
                        help='Output JSON file (default: player_<jersey>_<team>_actions.json)')
    parser.add_argument('--state-cache', type=str, required=True,
                        help='Path to cached tracking state .pklz file (REQUIRED - run tracklab separately first)')
    parser.add_argument('--frames-dir', type=str,
                        help='Path to directory with image frames (auto-detected if not provided)')
    parser.add_argument('--experiment', type=str, default='ball_finetune_long_004',
                        help='Ball-action experiment name (default: ball_finetune_long_004)')
    parser.add_argument('--fold', type=int, default=5,
                        help='Model fold number (default: 5, which has 90.1%% accuracy)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device for ball-action model (default: cuda:0)')
    parser.add_argument('--save-video', type=str,
                        help='Save the converted video to this path (optional)')
    
    return parser.parse_args()


def convert_frames_to_video(frames_dir: Path, output_video: Path, fps: int = 25) -> bool:
    """
    Convert image frames to video using ffmpeg
    
    Args:
        frames_dir: Directory containing image frames
        output_video: Output video file path
        fps: Frames per second (default: 25)
        
    Returns:
        True if successful, False otherwise
    """
    print("\n" + "="*60)
    print("Converting Image Frames to Video")
    print("="*60)
    print(f"  Input: {frames_dir}")
    print(f"  Output: {output_video}")
    print(f"  FPS: {fps}")
    
    # Check if ffmpeg is available
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("\n❌ ERROR: ffmpeg not found. Please install ffmpeg:")
        print("  Ubuntu/Debian: sudo apt-get install ffmpeg")
        print("  macOS: brew install ffmpeg")
        print("  Or download from: https://ffmpeg.org/download.html")
        return False
    
    # Find image files
    image_files = sorted(frames_dir.glob('*.jpg'))
    if not image_files:
        image_files = sorted(frames_dir.glob('*.png'))
    
    if not image_files:
        print(f"\n❌ ERROR: No image files found in {frames_dir}")
        return False
    
    print(f"  Found {len(image_files)} frames")
    
    # Build ffmpeg command
    # Use pattern matching for numbered frames
    input_pattern = str(frames_dir / "%06d.jpg") if image_files[0].suffix == '.jpg' else str(frames_dir / "%06d.png")
    
    cmd = [
        'ffmpeg',
        '-framerate', str(fps),
        '-i', input_pattern,
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-y',  # Overwrite output file
        str(output_video)
    ]
    
    print("\n▶ Running ffmpeg...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"\n❌ ffmpeg failed: {result.stderr}")
            return False
        
        print(f"✓ Video created: {output_video}")
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR during video conversion: {e}")
        return False


def load_tracking_state(state_path: str, game_name: str = None) -> pd.DataFrame:
    """
    Load tracking state from .pklz file
    
    Args:
        state_path: Path to .pklz file (can be single game or multi-game archive)
        game_name: Game ID (e.g., "SNGS-021") for multi-game archives
    
    Returns:
        DataFrame with detections
    """
    print(f"\n📂 Loading tracking state from: {state_path}")
    
    import zipfile
    
    try:
        # First try gzip (single-game state files)
        with gzip.open(state_path, 'rb') as f:
            tracker_state = pickle.load(f)
            
        # Check if it's a TrackerState object or DataFrame
        if hasattr(tracker_state, 'detections_pred'):
            detections = tracker_state.detections_pred
        else:
            detections = tracker_state
            
    except gzip.BadGzipFile:
        # Multi-game archive (zip file with individual game pickle files)
        with zipfile.ZipFile(state_path, 'r') as zf:
            filenames = zf.namelist()
            
            # Extract game number from game_name (e.g., SNGS-021 -> 021)
            if game_name:
                game_num = game_name.split('-')[-1]  # Get "021" from "SNGS-021"
                pkl_filename = f"{game_num}.pkl"
                
                if pkl_filename not in filenames:
                    raise ValueError(f"Game {game_name} (file {pkl_filename}) not found in archive. Available games: {[f for f in filenames if f.endswith('.pkl') and not f.endswith('_image.pkl')]}")
                
                print(f"  Loading game {game_name} from {pkl_filename}")
                with zf.open(pkl_filename, 'r') as f:
                    detections = pickle.load(f)
            else:
                # No game specified, load first pickle file
                pkl_files = [f for f in filenames if f.endswith('.pkl') and not f.endswith('_image.pkl')]
                if not pkl_files:
                    raise ValueError(f"No pickle files found in archive")
                
                print(f"  No game specified, loading first game: {pkl_files[0]}")
                with zf.open(pkl_files[0], 'r') as f:
                    detections = pickle.load(f)
    
    print(f"✓ Loaded {len(detections)} detections")
    
    return detections


def filter_player_by_jersey(detections: pd.DataFrame, team: str, jersey: int) -> pd.DataFrame:
    """
    Filter detections for specific player by team and jersey number
    
    Args:
        detections: Full tracking detections dataframe
        team: 'left' or 'right'
        jersey: Jersey number
        
    Returns:
        Filtered dataframe for the specific player
    """
    print("\n" + "="*60)
    print(f"STEP 2: Filtering Player (Team: {team}, Jersey: {jersey})")
    print("="*60)
    
    # Debug: Print available columns
    print(f"Available columns: {list(detections.columns)}")
    
    # Check available teams and jerseys
    if 'team' in detections.columns:
        available_teams = detections['team'].dropna().unique()
        print(f"Available teams: {list(available_teams)}")
    
    # Try different possible jersey column names
    jersey_col = None
    for col_name in ['jn_tracklet', 'jersey_number', 'jersey', 'jn']:
        if col_name in detections.columns:
            jersey_col = col_name
            break
    
    if jersey_col:
        available_jerseys = detections[jersey_col].dropna().unique()
        print(f"Available jersey numbers: {sorted([int(j) for j in available_jerseys if pd.notna(j)])}")
        print(f"Using jersey column: '{jersey_col}'")
    else:
        print(f"⚠️  Warning: No jersey column found in detections")
        print(f"   Cannot filter by jersey number")
    
    # Filter by team and jersey
    if jersey_col:
        # Convert jersey column to int for comparison (handles float/string issues)
        detections_filtered = detections.copy()
        detections_filtered[jersey_col] = pd.to_numeric(detections_filtered[jersey_col], errors='coerce')
        
        player_dets = detections_filtered[
            (detections_filtered['team'] == team) & 
            (detections_filtered[jersey_col] == jersey)
        ].copy()
        
        # Debug: show what we're filtering
        team_count = len(detections_filtered[detections_filtered['team'] == team])
        jersey_count = len(detections_filtered[detections_filtered[jersey_col] == jersey])
        print(f"Detections with team='{team}': {team_count}")
        print(f"Detections with jersey={jersey}: {jersey_count}")
    else:
        # Fall back to team only
        player_dets = detections[detections['team'] == team].copy()
        print(f"⚠️  Filtering by team only (no jersey column available)")
    
    if len(player_dets) == 0:
        print(f"\n❌ ERROR: No player found with team='{team}' and jersey={jersey}")
        print("\nAvailable combinations:")
        
        if 'team' in detections.columns and jersey_col and jersey_col in detections.columns:
            team_jersey = detections[['team', jersey_col]].dropna().drop_duplicates()
            for _, row in team_jersey.iterrows():
                print(f"  - Team: {row['team']}, Jersey: {int(row[jersey_col])}")
        
        sys.exit(1)
    
    # Get unique track IDs for this player
    track_ids = player_dets['track_id'].unique()
    
    print(f"\n✓ Found player:")
    print(f"  - Detections: {len(player_dets)}")
    print(f"  - Track IDs: {list(track_ids)}")
    print(f"  - Frame range: {player_dets['image_id'].min()} - {player_dets['image_id'].max()}")
    
    return player_dets


def run_ball_action_detection(video_path: str, experiment: str, fold: int, device: str) -> List[Dict]:
    """
    Run ball-action detection on video
    
    Args:
        video_path: Path to video file
        experiment: Experiment name
        fold: Model fold number
        device: Device for inference
        
    Returns:
        List of detected actions with frame, class, and confidence
    """
    print("\n" + "="*60)
    print("STEP 3: Running Ball Action Detection")
    print("="*60)
    
    # Get model path
    experiment_dir = BALL_ACTION_DIR / "data" / "ball_action" / "experiments" / experiment / f"fold_{fold}"
    model_path = get_best_model_path(experiment_dir)
    
    if model_path is None:
        print(f"\n❌ ERROR: No model found in {experiment_dir}")
        print(f"\nPlease ensure ball-action-spotting models are downloaded to:")
        print(f"  {BALL_ACTION_DIR / 'data' / 'ball_action' / 'experiments'}")
        sys.exit(1)
    
    print(f"\n📦 Model: {model_path}")
    print(f"   Experiment: {experiment}, Fold: {fold}")
    
    # Load predictor
    predictor = MultiDimStackerPredictor(model_path, device=device, tta=True)
    
    # Get video info
    video_info = get_video_info(video_path)
    print(f"\n🎬 Video: {video_info['frame_count']} frames @ {video_info['fps']} fps")
    
    # Run inference
    from src.frame_fetchers import NvDecFrameFetcher, OpencvFrameFetcher
    
    try:
        frame_fetcher = NvDecFrameFetcher(video_path, gpu_id=int(device.split(':')[-1]) if ':' in device else 0)
        print("   Using NvDec for video decoding")
    except:
        frame_fetcher = OpencvFrameFetcher(video_path, gpu_id=int(device.split(':')[-1]) if ':' in device else 0)
        print("   Using OpenCV for video decoding")
    
    frame_fetcher.num_frames = video_info['frame_count']
    
    # Predict actions
    print("\n▶ Detecting ball actions...")
    indexes_generator = predictor.indexes_generator
    INDEX_SAVE_ZONE = 1
    min_frame_index = indexes_generator.clip_index(0, video_info['frame_count'], INDEX_SAVE_ZONE)
    max_frame_index = indexes_generator.clip_index(video_info['frame_count'], video_info['frame_count'], INDEX_SAVE_ZONE)
    
    frame_index2prediction = {}
    predictor.reset_buffers()
    
    with tqdm(total=video_info['frame_count'], desc="Processing frames") as pbar:
        while True:
            frame = frame_fetcher.fetch_frame()
            frame_index = frame_fetcher.current_index
            prediction, predict_index = predictor.predict(frame, frame_index)
            
            if predict_index < min_frame_index:
                continue
            
            if prediction is not None:
                frame_index2prediction[predict_index] = prediction.cpu().numpy()
            
            pbar.update(1)
            
            if predict_index == max_frame_index:
                break
    
    predictor.reset_buffers()
    
    # Convert predictions to actions
    frame_indexes = sorted(frame_index2prediction.keys())
    raw_predictions = np.stack([frame_index2prediction[i] for i in frame_indexes], axis=0)
    
    # Determine number of classes from model output
    num_model_classes = raw_predictions.shape[1]
    print(f"   Model outputs {num_model_classes} classes")
    
    # Post-process to get action events
    from src.utils import post_processing
    
    actions = []
    # Only process classes that the model actually outputs
    for cls_name, cls_idx in ball_constants.class2target.items():
        if cls_idx >= num_model_classes:
            # Skip classes not in model output
            continue
            
        action_frames, action_confidences = post_processing(
            frame_indexes, 
            raw_predictions[:, cls_idx],
            **ball_constants.postprocess_params
        )
        
        for frame, conf in zip(action_frames, action_confidences):
            actions.append({
                'frame': int(frame),
                'action': cls_name,
                'confidence': float(conf)
            })
    
    # Sort by frame
    actions.sort(key=lambda x: x['frame'])
    
    print(f"\n✓ Detected {len(actions)} ball actions")
    print(f"   Action types: {set(a['action'] for a in actions)}")
    
    return actions


def match_actions_to_player(actions: List[Dict], player_dets: pd.DataFrame, 
                            window_frames: int = 50, min_confidence: float = 0.7,
                            min_time_between_actions: float = 2.0, fps: float = 25.0,
                            frames_dir: Optional[Path] = None) -> List[Dict]:
    """
    Match detected actions to player using temporal proximity and spatial overlap
    
    Args:
        actions: List of detected actions
        player_dets: Player detections dataframe
        window_frames: Frame window for matching (±N frames)
        min_confidence: Minimum confidence threshold for actions
        min_time_between_actions: Minimum seconds between consecutive actions (filters false positives)
        fps: Frames per second (for time-based filtering)
        frames_dir: Directory containing image frames (for ball verification)
        
    Returns:
        List of matched actions with player info
    """
    print("\n" + "="*60)
    print("STEP 4: Matching Actions to Player")
    print("="*60)
    print(f"   Using ±{window_frames} frame window for matching")
    print(f"   Minimum confidence: {min_confidence}")
    print(f"   Minimum time between actions: {min_time_between_actions}s")
    
    # Filter by confidence first
    filtered_actions = [a for a in actions if a['confidence'] >= min_confidence]
    print(f"   Actions after confidence filter: {len(filtered_actions)}/{len(actions)}")
    
    if len(filtered_actions) == 0:
        print("   ⚠️  No actions passed confidence threshold - try lowering min_confidence")
        return []
    
    print(f"\n   Actions to match:")
    for i, a in enumerate(filtered_actions[:10], 1):  # Show first 10
        print(f"     {i}. Frame {a['frame']}: {a['action']} (conf={a['confidence']:.3f})")
    if len(filtered_actions) > 10:
        print(f"     ... and {len(filtered_actions) - 10} more")
    
    matched_actions = []
    
    # Convert image_id to numeric for comparison if it's a string
    # image_id format is often like "2021000001" (game_id + frame_num)
    player_dets_work = player_dets.copy()
    if player_dets_work['image_id'].dtype == 'object':
        # Extract frame number from image_id (last 6 digits typically)
        # Format: GGGG0000FFF where G=game, F=frame
        player_dets_work['frame_num'] = player_dets_work['image_id'].astype(str).str[-6:].astype(int)
    else:
        player_dets_work['frame_num'] = pd.to_numeric(player_dets_work['image_id'], errors='coerce')
    
    # Assume 1920x1080 resolution for spatial filtering
    frame_width = 1920
    frame_height = 1080
    center_x = frame_width / 2
    center_y = frame_height / 2
    
    # Initialize Ball Detector (YOLOv8)
    print("Initializing Ball Detector for verification...")
    try:
        from src.player_tracking.detector import PlayerDetector
        # Class 32 is 'sports ball' in COCO dataset
        ball_detector = PlayerDetector(model_name='yolov8n.pt', classes=[32], conf_threshold=0.15, device='cpu')
        has_ball_detector = True
        print("   ✓ Ball detector ready")
    except Exception as e:
        print(f"   ⚠️  Could not initialize ball detector: {e}")
        print("   Continuing with spatial filtering only.")
        has_ball_detector = False
        ball_detector = None

    for action in tqdm(filtered_actions, desc="Matching actions"):
        action_frame = action['frame']
        
        # Get player detections within time window
        nearby_dets = player_dets_work[
            (player_dets_work['frame_num'] >= action_frame - window_frames) &
            (player_dets_work['frame_num'] <= action_frame + window_frames)
        ]
        
        if len(nearby_dets) == 0:
            print(f"\nAction at frame {action_frame} ({action['action']}, conf={action['confidence']:.3f}): No player detections in ±{window_frames} frame window")
        
        if len(nearby_dets) > 0:
            # Find closest detection by frame
            nearby_dets = nearby_dets.copy()
            nearby_dets['frame_diff'] = abs(nearby_dets['frame_num'] - action_frame)
            
            # Add spatial scoring
            def calculate_spatial_score(bbox):
                if pd.isna(bbox).any(): return 0.0
                left, top, width, height = bbox
                bbox_center_x = left + width / 2
                bbox_center_y = top + height / 2
                dist_x = abs(bbox_center_x - center_x)
                dist_y = abs(bbox_center_y - center_y)
                return max(0, 1 - (dist_x / 400 + dist_y / 300) / 2)
            
            nearby_dets['spatial_score'] = nearby_dets['bbox_ltwh'].apply(calculate_spatial_score)
            
            # Combined score
            nearby_dets['combined_score'] = 0.6 * nearby_dets['spatial_score'] + 0.4 * (1.0 - (nearby_dets['frame_diff'] / 10.0).clip(0, 1))
            
            # Get best match candidate
            closest_det = nearby_dets.loc[nearby_dets['combined_score'].idxmax()]
            
            # ---------------------------------------------------------
            # BALL PROXIMITY VERIFICATION
            # ---------------------------------------------------------
            ball_verified = False
            ball_dist = float('inf')
            
            if has_ball_detector and frames_dir and frames_dir.exists():
                # Construct frame path (assuming standard SoccerNet structure)
                # Image IDs are usually like "2021000001" -> we need to map frame number to filename
                # Simple heuristic: frame number formatted with leading zeros? 
                # Or just list dir and find matching number?
                # SoccerNetGS usually has 00001.jpg or similar
                
                # Try to find the image file
                # We know action_frame is an integer (e.g. 298)
                # We need to find the file that corresponds to this frame
                # Let's try a few common formats
                candidates = [
                    frames_dir / f"{action_frame:06d}.jpg",
                    frames_dir / f"{action_frame:06d}.png",
                    frames_dir / f"{action_frame}.jpg",
                    frames_dir / f"img1/{action_frame:06d}.jpg"
                ]
                
                frame_path = None
                for c in candidates:
                    if c.exists():
                        frame_path = c
                        break
                
                if frame_path:
                    import cv2
                    frame_img = cv2.imread(str(frame_path))
                    if frame_img is not None:
                        ball_dets = ball_detector.detect(frame_img)
                        
                        if ball_dets:
                            # Get player bbox
                            p_bbox = closest_det['bbox_ltwh'] # [left, top, width, height]
                            p_center = (p_bbox[0] + p_bbox[2]/2, p_bbox[1] + p_bbox[3]/2)
                            
                            # Find closest ball
                            min_dist = float('inf')
                            for b in ball_dets:
                                b_center = b.center
                                # Euclidean distance
                                d = ((p_center[0] - b_center[0])**2 + (p_center[1] - b_center[1])**2)**0.5
                                if d < min_dist:
                                    min_dist = d
                            
                            ball_dist = min_dist
                            # Threshold: 150 pixels (approx 1-2 meters in 1080p)
                            if ball_dist < 150:
                                ball_verified = True
                        else:
                            # No ball detected - ambiguous
                            ball_dist = -1 
                
            # ---------------------------------------------------------
            
            print(f"\nAction at frame {action_frame} ({action['action']}, conf={action['confidence']:.3f}):")
            print(f"  Best match: frame_diff={int(closest_det['frame_diff'])}, spatial={closest_det['spatial_score']:.3f}")
            if has_ball_detector:
                status = "✅ VERIFIED" if ball_verified else ("❌ TOO FAR" if ball_dist > 0 else "⚠️ NO BALL DETECTED")
                print(f"  Ball Distance: {ball_dist:.1f} px -> {status}")
            
            # DECISION LOGIC
            # 1. If ball verified: MATCH!
            # 2. If no ball detected but spatial score is very high (>0.6): MATCH (fallback)
            # 3. Otherwise: REJECT
            
            is_match = False
            if ball_verified:
                is_match = True
            elif ball_dist == -1 and closest_det['spatial_score'] > 0.6 and closest_det['frame_diff'] <= 5:
                # Fallback if ball detection failed but player is dead center and perfectly synced
                is_match = True
                print("  -> Accepted by spatial fallback (ball not detected)")
            
            if is_match:
                matched_actions.append({
                    'action': action['action'],
                    'frame': action_frame,
                    'confidence': action['confidence'],
                    'player_frame': int(closest_det['frame_num']),
                    'frame_offset': int(closest_det['frame_diff']),
                    'spatial_score': float(closest_det['spatial_score']),
                    'ball_distance': float(ball_dist) if ball_dist != float('inf') else None
                })
            else:
                print(f"  ❌ Filtered out")
    
    for action in tqdm(filtered_actions, desc="Matching actions"):
        action_frame = action['frame']
        
        # Get player detections within time window
        nearby_dets = player_dets_work[
            (player_dets_work['frame_num'] >= action_frame - window_frames) &
            (player_dets_work['frame_num'] <= action_frame + window_frames)
        ]
        
        if len(nearby_dets) == 0:
            print(f"\nAction at frame {action_frame} ({action['action']}, conf={action['confidence']:.3f}): No player detections in ±{window_frames} frame window")
        
        if len(nearby_dets) > 0:
            # Find closest detection by frame
            nearby_dets = nearby_dets.copy()
            nearby_dets['frame_diff'] = abs(nearby_dets['frame_num'] - action_frame)
            
            # Add spatial scoring - actions typically happen in center of frame
            # bbox_ltwh format: [left, top, width, height]
            def calculate_spatial_score(bbox):
                """Calculate how central the player is (0-1, higher = more central)"""
                if pd.isna(bbox).any():
                    return 0.0
                left, top, width, height = bbox
                # Calculate center of bbox
                bbox_center_x = left + width / 2
                bbox_center_y = top + height / 2
                # Distance from frame center
                dist_x = abs(bbox_center_x - center_x)
                dist_y = abs(bbox_center_y - center_y)
                # Normalize to 0-1 (closer to center = higher score)
                # Use reasonable thresholds: 400px horizontal, 300px vertical
                spatial_score = max(0, 1 - (dist_x / 400 + dist_y / 300) / 2)
                return spatial_score
            
            nearby_dets['spatial_score'] = nearby_dets['bbox_ltwh'].apply(calculate_spatial_score)
            
            # Combined score: temporal (frame_diff) + spatial (centrality)
            # Normalize frame_diff to 0-1 scale (0 frames = 1.0, 10 frames = 0.0)
            nearby_dets['temporal_score'] = 1.0 - (nearby_dets['frame_diff'] / 10.0).clip(0, 1)
            # Combined: 60% spatial, 40% temporal (spatial is more important for distinguishing players)
            nearby_dets['combined_score'] = 0.6 * nearby_dets['spatial_score'] + 0.4 * nearby_dets['temporal_score']
            
            # Get best match by combined score
            closest_det = nearby_dets.loc[nearby_dets['combined_score'].idxmax()]
            
            # Debug: print filtering decisions
            print(f"\nAction at frame {action_frame} ({action['action']}, conf={action['confidence']:.3f}):")
            print(f"  Best match: frame_diff={int(closest_det['frame_diff'])}, spatial={closest_det['spatial_score']:.3f}, combined={closest_det['combined_score']:.3f}")
            
            # Only match if:
            # 1. Player is close in time (within 10 frames)
            # 2. Player has decent spatial score (>0.6 = VERY central)
            # NOTE: Increased threshold to 0.6 to be much stricter
            if closest_det['frame_diff'] <= 10 and closest_det['spatial_score'] > 0.6:
                matched_actions.append({
                    'action': action['action'],
                    'frame': action_frame,
                    'confidence': action['confidence'],
                    'player_frame': int(closest_det['frame_num']),
                    'frame_offset': int(closest_det['frame_diff']),
                    'spatial_score': float(closest_det['spatial_score']),
                    'combined_score': float(closest_det['combined_score'])
                })
            else:
                print(f"  ❌ Filtered out: spatial < 0.6 or frame_diff > 10")
    
    # Filter by minimum time between actions (remove rapid-fire detections)
    if len(matched_actions) > 1:
        filtered_matched = [matched_actions[0]]  # Keep first action
        min_frame_gap = int(min_time_between_actions * fps)
        
        for action in matched_actions[1:]:
            last_frame = filtered_matched[-1]['frame']
            if action['frame'] - last_frame >= min_frame_gap:
                filtered_matched.append(action)
        
        removed = len(matched_actions) - len(filtered_matched)
        if removed > 0:
            print(f"   Removed {removed} actions (too close together)")
        matched_actions = filtered_matched
    
    print(f"\n✓ Matched {len(matched_actions)} actions to player")
    print(f"   Unmatched: {len(filtered_actions) - len(matched_actions)} (player not visible or too close in time)")
    
    return matched_actions


def format_time(frame: int, fps: float) -> str:
    """Convert frame number to MM:SS format"""
    seconds = int(frame / fps)
    minutes = seconds // 60
    secs = seconds % 60
    return f"{minutes}:{secs:02d}"


def create_output_json(matched_actions: List[Dict], team: str, jersey: int, 
                      video_name: str, fps: float = 25.0) -> Dict:
    """Create output JSON structure"""
    # Add timestamps
    for action in matched_actions:
        action['time'] = format_time(action['frame'], fps)
    
    # Create output
    output = {
        'player': {
            'team': team,
            'jersey': jersey
        },
        'video': video_name,
        'fps': fps,
        'total_actions': len(matched_actions),
        'action_counts': {},
        'actions': matched_actions
    }
    
    # Count actions by type
    for action in matched_actions:
        action_type = action['action']
        output['action_counts'][action_type] = output['action_counts'].get(action_type, 0) + 1
    
    return output


def main():
    """Main execution"""
    args = parse_args()
    
    print("\n" + "="*70)
    print(" Ball Action Spotting - Player Actions Extraction")
    print("="*70)
    print(f"  Game/Video: {args.game}")
    print(f"  Split: {args.split}")
    print(f"  Target Player: Team={args.team}, Jersey=#{args.jersey}")
    print(f"  Ball-Action Model: {args.experiment}/fold_{args.fold}")
    print("="*70)
    
    # Determine if it's a game name or video path
    game_path = Path(args.game)
    if game_path.exists() and game_path.suffix in ['.mp4', '.avi', '.mov']:
        # It's a video file
        video_path = game_path
        game_name = None
    else:
        # It's a game name - construct path from config
        game_name = args.game
        video_path = None
    
    # Step 1: Load tracking detections from pre-computed state
    print("\n📂 Loading pre-computed tracking state...")
    print("   (Run tracklab separately first to generate this file)")
    
    if not Path(args.state_cache).exists():
        print(f"\n❌ ERROR: Tracker state file not found: {args.state_cache}")
        print("\nTo generate tracker state, run tracklab first:")
        print(f"  cd /workspace/sn-gamestate")
        print(f"  uv run tracklab -cn soccernet dataset.eval_set={args.split} dataset.vids_dict.{args.split}=[{game_name}]")
        print(f"\nThis will create a .pklz file in the outputs directory")
        sys.exit(1)
    
    detections = load_tracking_state(args.state_cache, game_name)
    
    # Get frames directory
    if args.frames_dir:
        frames_dir = Path(args.frames_dir)
    else:
        # Default path based on standard SoccerNetGS structure
        # Assumes data is at /netscratch/eattar/ds/SoccerNet/2024/data/SoccerNetGS
        # Frames are in img1 subdirectory
        data_dir = Path("/netscratch/eattar/ds/SoccerNet/2024/data/SoccerNetGS")
        frames_dir = data_dir / args.split / game_name / "img1"
        print(f"  Frames directory (auto-detected): {frames_dir}")
    
    # Step 1.5: Convert frames to video for ball-action-spotting
    if video_path is None and game_name:
        # Need to convert frames to video
        if not frames_dir.exists():
            print(f"\n❌ ERROR: Frames directory not found: {frames_dir}")
            print(f"   Please specify --frames-dir or ensure data is at default location")
            sys.exit(1)
        
        # Create temporary video file
        temp_video = Path(tempfile.gettempdir()) / f"{game_name}_temp.mp4"
        print(f"\n📹 Converting frames to video for ball-action detection...")
        
        if not convert_frames_to_video(frames_dir, temp_video, fps=25):
            print(f"\n❌ ERROR: Failed to convert frames to video")
            sys.exit(1)
        
        video_path = temp_video
        
        # Save video if requested
        if args.save_video:
            import shutil
            save_path = Path(args.save_video)
            shutil.copy(temp_video, save_path)
            print(f"✓ Video saved to: {save_path}")
            cleanup_video = False  # Don't delete if we saved a copy
        else:
            cleanup_video = True
    else:
        cleanup_video = False
    
    # Step 2: Filter player by jersey
    player_dets = filter_player_by_jersey(detections, args.team, args.jersey)
    
    # Step 3: Run ball action detection
    actions = run_ball_action_detection(str(video_path), args.experiment, args.fold, args.device)
    
    # Step 4: Match actions to player
    matched_actions = match_actions_to_player(
        actions, player_dets,
        window_frames=50,
        min_confidence=0.75,  # Higher threshold to reduce false positives
        min_time_between_actions=3.0,  # At least 3 seconds between actions
        fps=25.0,
        frames_dir=frames_dir
    )
    
    # Step 5: Create output
    print("\n" + "="*60)
    print("STEP 5: Creating Output")
    print("="*60)
    
    output = create_output_json(matched_actions, args.team, args.jersey, 
                               game_name or str(video_path), fps=25.0)
    
    # Determine output filename
    if args.output:
        output_file = args.output
    else:
        output_file = f"player_{args.jersey}_{args.team}_actions.json"
    
    # Save output
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    print(f"  Player: Team {args.team}, Jersey #{args.jersey}")
    print(f"  Total Actions: {output['total_actions']}")
    print(f"\n  Action Breakdown:")
    for action_type, count in sorted(output['action_counts'].items()):
        print(f"    - {action_type}: {count}")
    
    if matched_actions:
        print(f"\n  Sample Actions:")
        for action in matched_actions[:5]:
            print(f"    - {action['time']} | {action['action']} (confidence: {action['confidence']:.3f})")
        
        if len(matched_actions) > 5:
            print(f"    ... and {len(matched_actions) - 5} more")
    
    print("\n" + "="*60)
    print("✓ Processing Complete!")
    print("="*60)
    
    # Cleanup temporary video if created
    if cleanup_video and video_path.exists():
        print(f"\n🧹 Cleaning up temporary video: {video_path}")
        video_path.unlink()


if __name__ == '__main__':
    main()
