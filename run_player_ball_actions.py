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
    parser.add_argument('--state-cache', type=str,
                        help='Path to cached tracking state .pklz file (required if not using --use-ground-truth-detections)')
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
    parser.add_argument('--analyze-data', action='store_true',
                        help='Run data quality analysis on tracking state and exit')
    parser.add_argument('--validate-detections', action='store_true',
                        help='Validate tracking detections against ground truth labels')
    parser.add_argument('--data-dir', type=str, default='/netscratch/eattar/ds/SoccerNet/2024/data/SoccerNetGS',
                        help='Root directory of SoccerNetGS dataset (default: /netscratch/eattar/ds/SoccerNet/2024/data/SoccerNetGS)')
    parser.add_argument('--labels-path', type=str,
                        help='Direct path to Labels-GameState.json (overrides data-dir + split + game logic)')
    parser.add_argument('--analyze-jerseys', action='store_true',
                        help='Analyze per-track jersey stability and majority vote reassignment (diagnostic)')
    parser.add_argument('--show-all-actions', action='store_true',
                        help='Show all detected actions with nearest player attribution (for verification)')
    parser.add_argument('--use-ground-truth-detections', action='store_true',
                        help='Use player detections directly from ground truth JSON instead of tracker state')
    parser.add_argument('--ball-confidence', type=float, default=0.25,
                        help='YOLO confidence threshold for ball detection (default: 0.25)')
    parser.add_argument('--ball-max-height', type=float, default=0.55,
                        help='Maximum height ratio in frame (0-1) where ball can be detected (default: 0.55, lower=ground)')
    parser.add_argument('--ball-min-size', type=int, default=18,
                        help='Minimum ball dimension in pixels (default: 18)')
    parser.add_argument('--ball-max-size', type=int, default=100,
                        help='Maximum ball dimension in pixels (default: 100)')
    parser.add_argument('--ball-search-window', type=int, default=5,
                        help='Number of frames to search around action (default: 5, ±5 frames)')
    parser.add_argument('--max-ball-distance', type=int, default=150,
                        help='Maximum distance in pixels between player and ball (default: 150)')
    
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
        
        if str(team).lower() == 'nan':
            print(f"Filtering for team='nan'...")
            player_dets = detections_filtered[
                (detections_filtered['team'].astype(str) == 'nan') & 
                (detections_filtered[jersey_col] == jersey)
            ].copy()
        else:
            player_dets = detections_filtered[
                (detections_filtered['team'] == team) & 
                (detections_filtered[jersey_col] == jersey)
            ].copy()
        
        # Debug: show what we're filtering
        if str(team).lower() == 'nan':
            team_count = len(detections_filtered[detections_filtered['team'].astype(str) == 'nan'])
        else:
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
        print("\nAvailable combinations (from tracking detections):")
        
        if 'team' in detections.columns and jersey_col and jersey_col in detections.columns:
            team_jersey = detections[['team', jersey_col]].dropna().drop_duplicates()
            # Sort by team then jersey
            team_jersey = team_jersey.sort_values(['team', jersey_col])
            for _, row in team_jersey.iterrows():
                # Check if this detection was validated
                is_validated = ''
                if 'gt_matched' in detections.columns:
                    validated_combo = detections[
                        (detections['team'] == row['team']) & 
                        (detections[jersey_col] == row[jersey_col]) &
                        (detections['gt_matched'] == True) &
                        (detections['gt_correct_jersey'] == True) &
                        (detections['gt_correct_team'] == True)
                    ]
                    if len(validated_combo) > 0:
                        is_validated = ' ✓ (verified by ground truth)'
                    else:
                        is_validated = ' ⚠️  (not verified - may be tracker error)'
                
                print(f"  - Team: {row['team']}, Jersey: {int(row[jersey_col])}{is_validated}")
        
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
                            frames_dir: Optional[Path] = None,
                            ball_confidence: float = 0.25,
                            ball_max_height: float = 0.55,
                            ball_min_size: int = 18,
                            ball_max_size: int = 100,
                            ball_search_window: int = 5,
                            max_ball_distance: int = 150) -> List[Dict]:
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
        from ultralytics import YOLO
        # Class 32 is 'sports ball' in COCO dataset
        # Using 'm' (medium) model for better small object detection
        ball_detector = YOLO('yolov8m.pt')
        has_ball_detector = True
        print("   ✓ Ball detector ready")
    except Exception as e:
        print(f"   ⚠️  Could not initialize ball detector: {e}")
        print("   Actions will be rejected without ball verification.")
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
            
            # ---------------------------------------------------------
            # BALL PROXIMITY VERIFICATION & CANDIDATE SELECTION
            # ---------------------------------------------------------
            # We do this BEFORE picking the best candidate, so we can use ball proximity
            # to resolve duplicates (e.g. if there are 3 players with Jersey 11, pick the one with the ball)
            
            # New logic: Search for ball in a window around the action
            best_ball_info = {'dist': float('inf'), 'frame': None}
            search_range = range(action_frame - ball_search_window, action_frame + ball_search_window + 1)

            if has_ball_detector and frames_dir and frames_dir.exists():
                for frame_to_check in search_range:
                    # Find player's detection at this specific frame
                    player_at_frame = nearby_dets[nearby_dets['frame_num'] == frame_to_check]
                    if player_at_frame.empty:
                        continue
                    
                    player_bbox_ltwh = player_at_frame.iloc[0]['bbox_ltwh']
                    if pd.isna(player_bbox_ltwh).any():
                        continue
                    
                    player_center = (player_bbox_ltwh[0] + player_bbox_ltwh[2] / 2, 
                                     player_bbox_ltwh[1] + player_bbox_ltwh[3] / 2)

                    # Find and check the image file for this frame
                    frame_path_candidates = [
                        frames_dir / f"{frame_to_check:06d}.jpg",
                        frames_dir / f"{frame_to_check:06d}.png",
                        frames_dir / f"img1/{frame_to_check:06d}.jpg"
                    ]
                    frame_path = next((c for c in frame_path_candidates if c.exists()), None)

                    if not frame_path:
                        continue

                    # Run YOLO detection with lower confidence to catch more candidates
                    results = ball_detector(str(frame_path), classes=[32], conf=ball_confidence, verbose=False)
                    if results and len(results) > 0 and len(results[0].boxes) > 0:
                        num_detections = len(results[0].boxes)
                        filtered_reasons = []
                        print(f"    Frame {frame_to_check}: {num_detections} raw YOLO detections")
                        
                        # Prepare debug image for this frame
                        debug_img = None
                        img_height = frame_height  # Default
                        try:
                            import cv2
                            debug_img = cv2.imread(str(frame_path))
                            if debug_img is not None:
                                img_height = debug_img.shape[0]
                                # Draw player bbox (green)
                                p_x1, p_y1 = int(player_bbox_ltwh[0]), int(player_bbox_ltwh[1])
                                p_x2, p_y2 = int(player_bbox_ltwh[0] + player_bbox_ltwh[2]), int(player_bbox_ltwh[1] + player_bbox_ltwh[3])
                                cv2.rectangle(debug_img, (p_x1, p_y1), (p_x2, p_y2), (0, 255, 0), 2)
                        except:
                            pass
                        
                        for b in results[0].boxes:
                            b_xywh = b.xywh[0].cpu().numpy()
                            b_x, b_y, b_w, b_h = float(b_xywh[0]), float(b_xywh[1]), float(b_xywh[2]), float(b_xywh[3])
                            
                            # Soccer-specific Filter 1: Height constraint (ball is usually on ground)
                            # Normalize Y coordinate to 0-1 range (0=top, 1=bottom)
                            height_ratio = b_y / img_height
                            if height_ratio < (1.0 - ball_max_height):  # Ball too high in frame
                                filtered_reasons.append(f"height={height_ratio:.2f}")
                                continue
                            
                            # Soccer-specific Filter 2: Size constraints (consistent ball size)
                            if b_w < ball_min_size or b_h < ball_min_size:
                                filtered_reasons.append(f"too_small={min(b_w,b_h):.0f}px")
                                continue  # Too small
                            if b_w > ball_max_size or b_h > ball_max_size:
                                filtered_reasons.append(f"too_large={max(b_w,b_h):.0f}px")
                                continue  # Too large (likely not a ball)
                            
                            # Soccer-specific Filter 3: Aspect ratio (ball should be roughly circular)
                            aspect_ratio = b_w / b_h if b_h > 0 else 999
                            if aspect_ratio > 1.35 or aspect_ratio < 0.74:  # Tighter: shoes are often elongated
                                filtered_reasons.append(f"aspect={aspect_ratio:.2f}")
                                continue
                            
                            b_center = (b_x, b_y)
                            d = ((player_center[0] - b_center[0])**2 + (player_center[1] - b_center[1])**2)**0.5
                            
                            # Draw ball bbox on debug image if available
                            if debug_img is not None:
                                passes_filters = (
                                    height_ratio >= (1.0 - ball_max_height) and
                                    b_w >= ball_min_size and b_h >= ball_min_size and
                                    b_w <= ball_max_size and b_h <= ball_max_size and
                                    aspect_ratio <= 1.35 and aspect_ratio >= 0.74
                                )
                                b_x1, b_y1 = int(b_x - b_w/2), int(b_y - b_h/2)
                                b_x2, b_y2 = int(b_x + b_w/2), int(b_y + b_h/2)
                                # Blue for valid ball, Red for filtered out
                                color = (255, 0, 0) if passes_filters else (0, 0, 255)
                                cv2.rectangle(debug_img, (b_x1, b_y1), (b_x2, b_y2), color, 2)
                            
                            if d < best_ball_info['dist']:
                                best_ball_info['dist'] = d
                                best_ball_info['frame'] = frame_to_check
                        
                        # Save debug image for this frame
                        if debug_img is not None:
                            try:
                                img_filename = f"frame_{action_frame}_{action['action']}_search_f{frame_to_check}.jpg"
                                cv2.imwrite(img_filename, debug_img)
                                print(f"      Saved: {img_filename}")
                            except:
                                pass
                        
                        # Log filtering results for this frame
                        if frame_to_check == action_frame and num_detections > 0:
                            passed = num_detections - len(filtered_reasons)
                            if passed == 0:
                                print(f"    Frame {frame_to_check}: {num_detections} detections, all filtered: {', '.join(filtered_reasons[:3])}")
                            else:
                                print(f"    Frame {frame_to_check}: {num_detections} detections, {passed} passed filters")
            
            # Evaluate all candidates
            candidates = []
            for idx, row in nearby_dets.iterrows():
                # 1. Spatial Score
                if pd.isna(row['bbox_ltwh']).any():
                    spatial_score = 0.0
                else:
                    left, top, width, height = row['bbox_ltwh']
                    bbox_center = (left + width / 2, top + height / 2)
                    dist_x = abs(bbox_center[0] - center_x)
                    dist_y = abs(bbox_center[1] - center_y)
                    spatial_score = max(0, 1 - (dist_x / 800 + dist_y / 500) / 2)
                
                # 2. Temporal Score
                frame_diff = abs(row['frame_num'] - action_frame)
                # Clamp ratio to [0,1] without relying on numpy clip on scalar
                temp_ratio = frame_diff / 10.0
                if temp_ratio < 0:
                    temp_ratio = 0.0
                elif temp_ratio > 1:
                    temp_ratio = 1.0
                temporal_score = 1.0 - temp_ratio
                
                # 3. Ball Proximity Score (use the best distance found in the window)
                ball_score = 0.0
                if best_ball_info['dist'] < max_ball_distance:
                    # Apply bonus if a close ball was found anywhere in the window
                    ball_score = 2.0
                
                # Combined Score
                # Base: 60% spatial, 40% temporal
                # Bonus: +2.0 if ball is confirmed
                final_score = (0.6 * spatial_score) + (0.4 * temporal_score) + ball_score
                
                candidates.append({
                    'det': row,
                    'score': final_score,
                    'spatial': spatial_score,
                    'ball_dist': best_ball_info['dist'], # Report the best distance
                    'frame_diff': frame_diff
                })
            
            # Pick the best candidate
            if not candidates:
                continue
                
            best_match = max(candidates, key=lambda x: x['score'])
            closest_det = best_match['det']
            ball_dist = best_match['ball_dist']

            print(f"\nAction at frame {action_frame} ({action['action']}, conf={action['confidence']:.3f}):")
            print(f"  Selected candidate from {len(candidates)} options:")
            print(f"    - Frame diff: {best_match['frame_diff']}")
            print(f"    - Spatial score: {best_match['spatial']:.3f}")

            # Save debug image for every analyzed action
            # Use the frame where the ball was closest, or the action frame if no ball was found
            debug_frame_num = best_ball_info['frame'] if best_ball_info['frame'] is not None else action_frame
            debug_frame_path_candidates = [
                frames_dir / f"{debug_frame_num:06d}.jpg",
                frames_dir / f"img1/{debug_frame_num:06d}.jpg"
            ]
            frame_path = next((c for c in debug_frame_path_candidates if c.exists()), None)

            if frame_path:
                try:
                    import cv2
                    debug_img = cv2.imread(str(frame_path))
                    if debug_img is not None:
                        # Get frame dimensions for filtering
                        img_height = debug_img.shape[0]
                        # Draw player bbox (green)
                        p_bbox = closest_det['bbox_ltwh']
                        p_x1, p_y1 = int(p_bbox[0]), int(p_bbox[1])
                        p_x2, p_y2 = int(p_bbox[0] + p_bbox[2]), int(p_bbox[1] + p_bbox[3])
                        cv2.rectangle(debug_img, (p_x1, p_y1), (p_x2, p_y2), (0, 255, 0), 2)
                        
                        # Re-run ball detection on this specific frame for visualization
                        final_results = ball_detector(str(frame_path), classes=[32], conf=ball_confidence, verbose=False)
                        if final_results and len(final_results) > 0 and len(final_results[0].boxes) > 0:
                            for b in final_results[0].boxes:
                                b_xywh = b.xywh[0].cpu().numpy()
                                b_x, b_y, b_w, b_h = float(b_xywh[0]), float(b_xywh[1]), float(b_xywh[2]), float(b_xywh[3])
                                
                                # Apply same filters to determine color
                                height_ratio = b_y / frame_height
                                aspect_ratio = b_w / b_h if b_h > 0 else 999
                                
                                # Check if detection passes all filters
                                height_ratio_calc = b_y / img_height
                                passes_filters = (
                                    height_ratio_calc >= (1.0 - ball_max_height) and
                                    b_w >= ball_min_size and b_h >= ball_min_size and
                                    b_w <= ball_max_size and b_h <= ball_max_size and
                                    aspect_ratio <= 1.35 and aspect_ratio >= 0.74
                                )
                                
                                b_x1, b_y1 = int(b_x - b_w/2), int(b_y - b_h/2)
                                b_x2, b_y2 = int(b_x + b_w/2), int(b_y + b_h/2)
                                
                                # Blue for valid ball, Red for filtered out
                                color = (255, 0, 0) if passes_filters else (0, 0, 255)
                                cv2.rectangle(debug_img, (b_x1, b_y1), (b_x2, b_y2), color, 2)

                        # Save the image
                        img_filename = f"frame_{action_frame}_{action['action']}_ball_detection.jpg"
                        cv2.imwrite(img_filename, debug_img)
                        print(f"    - Saved debug image: {img_filename} (visualizing frame {debug_frame_num})")

                except Exception as e:
                    print(f"    - ⚠️  Failed to save debug image: {e}")
            
            if has_ball_detector:
                status = "✅ VERIFIED" if ball_dist < max_ball_distance else ("❌ TOO FAR" if ball_dist != float('inf') else "⚠️ NO BALL DETECTED")
                ball_frame_info = f" at frame {best_ball_info['frame']}" if best_ball_info['frame'] is not None else ""
                print(f"    - Ball Distance: {ball_dist:.1f} px{ball_frame_info} -> {status}")
            
            # DECISION LOGIC: Require ball verification
            is_match = False
            if ball_dist < max_ball_distance:
                is_match = True
            elif not has_ball_detector:
                # No ball detector available - cannot verify, reject all
                print("  -> Rejected: ball detector unavailable")
            elif ball_dist == float('inf'):
                # Ball detector ran but found no ball - likely not a real action
                print("  -> Rejected: no ball detected in window")
            
            if is_match:
                matched_actions.append({
                    'action': action['action'],
                    'frame': action_frame,
                    'confidence': action['confidence'],
                    'player_frame': int(closest_det['frame_num']),
                    'frame_offset': int(best_match['frame_diff']),
                    'spatial_score': float(best_match['spatial']),
                    'ball_distance': float(ball_dist) if ball_dist != float('inf') else None
                })
            else:
                print(f"  ❌ Filtered out")
    
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
    seconds = int((frame - 1) / fps)
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


def load_ground_truth_labels(game_name: str, split: str, data_dir: str) -> Optional[Dict]:
    """
    Load ground truth labels from Labels-GameState.json
    
    Args:
        game_name: Game ID (e.g., "SNGS-021")
        split: Dataset split (train/valid/test/challenge)
        data_dir: Root directory of SoccerNetGS dataset
        
    Returns:
        Dictionary with ground truth annotations, or None if not found
    """
    # Flexible resolution:
    # 1. If data_dir is actually a file pointing to Labels-GameState.json -> use directly
    # 2. If data_dir points to a directory that already contains Labels-GameState.json -> use it
    # 3. Otherwise construct path from root: <data_dir>/<split>/SNGS-<num>/Labels-GameState.json
    data_path = Path(data_dir)
    if data_path.is_file() and data_path.name == 'Labels-GameState.json':
        labels_path = data_path
    elif (data_path / 'Labels-GameState.json').exists():
        labels_path = data_path / 'Labels-GameState.json'
    else:
        game_num = game_name.split('-')[-1]
        labels_path = data_path / split / f"SNGS-{game_num}" / "Labels-GameState.json"
    
    if not labels_path.exists():
        print(f"⚠️  Ground truth labels not found: {labels_path}")
        return None
    
    with open(labels_path, 'r') as f:
        labels = json.load(f)
    
    print(f"✓ Loaded {len(labels.get('annotations', []))} ground truth annotations")
    return labels


def calculate_iou(bbox1, bbox2) -> float:
    """
    Calculate Intersection over Union between two bounding boxes
    
    Args:
        bbox1, bbox2: [left, top, width, height]
        
    Returns:
        IoU score (0.0 to 1.0)
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    # Calculate intersection
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def validate_detections_with_ground_truth(detections: pd.DataFrame, ground_truth: Dict,
                                         game_name: str, iou_threshold: float = 0.5) -> Tuple[pd.DataFrame, Dict]:
    """
    Validate tracking detections against ground truth labels
    
    Args:
        detections: Tracking detections DataFrame
        ground_truth: Ground truth labels dictionary
        game_name: Game ID for reporting
        iou_threshold: Minimum IoU to consider a match (default: 0.5)
        
    Returns:
        Tuple of (validated_detections_df, validation_report_dict)
    """
    print("\n" + "="*60)
    print("🔍 VALIDATING DETECTIONS WITH GROUND TRUTH")
    print("="*60)
    
    if ground_truth is None:
        print("⚠️  No ground truth available - skipping validation")
        return detections, {}
    
    # Build lookup: image_id -> list of GT annotations plus suffix index for fallback
    gt_by_frame = {}
    gt_by_suffix = {}
    for ann in ground_truth.get('annotations', []):
        image_id = str(ann.get('image_id', ann.get('id', '')))
        if image_id:
            gt_by_frame.setdefault(image_id, []).append(ann)
            # Suffix: last 6 digits often correspond to frame number
            suffix = image_id[-6:]
            gt_by_suffix.setdefault(suffix, []).append(ann)
    
    # Determine jersey column
    jersey_col = None
    for col in ['jersey_number', 'jn_tracklet', 'jersey', 'jn']:
        if col in detections.columns:
            jersey_col = col
            break
    
    # Validate each detection
    detections = detections.copy()
    detections['gt_matched'] = False
    detections['gt_correct_jersey'] = False
    detections['gt_correct_team'] = False
    detections['gt_iou'] = 0.0
    
    matched_count = 0
    correct_jersey_count = 0
    correct_team_count = 0
    fully_correct_count = 0
    
    validation_details = []
    
    for idx, det in tqdm(detections.iterrows(), total=len(detections), desc="Validating detections"):
        image_id = str(det['image_id'])
        
        # Get GT annotations for this frame
        # Try exact match first, then suffix fallback if no annotations found
        gt_anns = gt_by_frame.get(image_id, [])
        if not gt_anns:
            suffix = image_id[-6:]
            gt_anns = gt_by_suffix.get(suffix, [])
        if not gt_anns:
            continue
        
        det_bbox = det['bbox_ltwh']
        if pd.isna(det_bbox).any():
            continue
        
        # Find best matching GT annotation
        best_iou = 0.0
        best_match = None
        
        for gt_ann in gt_anns:
            # Extract GT bbox in [x, y, w, h] format
            gt_bbox = gt_ann.get('bbox_ltwh') or gt_ann.get('bbox')
            if gt_bbox is None or (isinstance(gt_bbox, list) and len(gt_bbox) == 0):
                # Try bbox_image dict
                bbox_image = gt_ann.get('bbox_image')
                if bbox_image and all(k in bbox_image for k in ['x','y','w','h']):
                    gt_bbox = [bbox_image['x'], bbox_image['y'], bbox_image['w'], bbox_image['h']]
                else:
                    continue
            iou = calculate_iou(det_bbox, gt_bbox)
            
            if iou > best_iou:
                best_iou = iou
                best_match = gt_ann
        
        # Check if match is good enough
        if best_iou >= iou_threshold and best_match:
            detections.at[idx, 'gt_matched'] = True
            detections.at[idx, 'gt_iou'] = best_iou
            matched_count += 1
            
            jersey_correct = False
            team_correct = False
            
            # Check jersey number
            if jersey_col:
                det_jersey = det[jersey_col]
                gt_jersey = best_match.get('attributes', {}).get('jersey')
                if pd.notna(det_jersey) and gt_jersey is not None:
                    if int(det_jersey) == int(gt_jersey):
                        detections.at[idx, 'gt_correct_jersey'] = True
                        correct_jersey_count += 1
                        jersey_correct = True
            
            # Check team
            det_team = det.get('team')
            gt_team = best_match.get('attributes', {}).get('team')
            if det_team and gt_team and str(det_team) == str(gt_team):
                detections.at[idx, 'gt_correct_team'] = True
                correct_team_count += 1
                team_correct = True
            
            # Track fully correct detections
            if jersey_correct and team_correct:
                fully_correct_count += 1
            
            # Log validation details
            validation_details.append({
                'frame': image_id,
                'detection_team': str(det_team),
                'detection_jersey': int(det_jersey) if pd.notna(det_jersey) else None,
                'gt_team': gt_team,
                'gt_jersey': gt_jersey,
                'iou': float(best_iou),
                'jersey_correct': jersey_correct,
                'team_correct': team_correct,
                'fully_correct': jersey_correct and team_correct
            })
    
    # Print validation statistics
    print(f"\n📊 Validation Results:")
    print(f"  Total detections: {len(detections)}")
    print(f"  Matched to GT (IoU≥{iou_threshold}): {matched_count} ({100*matched_count/len(detections):.1f}%)")

    if matched_count == 0:
        # Provide debug info to help diagnose
        print("\n🔎 Debug: No matches found. Showing first 3 detection samples and GT availability.")
        sample_dets = detections.head(3)
        for i, (_, det_row) in enumerate(sample_dets.iterrows(), 1):
            did = str(det_row.get('image_id'))
            suffix = did[-6:] if did else 'N/A'
            gt_exact = len(gt_by_frame.get(did, []))
            gt_suffix = len(gt_by_suffix.get(suffix, []))
            print(f"  Det {i}: image_id={did} suffix={suffix} bbox={det_row.get('bbox_ltwh')} -> GT exact:{gt_exact} GT suffix:{gt_suffix}")
        print("   Potential causes: differing image_id formats, bbox mismatch, or tracker file not aligned with labels.")
    
    if matched_count > 0:
        print(f"  Correct jersey: {correct_jersey_count} ({100*correct_jersey_count/matched_count:.1f}% of matched)")
        print(f"  Correct team: {correct_team_count} ({100*correct_team_count/matched_count:.1f}% of matched)")
        print(f"  Fully correct (jersey + team): {fully_correct_count} ({100*fully_correct_count/matched_count:.1f}% of matched)")
    
    # Analyze errors
    matched_dets = detections[detections['gt_matched']]
    if len(matched_dets) > 0:
        wrong_jersey = matched_dets[~matched_dets['gt_correct_jersey']]
        wrong_team = matched_dets[~matched_dets['gt_correct_team']]
        
        print(f"\n⚠️  Detection Errors:")
        print(f"  Wrong jersey number: {len(wrong_jersey)} ({100*len(wrong_jersey)/len(matched_dets):.1f}%)")
        print(f"  Wrong team: {len(wrong_team)} ({100*len(wrong_team)/len(matched_dets):.1f}%)")
    
    # Create validation report
    validation_report = {
        'game': game_name,
        'total_detections': len(detections),
        'matched_detections': matched_count,
        'match_rate': matched_count / len(detections) if len(detections) > 0 else 0.0,
        'correct_jersey_count': correct_jersey_count,
        'correct_team_count': correct_team_count,
        'fully_correct_count': fully_correct_count,
        'jersey_accuracy': correct_jersey_count / matched_count if matched_count > 0 else 0.0,
        'team_accuracy': correct_team_count / matched_count if matched_count > 0 else 0.0,
        'full_accuracy': fully_correct_count / matched_count if matched_count > 0 else 0.0,
        'iou_threshold': iou_threshold,
        'validation_details': validation_details[:100]  # Limit to first 100 for file size
    }
    
    return detections, validation_report


def get_ground_truth_player_combinations(ground_truth: Dict) -> List[Tuple[str, int]]:
    """
    Extract unique (team, jersey) combinations from ground truth
    
    Args:
        ground_truth: Ground truth labels dictionary
        
    Returns:
        List of (team, jersey) tuples that exist in ground truth
    """
    if ground_truth is None:
        return []
    
    combinations = set()
    for ann in ground_truth.get('annotations', []):
        attrs = ann.get('attributes', {})
        team = attrs.get('team')
        jersey = attrs.get('jersey')
        role = attrs.get('role')
        
        # Only include players/goalkeepers with valid team and jersey
        if role in ['player', 'goalkeeper'] and team and jersey is not None:
            combinations.add((str(team), int(jersey)))
    
    return sorted(list(combinations))


def analyze_tracking_data(detections: pd.DataFrame):
    """
    Analyze tracking data quality to identify issues like duplicate jerseys
    """
    print("\n" + "="*60)
    print("📊 DATA QUALITY ANALYSIS")
    print("="*60)
    
    print(f"Total detections: {len(detections)}")
    print(f"Columns: {list(detections.columns)}")
    
    # Check teams
    if 'team' in detections.columns:
        print("\nTeam distribution:")
        print(detections['team'].value_counts(dropna=False))
    
    # Check jerseys
    jersey_col = None
    for col in ['jn_tracklet', 'jersey_number', 'jersey', 'jn']:
        if col in detections.columns:
            jersey_col = col
            break
            
    if jersey_col:
        print(f"\nJersey column: '{jersey_col}'")
        
        # Convert to numeric for analysis, coercing errors
        detections_work = detections.copy()
        detections_work[jersey_col] = pd.to_numeric(detections_work[jersey_col], errors='coerce')
        
        print("Jersey distribution (top 20):")
        print(detections_work[jersey_col].value_counts(dropna=False).head(20))
        
        # Check for duplicates (same frame, same jersey)
        print("\nChecking for duplicate jerseys in same frame...")
        if 'image_id' in detections.columns:
            frame_col = 'image_id'
        else:
            # Fallback to index if it looks like frame info
            frame_col = detections.index.name or 'index'
            
        # Filter out NaNs for jersey
        valid_jerseys = detections_work[detections_work[jersey_col].notna()]
        
        # Group by frame and jersey
        dupes = valid_jerseys.groupby([frame_col, jersey_col]).size()
        dupes = dupes[dupes > 1]
        
        if len(dupes) > 0:
            print(f"\n⚠️  Found {len(dupes)} instances of same jersey appearing multiple times in a single frame!")
            print("Sample duplicates (Frame, Jersey) -> Count:")
            print(dupes.head(10))
            
            # Analyze a few examples
            print("\nDetailed analysis of first 5 duplicates:")
            for (frame, jersey), count in dupes.head(5).items():
                subset = detections_work[
                    (detections_work[frame_col] == frame) & 
                    (detections_work[jersey_col] == jersey)
                ]
                print(f"  Frame {frame}, Jersey {int(jersey)}:")
                for _, row in subset.iterrows():
                    team_val = row.get('team', 'N/A')
                    conf = row.get('conf', 'N/A')
                    bbox = row.get('bbox_ltwh', 'N/A')
                    print(f"    - Team: {team_val}, Conf: {conf}, BBox: {bbox}")
        else:
            print("\n✓ No duplicate jerseys found in any single frame.")
            
        # Check for 'nan' teams with valid jerseys
        nan_team_players = detections_work[
            (detections_work['team'].astype(str) == 'nan') & 
            (detections_work[jersey_col].notna())
        ]
        if len(nan_team_players) > 0:
            print(f"\n⚠️  Found {len(nan_team_players)} detections with valid jersey but 'nan' team")
            print("Sample jerseys with nan team:")
            print(nan_team_players[jersey_col].value_counts().head(10))
            
    else:
        print("No jersey column found.")
        
    print("\n" + "="*60)
    sys.exit(0)


def show_all_actions_with_players(actions: List[Dict], detections: pd.DataFrame, 
                                   frames_dir: Optional[Path] = None, fps: float = 25.0):
    """Display all detected actions with nearest player attribution.
    
    For each action, finds the nearest player(s) in space and time,
    and reports team, jersey, distance, and ball proximity.
    Useful for manual verification of true/false positives.
    """
    print("\n" + "="*70)
    print("🎯 ALL DETECTED ACTIONS WITH PLAYER ATTRIBUTION")
    print("="*70)
    print(f"Total actions: {len(actions)}\n")
    sys.stdout.flush()  # Force output to display
    
    # Prepare detections
    dets_work = detections.copy()
    if dets_work['image_id'].dtype == 'object':
        dets_work['frame_num'] = dets_work['image_id'].astype(str).str[-6:].astype(int)
    else:
        dets_work['frame_num'] = pd.to_numeric(dets_work['image_id'], errors='coerce')
    
    jersey_col = None
    for col in ['jersey_number', 'jn_tracklet', 'jersey', 'jn']:
        if col in dets_work.columns:
            jersey_col = col
            break
    
    # Initialize ball detector
    ball_detector = None
    try:
        from ultralytics import YOLO
        ball_detector = YOLO('yolov8m.pt')
    except:
        pass
    
    for i, action in enumerate(actions, 1):
        action_frame = action['frame']
        action_time = format_time(action_frame, fps)
        
        print(f"\n{i}. Frame {action_frame} ({action_time}) | {action['action']} | conf={action['confidence']:.3f}")
        
        # Find nearby players (±50 frames)
        nearby = dets_work[
            (dets_work['frame_num'] >= action_frame - 50) &
            (dets_work['frame_num'] <= action_frame + 50)
        ].copy()
        
        if len(nearby) == 0:
            print("   ⚠️  No players detected nearby")
            continue
        
        # Calculate distances
        frame_center = (960, 540)  # 1920x1080 center
        candidates = []
        
        for _, row in nearby.iterrows():
            if pd.isna(row['bbox_ltwh']).any():
                continue
            
            left, top, width, height = row['bbox_ltwh']
            bbox_center = (left + width/2, top + height/2)
            
            # Spatial distance from frame center
            spatial_dist = ((bbox_center[0] - frame_center[0])**2 + 
                           (bbox_center[1] - frame_center[1])**2)**0.5
            
            # Temporal distance
            frame_diff = abs(row['frame_num'] - action_frame)
            
            team = row.get('team', 'unknown')
            jersey = int(row[jersey_col]) if jersey_col and pd.notna(row[jersey_col]) else None
            
            candidates.append({
                'team': team,
                'jersey': jersey,
                'frame_diff': frame_diff,
                'spatial_dist': spatial_dist,
                'bbox_center': bbox_center,
                'track_id': row.get('track_id')
            })
        
        # Sort by combined score (temporal + spatial)
        candidates.sort(key=lambda x: x['frame_diff'] + x['spatial_dist']/500)
        
        # Try ball detection for top candidate
        ball_info = ""
        if ball_detector and frames_dir and len(candidates) > 0:
            frame_candidates = [
                frames_dir / f"{action_frame:06d}.jpg",
                frames_dir / f"{action_frame:06d}.png",
                frames_dir / f"img1/{action_frame:06d}.jpg"
            ]
            frame_path = None
            for c in frame_candidates:
                if c.exists():
                    frame_path = c
                    break
            
            if frame_path:
                try:
                    import cv2
                    frame_img = cv2.imread(str(frame_path))
                    if frame_img is not None:
                        results = ball_detector(frame_img, classes=[32], conf=0.15, verbose=False)
                        if results and len(results) > 0 and len(results[0].boxes) > 0:
                            ball_dets = results[0].boxes
                            # Find closest ball to top candidate
                            min_ball_dist = float('inf')
                            for b in ball_dets:
                                b_xywh = b.xywh[0].cpu().numpy()
                                b_center = (float(b_xywh[0]), float(b_xywh[1]))
                                d = ((candidates[0]['bbox_center'][0] - b_center[0])**2 + 
                                    (candidates[0]['bbox_center'][1] - b_center[1])**2)**0.5
                                if d < min_ball_dist:
                                    min_ball_dist = d
                            
                            if min_ball_dist < 300:
                                ball_info = f" | ball_dist={min_ball_dist:.0f}px"
                            else:
                                ball_info = f" | ball_dist={min_ball_dist:.0f}px (far)"
                except:
                    pass
        
        # Show top 3 candidates
        print("   Nearest players:")
        for j, c in enumerate(candidates[:3], 1):
            jersey_str = f"#{c['jersey']}" if c['jersey'] else "#?"
            track_str = f"track_{c['track_id']}" if c.get('track_id') else ""
            extra = ball_info if j == 1 else ""
            print(f"     {j}. Team {c['team']}, Jersey {jersey_str} {track_str} | "
                  f"Δframe={c['frame_diff']}, spatial_dist={c['spatial_dist']:.0f}px{extra}")
        sys.stdout.flush()  # Force output
    
    print("\n" + "="*70)
    print("💡 Use this output to manually verify actions in the video.")
    print("="*70 + "\n")
    sys.stdout.flush()


def analyze_jersey_assignments(detections: pd.DataFrame, ground_truth: Optional[Dict] = None):
    """Analyze per-track jersey assignment stability and optionally compare to ground truth.

    Prints summary:
      - track_id, total frames, distinct jersey_number values, majority jersey, majority fraction
      - jersey_number_confidence stats if available
      - mismatch with ground truth (if validated columns present)
    """
    print("\n" + "="*60)
    print("🧪 JERSEY ASSIGNMENT ANALYSIS")
    print("="*60)

    jersey_col = None
    for col in ['jersey_number', 'jn_tracklet', 'jersey', 'jn']:
        if col in detections.columns:
            jersey_col = col
            break
    if jersey_col is None:
        print("No jersey column found; aborting jersey analysis.")
        return

    if 'track_id' not in detections.columns:
        print("No track_id column; cannot group by tracks.")
        return

    df = detections.copy()
    df[jersey_col] = pd.to_numeric(df[jersey_col], errors='coerce')

    groups = []
    for tid, g in df.groupby('track_id'):
        jerseys = g[jersey_col].dropna().astype(int)
        total = len(g)
        distinct = sorted(jerseys.unique().tolist()) if len(jerseys) else []
        majority_jersey = None
        majority_frac = 0.0
        if len(jerseys):
            vc = jerseys.value_counts()
            majority_jersey = int(vc.index[0])
            majority_frac = vc.iloc[0] / len(jerseys)
        conf_stats = None
        if 'jersey_number_confidence' in g.columns:
            conf_vals = pd.to_numeric(g['jersey_number_confidence'], errors='coerce').dropna()
            if len(conf_vals):
                conf_stats = {
                    'mean_conf': float(conf_vals.mean()),
                    'min_conf': float(conf_vals.min()),
                    'max_conf': float(conf_vals.max())
                }
        groups.append({
            'track_id': tid,
            'frames': total,
            'distinct_jerseys': distinct,
            'majority_jersey': majority_jersey,
            'majority_frac': majority_frac,
            'conf_stats': conf_stats
        })

    # Sort tracks by majority stability descending
    groups.sort(key=lambda x: x['majority_frac'], reverse=True)

    print(f"Total tracks: {len(groups)}")
    print("\nTrack Summary (top 15):")
    for row in groups[:15]:
        conf_str = ''
        if row['conf_stats']:
            conf_str = f" | conf mean={row['conf_stats']['mean_conf']:.2f}"
        print(f"  Track {row['track_id']}: frames={row['frames']}, jerseys={row['distinct_jerseys']}, majority={row['majority_jersey']} ({row['majority_frac']*100:.1f}%){conf_str}")

    unstable = [r for r in groups if r['majority_frac'] < 0.6 and len(r['distinct_jerseys']) > 1]
    if unstable:
        print(f"\n⚠️  Unstable jersey assignments (majority <60% & >1 distinct): {len(unstable)}")
        for r in unstable[:10]:
            print(f"  Track {r['track_id']}: jerseys={r['distinct_jerseys']} majority={r['majority_jersey']} ({r['majority_frac']*100:.1f}%)")
    else:
        print("\n✓ All tracks have stable jersey assignment (>=60% majority or single value).")

    # Ground truth comparison if validated detections present
    if 'gt_matched' in df.columns and 'gt_correct_jersey' in df.columns:
        print("\nGround Truth Comparison:")
        for r in groups[:15]:
            track_rows = df[df['track_id'] == r['track_id']]
            gt_rows = track_rows[track_rows['gt_matched']]
            if len(gt_rows):
                correct = gt_rows[gt_rows['gt_correct_jersey']]
                rate = len(correct)/len(gt_rows) if len(gt_rows) else 0
                print(f"  Track {r['track_id']}: GT matched={len(gt_rows)} jersey_accuracy={rate*100:.1f}%")

    print("\nSuggestion: consider majority-vote reassignment for unstable tracks before filtering.")

    print("\n" + "="*60)
    # Do not exit; allow pipeline to continue


def build_player_detections_from_ground_truth(ground_truth: Dict) -> pd.DataFrame:
    """
    Build a detections DataFrame from ground truth annotations.

    This creates a DataFrame compatible with the rest of the pipeline,
    using the "perfect" detections from the labels file.

    Args:
        ground_truth: Loaded Labels-GameState.json data.

    Returns:
        A pandas DataFrame with player detections.
    """
    print("\n" + "="*60)
    print("BUILDING DETECTIONS FROM GROUND TRUTH")
    print("="*60)

    if not ground_truth or 'annotations' not in ground_truth:
        print("⚠️  Ground truth data is empty or invalid.")
        return pd.DataFrame()

    records = []
    for ann in tqdm(ground_truth['annotations'], desc="Processing GT annotations"):
        attrs = ann.get('attributes', {})
        role = attrs.get('role')

        # We only care about players and goalkeepers
        if role not in ['player', 'goalkeeper']:
            continue

        team = attrs.get('team')
        jersey = attrs.get('jersey')
        image_id = ann.get('image_id')

        # Extract bbox
        bbox = ann.get('bbox_ltwh') or ann.get('bbox')
        if bbox is None:
            bbox_image = ann.get('bbox_image')
            if bbox_image and all(k in bbox_image for k in ['x', 'y', 'w', 'h']):
                bbox = [bbox_image['x'], bbox_image['y'], bbox_image['w'], bbox_image['h']]

        if all(v is not None for v in [team, jersey, image_id, bbox]):
            records.append({
                'image_id': image_id,
                'team': team,
                'jersey_number': int(jersey),
                'bbox_ltwh': bbox,
                'track_id': f"gt_{team}_{jersey}" # Create a stable pseudo-track_id
            })

    if not records:
        print("⚠️  No valid player annotations found in ground truth.")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    print(f"✓ Built {len(df)} detections from ground truth for {len(df['track_id'].unique())} unique players.")
    return df


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
    
    # Step 1: Load detections from ground truth or tracker state
    if args.use_ground_truth_detections:
        print("\n📋 Loading detections from GROUND TRUTH...")
        label_source = args.labels_path if args.labels_path else args.data_dir
        if not game_name:
            print("\n❌ ERROR: Must specify a game name (e.g., SNGS-001) when using --use-ground-truth-detections.")
            sys.exit(1)
        gt_labels = load_ground_truth_labels(game_name, args.split, label_source)
        if not gt_labels:
            print("\n❌ ERROR: Could not load ground truth labels. Please check --labels-path or --data-dir.")
            sys.exit(1)
        detections = build_player_detections_from_ground_truth(gt_labels)

    else:
        # Load from tracker state cache
        print("\n📂 Loading pre-computed tracking state...")
        print("   (Run tracklab separately first to generate this file)")
        if not args.state_cache or not Path(args.state_cache).exists():
            print(f"\n❌ ERROR: Tracker state file not found or not specified: {args.state_cache}")
            print("\nTo generate tracker state, run tracklab first:")
            print(f"  cd /path/to/sn-gamestate")
            print(f"  uv run tracklab -cn soccernet dataset.eval_set={args.split} dataset.vids_dict.{args.split}=[{game_name}]")
            print(f"\nOr, use the --use-ground-truth-detections flag to bypass the tracker.")
            sys.exit(1)
        detections = load_tracking_state(args.state_cache, game_name)
    
    # Run analysis if requested
    if args.analyze_data:
        analyze_tracking_data(detections)
    if args.analyze_jerseys:
        analyze_jersey_assignments(detections)
    
    # Load and validate with ground truth if requested
    validation_report = None
    if args.validate_detections and game_name and not args.use_ground_truth_detections:
        print(f"\n📋 Loading ground truth labels for validation...")
        # Prefer explicit labels-path if provided
        label_source = args.labels_path if args.labels_path else args.data_dir
        gt_labels = load_ground_truth_labels(game_name, args.split, label_source)
        
        if gt_labels:
            # Validate all detections
            detections, validation_report = validate_detections_with_ground_truth(
                detections, gt_labels, game_name, iou_threshold=0.5
            )
            
            # Get ground truth player combinations
            gt_combinations = get_ground_truth_player_combinations(gt_labels)
            print(f"\n✅ Ground Truth Player Combinations ({len(gt_combinations)}):")
            for team, jersey in gt_combinations:
                # Check if this player exists in validated detections
                jersey_col = None
                for col in ['jersey_number', 'jn_tracklet', 'jersey', 'jn']:
                    if col in detections.columns:
                        jersey_col = col
                        break
                
                if jersey_col:
                    validated_dets = detections[
                        (detections['team'] == team) &
                        (detections[jersey_col] == jersey) &
                        (detections['gt_matched'] == True) &
                        (detections['gt_correct_jersey'] == True) &
                        (detections['gt_correct_team'] == True)
                    ]
                    
                    status = f"✓ {len(validated_dets)} correct detections" if len(validated_dets) > 0 else "⚠️  Not detected by tracker"
                    print(f"  - Team: {team}, Jersey: {jersey} - {status}")
            
            # Filter to only fully correct detections
            before_count = len(detections)
            detections = detections[
                (detections['gt_matched'] == True) &
                (detections['gt_correct_jersey'] == True) &
                (detections['gt_correct_team'] == True)
            ]
            print(f"\n🔧 Filtered to only ground-truth validated detections: {len(detections)}/{before_count}")
            print(f"   Removed {before_count - len(detections)} incorrect/unmatched detections")
            
            # Save validation report
            report_file = f"validation_report_{game_name}.json"
            with open(report_file, 'w') as f:
                json.dump(validation_report, f, indent=2)
            print(f"\n💾 Validation report saved to: {report_file}")
    
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
            # Handle directory vs file path
            save_arg_path = Path(args.save_video)
            if save_arg_path.suffix == '': # It's a directory
                save_arg_path.mkdir(parents=True, exist_ok=True)
                save_path = (save_arg_path / f"{game_name}_video.mp4").resolve()
            else: # It's a file path
                save_arg_path.parent.mkdir(parents=True, exist_ok=True)
                save_path = save_arg_path.resolve()
                
            try:
                shutil.copy(temp_video, save_path)
                print(f"✓ Video saved to: {save_path}")
                # Ensure we don't delete the temp video if it's the same file
                if temp_video.resolve() == save_path:
                    cleanup_video = False
                else:
                    cleanup_video = True # We have a copy, so we can clean up temp
            except Exception as e:
                print(f"⚠️  Warning: Failed to save video to {save_path}: {e}")
                cleanup_video = True
        else:
            cleanup_video = True
    else:
        cleanup_video = False
    
    # Step 2: Filter player by jersey
    player_dets = filter_player_by_jersey(detections, args.team, args.jersey)
    
    # Step 3: Run ball action detection
    actions = run_ball_action_detection(str(video_path), args.experiment, args.fold, args.device)
    
    # Optional: Show all actions with player attribution before filtering
    if args.show_all_actions:
        show_all_actions_with_players(actions, detections, frames_dir, fps=25.0)
    
    # Step 4: Match actions to player
    matched_actions = match_actions_to_player(
        actions, player_dets,
        window_frames=50,
        min_confidence=0.75,  # Higher threshold to reduce false positives
        min_time_between_actions=1.0,  # At least 1 second between actions
        fps=25.0,
        frames_dir=frames_dir,
        ball_confidence=args.ball_confidence,
        ball_max_height=args.ball_max_height,
        ball_min_size=args.ball_min_size,
        ball_max_size=args.ball_max_size,
        ball_search_window=args.ball_search_window,
        max_ball_distance=args.max_ball_distance
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
            print(f"    - {action['time']} (frame {action['frame']}) | {action['action']} (confidence: {action['confidence']:.3f})")
        
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
