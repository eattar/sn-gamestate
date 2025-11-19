#!/usr/bin/env python3
"""
Ball Action Spotting Integration with SN-GameState
===================================================

This script integrates ball-action-spotting with sn-gamestate player tracking
to output ball actions performed by a specific player (identified by team and jersey number).

Usage:
    python run_player_ball_actions.py --video match.mp4 --team left --jersey 10 --output results.json

Requirements:
    - SN-GameState installed and configured
    - Ball-action-spotting models in ../ball-action-spotting/data/ball_action/experiments/
    - Input video file

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
  # Basic usage
  python run_player_ball_actions.py --video match.mp4 --team left --jersey 10
  
  # Specify output file
  python run_player_ball_actions.py --video match.mp4 --team right --jersey 7 --output player7_actions.json
  
  # Use cached tracking state
  python run_player_ball_actions.py --video match.mp4 --team left --jersey 10 --state-cache tracking_state.pklz
  
  # Use specific ball-action model
  python run_player_ball_actions.py --video match.mp4 --team left --jersey 10 --experiment ball_finetune_long_004 --fold 5
        """
    )
    
    parser.add_argument('--video', required=True, type=str,
                        help='Path to input video file')
    parser.add_argument('--team', required=True, choices=['left', 'right'],
                        help='Player team (left or right)')
    parser.add_argument('--jersey', required=True, type=int,
                        help='Player jersey number')
    parser.add_argument('--output', type=str,
                        help='Output JSON file (default: player_<jersey>_<team>_actions.json)')
    parser.add_argument('--state-cache', type=str,
                        help='Path to cached tracking state .pklz file (skip tracking if provided)')
    parser.add_argument('--experiment', type=str, default='ball_finetune_long_004',
                        help='Ball-action experiment name (default: ball_finetune_long_004)')
    parser.add_argument('--fold', type=int, default=5,
                        help='Model fold number (default: 5, which has 90.1%% accuracy)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device for ball-action model (default: cuda:0)')
    parser.add_argument('--skip-tracking', action='store_true',
                        help='Skip player tracking (requires --state-cache)')
    
    return parser.parse_args()


def run_sn_gamestate_tracking(video_path: str) -> Tuple[pd.DataFrame, str]:
    """
    Run SN-GameState tracking pipeline on video
    
    Args:
        video_path: Path to video file
        
    Returns:
        Tuple of (detections_dataframe, state_file_path)
    """
    print("\n" + "="*60)
    print("STEP 1: Running SN-GameState Player Tracking")
    print("="*60)
    
    from hydra import compose, initialize_config_dir
    from tracklab.engine import run_tracklab
    
    # Get config directory
    config_dir = str(Path(__file__).parent / "sn_gamestate" / "configs")
    
    with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
        # Load config and modify for single video
        cfg = compose(config_name="soccernet")
        
        # Disable visualization to save time
        cfg.visualization.cfg.save_videos = False
        
        # Remove unnecessary modules (pitch and calibration not needed for action matching)
        # Keep: bbox_detector, reid, track, jersey_number_detect, tracklet_agg, team, team_side
        cfg.pipeline = [
            'bbox_detector',
            'reid', 
            'track',
            'jersey_number_detect',
            'tracklet_agg',
            'team',
            'team_side'
        ]
        
        # TODO: Configure for external video input
        # For now, user needs to place video in SoccerNetGS dataset structure
        print(f"\n⚠️  Note: Currently requires video in SoccerNetGS dataset format")
        print(f"   Please ensure video is at: {cfg.dataset.dataset_path}/[game_name]/")
        
        # Run tracking
        print("\n▶ Running tracking pipeline...")
        tracker_state = run_tracklab(cfg)
        
        # Get detections
        detections = tracker_state.detections_pred
        
        # Save state for future use
        state_file = Path("tracking_state_temp.pklz")
        print(f"\n💾 Saving tracking state to: {state_file}")
        with gzip.open(state_file, 'wb') as f:
            pickle.dump(tracker_state, f)
        
        print(f"✓ Tracking complete: {len(detections)} detections")
        
        return detections, str(state_file)


def load_tracking_state(state_path: str) -> pd.DataFrame:
    """Load tracking state from .pklz file"""
    print(f"\n📂 Loading tracking state from: {state_path}")
    
    with gzip.open(state_path, 'rb') as f:
        tracker_state = pickle.load(f)
    
    detections = tracker_state.detections_pred
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
    
    # Check available teams and jerseys
    if 'team' in detections.columns:
        available_teams = detections['team'].dropna().unique()
        print(f"Available teams: {list(available_teams)}")
    
    if 'jn_tracklet' in detections.columns:
        available_jerseys = detections['jn_tracklet'].dropna().unique()
        print(f"Available jersey numbers: {sorted([int(j) for j in available_jerseys if pd.notna(j)])}")
    
    # Filter by team and jersey
    player_dets = detections[
        (detections['team'] == team) & 
        (detections['jn_tracklet'] == jersey)
    ].copy()
    
    if len(player_dets) == 0:
        print(f"\n❌ ERROR: No player found with team='{team}' and jersey={jersey}")
        print("\nAvailable combinations:")
        
        if 'team' in detections.columns and 'jn_tracklet' in detections.columns:
            team_jersey = detections[['team', 'jn_tracklet']].dropna().drop_duplicates()
            for _, row in team_jersey.iterrows():
                print(f"  - Team: {row['team']}, Jersey: {int(row['jn_tracklet'])}")
        
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
    
    # Post-process to get action events
    from src.utils import post_processing
    
    actions = []
    for cls_name, cls_idx in ball_constants.class2target.items():
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
                            window_frames: int = 50) -> List[Dict]:
    """
    Match detected actions to player using temporal proximity and spatial overlap
    
    Args:
        actions: List of detected actions
        player_dets: Player detections dataframe
        window_frames: Frame window for matching (±N frames)
        
    Returns:
        List of matched actions with player info
    """
    print("\n" + "="*60)
    print("STEP 4: Matching Actions to Player")
    print("="*60)
    print(f"   Using ±{window_frames} frame window for matching")
    
    matched_actions = []
    
    for action in tqdm(actions, desc="Matching actions"):
        action_frame = action['frame']
        
        # Get player detections within time window
        nearby_dets = player_dets[
            (player_dets['image_id'] >= action_frame - window_frames) &
            (player_dets['image_id'] <= action_frame + window_frames)
        ]
        
        if len(nearby_dets) > 0:
            # Find closest detection by frame
            nearby_dets = nearby_dets.copy()
            nearby_dets['frame_diff'] = abs(nearby_dets['image_id'] - action_frame)
            closest_det = nearby_dets.loc[nearby_dets['frame_diff'].idxmin()]
            
            matched_actions.append({
                'action': action['action'],
                'frame': action_frame,
                'confidence': action['confidence'],
                'player_frame': int(closest_det['image_id']),
                'frame_offset': int(closest_det['frame_diff'])
            })
    
    print(f"\n✓ Matched {len(matched_actions)} actions to player")
    print(f"   Unmatched: {len(actions) - len(matched_actions)} (player not visible in frame window)")
    
    return matched_actions


def format_time(frame: int, fps: float) -> str:
    """Convert frame number to MM:SS format"""
    seconds = int(frame / fps)
    minutes = seconds // 60
    secs = seconds % 60
    return f"{minutes}:{secs:02d}"


def create_output_json(matched_actions: List[Dict], team: str, jersey: int, 
                      video_path: str) -> Dict:
    """Create output JSON structure"""
    video_info = get_video_info(video_path)
    fps = video_info['fps']
    
    # Add timestamps
    for action in matched_actions:
        action['time'] = format_time(action['frame'], fps)
    
    # Create output
    output = {
        'player': {
            'team': team,
            'jersey': jersey
        },
        'video': str(video_path),
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
    print(f"  Video: {args.video}")
    print(f"  Target Player: Team={args.team}, Jersey=#{args.jersey}")
    print(f"  Ball-Action Model: {args.experiment}/fold_{args.fold}")
    print("="*70)
    
    # Validate video exists
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"\n❌ ERROR: Video file not found: {video_path}")
        sys.exit(1)
    
    # Step 1: Get tracking detections
    if args.state_cache and Path(args.state_cache).exists():
        detections = load_tracking_state(args.state_cache)
    elif args.skip_tracking:
        print("\n❌ ERROR: --skip-tracking requires --state-cache with valid file")
        sys.exit(1)
    else:
        detections, state_file = run_sn_gamestate_tracking(str(video_path))
    
    # Step 2: Filter player by jersey
    player_dets = filter_player_by_jersey(detections, args.team, args.jersey)
    
    # Step 3: Run ball action detection
    actions = run_ball_action_detection(str(video_path), args.experiment, args.fold, args.device)
    
    # Step 4: Match actions to player
    matched_actions = match_actions_to_player(actions, player_dets)
    
    # Step 5: Create output
    print("\n" + "="*60)
    print("STEP 5: Creating Output")
    print("="*60)
    
    output = create_output_json(matched_actions, args.team, args.jersey, str(video_path))
    
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


if __name__ == '__main__':
    main()
