"""
TrackLab module for ball action spotting integration.

This module integrates ball-action-spotting into the TrackLab pipeline,
allowing you to run ball action detection as part of the sn-gamestate workflow.
"""

from pathlib import Path
from typing import Optional
import subprocess
import json
import logging

from tracklab.pipeline import ImageLevelModule


logger = logging.getLogger(__name__)


class BallActionSpottingModule(ImageLevelModule):
    """
    TrackLab module for detecting ball actions (PASS, DRIVE, etc.) in soccer videos.
    
    This module runs the ball-action-spotting model on the input video and loads
    the predictions into the TrackLab state for downstream processing.
    
    Config parameters:
        ball_action_repo_path: Path to ball-action-spotting repository
        experiment_name: Name of the trained model experiment (e.g., 'sampling_weights_001')
        fold: Model fold to use (default: 0)
        use_cached: Whether to use existing predictions if available (default: True)
    """
    
    input_columns = []
    output_columns = [
        "ball_actions"  # List of detected ball actions with timestamps and confidence
    ]
    
    def __init__(self, 
                 ball_action_repo_path: str,
                 experiment_name: str = "sampling_weights_001",
                 fold: int = 0,
                 use_cached: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.repo_path = Path(ball_action_repo_path)
        self.experiment_name = experiment_name
        self.fold = fold
        self.use_cached = use_cached
        
        # Validate repository path
        if not self.repo_path.exists():
            raise ValueError(f"Ball action repository not found at: {self.repo_path}")
        
        predict_script = self.repo_path / "scripts/ball_action/predict.py"
        if not predict_script.exists():
            raise ValueError(f"Predict script not found at: {predict_script}")
    
    def preprocess_image_level(self):
        """
        Run ball action spotting on the entire video before frame-by-frame processing.
        This runs once at the start of the pipeline.
        """
        logger.info("Running ball action spotting on video...")
        
        # Get video path from TrackLab state
        video_path = self.video_metadata.get("video_path")
        if not video_path:
            logger.error("No video path found in metadata")
            return
        
        # Determine output path
        predictions_dir = self.repo_path / "data/ball_action/predictions" / self.experiment_name / "cv" / f"fold_{self.fold}"
        
        # Check if predictions already exist
        video_name = Path(video_path).stem
        results_file = predictions_dir / f"{video_name}/results_spotting.json"
        
        if self.use_cached and results_file.exists():
            logger.info(f"Using cached predictions from: {results_file}")
            self._load_predictions(results_file)
            return
        
        # Run prediction script
        logger.info(f"Running ball action prediction (this may take 15-20 minutes)...")
        cmd = [
            "python",
            str(self.repo_path / "scripts/ball_action/predict.py"),
            "--experiment", self.experiment_name,
            "--folds", str(self.fold),
            "--gpu_id", str(self.device_id if hasattr(self, 'device_id') else 0)
        ]
        
        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.repo_path),
                capture_output=True,
                text=True,
                check=True,
                env={**subprocess.os.environ, "PYTHONPATH": str(self.repo_path)}
            )
            logger.info(f"Ball action prediction completed")
            logger.debug(result.stdout)
            
            # Load predictions
            self._load_predictions(results_file)
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Ball action prediction failed: {e}")
            logger.error(f"stdout: {e.stdout}")
            logger.error(f"stderr: {e.stderr}")
            raise
    
    def _load_predictions(self, results_file: Path):
        """Load ball action predictions from JSON file into state."""
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        predictions = data.get('predictions', [])
        logger.info(f"Loaded {len(predictions)} ball action predictions")
        
        # Store in state
        self.state['ball_actions'] = predictions
        
        # Log statistics
        action_counts = {}
        for pred in predictions:
            label = pred['label']
            action_counts[label] = action_counts.get(label, 0) + 1
        
        for label, count in action_counts.items():
            logger.info(f"  {label}: {count} actions")
    
    def process(self, image, detections, metadata):
        """
        Process individual frame (not needed for ball actions since they're video-level).
        Just pass through the existing detections.
        """
        return detections


class ActionPlayerMatchingModule(ImageLevelModule):
    """
    TrackLab module for matching ball actions to player jersey numbers.
    
    This module takes ball action predictions and player tracking data,
    and assigns each action to the nearest player at that timestamp.
    
    Config parameters:
        max_distance: Maximum distance (pixels) to assign action to player (default: 200)
        use_ball_position: Whether to use ball position for matching (default: False)
    """
    
    input_columns = [
        "ball_actions",  # From BallActionSpottingModule
        "bbox",          # Player bounding boxes from tracking
        "jersey_number", # Jersey numbers from jersey recognition
        "track_id"       # Track IDs from tracking
    ]
    
    output_columns = [
        "ball_actions_with_jerseys"  # Ball actions with assigned jersey numbers
    ]
    
    def __init__(self,
                 max_distance: float = 200.0,
                 use_ball_position: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.max_distance = max_distance
        self.use_ball_position = use_ball_position
    
    def preprocess_image_level(self):
        """Match all ball actions to players before frame processing."""
        logger.info("Matching ball actions to player jersey numbers...")
        
        ball_actions = self.state.get('ball_actions', [])
        if not ball_actions:
            logger.warning("No ball actions found in state")
            return
        
        # Get all detections with track IDs and jersey numbers
        # This requires accessing the tracker state from previous modules
        # Implementation depends on TrackLab's state management
        
        matched_actions = []
        unmatched_count = 0
        
        for action in ball_actions:
            frame_num = int(action['position'])
            
            # Get all players visible at this frame
            # TODO: Access tracker state to get detections at frame_num
            players_at_frame = self._get_players_at_frame(frame_num)
            
            if not players_at_frame:
                unmatched_count += 1
                matched_actions.append({**action, 'jersey': None, 'match_status': 'no_players'})
                continue
            
            # Find nearest player
            nearest = self._find_nearest_player(players_at_frame, action)
            
            if nearest:
                matched_actions.append({
                    **action,
                    'jersey': nearest.get('jersey_number'),
                    'track_id': nearest.get('track_id'),
                    'match_status': 'matched'
                })
            else:
                unmatched_count += 1
                matched_actions.append({**action, 'jersey': None, 'match_status': 'no_nearby_player'})
        
        self.state['ball_actions_with_jerseys'] = matched_actions
        
        logger.info(f"Matched {len(matched_actions) - unmatched_count}/{len(ball_actions)} actions to players")
    
    def _get_players_at_frame(self, frame_num: int):
        """Get all player detections at a specific frame."""
        # TODO: Implement based on TrackLab's tracker state structure
        # This needs to access the tracking results from previous modules
        return []
    
    def _find_nearest_player(self, players, action):
        """Find the player nearest to the action location."""
        # TODO: Implement spatial matching logic
        # Calculate distance from each player bbox to action location
        return None
    
    def process(self, image, detections, metadata):
        """Process individual frame (pass through)."""
        return detections
