"""
Integration helper for connecting ball actions with player tracking.

This module helps you create the jersey_mapping that connects
ball actions (detected by ball-action-spotting) with specific players
(tracked by sn-gamestate tracking system).

Usage example:
    from sn_gamestate.ball_action.tracking_integration import create_jersey_mapping_from_tracklab
    
    # Load your tracking results (from TrackLab/sn-gamestate)
    tracking_results = load_tracklab_results("path/to/tracking_state.pkl")
    
    # Load ball action predictions
    ball_actions = load_ball_predictions("path/to/results_spotting.json")
    
    # Create the mapping
    jersey_mapping = create_jersey_mapping_from_tracklab(
        tracking_results, 
        ball_actions,
        distance_threshold=2.0  # meters on pitch
    )
    
    # Now filter actions by jersey number
    filtered = ball_action.filter_by_jersey_number(
        ball_actions, 
        jersey_number=10, 
        jersey_mapping=jersey_mapping
    )
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any
from pathlib import Path
import json


def create_jersey_mapping_from_tracklab(
    tracking_results: Any,  # TrackLab tracker state or similar
    ball_actions: Dict,
    distance_threshold: float = 2.0,  # meters
) -> Dict[Tuple[int, int], int]:
    """Create jersey mapping from TrackLab tracking results.
    
    This function maps each ball action to the nearest player at that timestamp.
    
    Args:
        tracking_results: TrackLab tracker state with player positions and jersey numbers
        ball_actions: Ball action predictions from ball-action-spotting
        distance_threshold: Maximum distance (in meters) to consider a player
                           as performing the action
    
    Returns:
        Dictionary mapping (half, position_ms) -> jersey_number
    
    Example:
        jersey_mapping = create_jersey_mapping_from_tracklab(
            tracking_results=tracklab_state,
            ball_actions=predictions,
            distance_threshold=2.0
        )
    """
    jersey_mapping = {}
    
    for action in ball_actions.get("predictions", []):
        half = int(action["half"])
        position_ms = int(action["position"])
        
        # Convert milliseconds to frame/timestamp used by your tracking system
        # Adjust this based on your video FPS
        fps = 25.0  # SoccerNet standard
        frame_idx = int((position_ms / 1000.0) * fps)
        
        # Find the nearest player at this timestamp
        # TODO: Implement this based on your tracking system structure
        nearest_player = find_nearest_player_at_timestamp(
            tracking_results,
            half=half,
            frame_idx=frame_idx,
            distance_threshold=distance_threshold
        )
        
        if nearest_player and nearest_player.get("jersey_number"):
            jersey_mapping[(half, position_ms)] = nearest_player["jersey_number"]
    
    return jersey_mapping


def find_nearest_player_at_timestamp(
    tracking_results: Any,
    half: int,
    frame_idx: int,
    distance_threshold: float = 2.0,
) -> Optional[Dict]:
    """Find the nearest player to the ball at a specific timestamp.
    
    Args:
        tracking_results: Your tracking system results
        half: Game half (1 or 2)
        frame_idx: Frame index
        distance_threshold: Maximum distance in meters
    
    Returns:
        Dictionary with player info including jersey_number, or None
    
    TODO: Implement this based on your specific tracking system structure.
    
    Example implementation structure:
        1. Get all player detections at frame_idx in the specified half
        2. Get ball position at frame_idx (if available)
        3. Calculate distance from each player to ball
        4. Return the nearest player within distance_threshold
    """
    # PLACEHOLDER IMPLEMENTATION
    # Replace this with your actual tracking system logic
    
    # Example structure:
    # players_at_frame = tracking_results.get_detections(half=half, frame=frame_idx)
    # ball_position = tracking_results.get_ball_position(half=half, frame=frame_idx)
    # 
    # if not ball_position:
    #     # If no ball position, use heuristics (e.g., center of action)
    #     return None
    # 
    # nearest_player = None
    # min_distance = float('inf')
    # 
    # for player in players_at_frame:
    #     if player.role not in ['player', 'goalkeeper']:
    #         continue
    #     
    #     distance = calculate_pitch_distance(player.position, ball_position)
    #     
    #     if distance < min_distance and distance < distance_threshold:
    #         min_distance = distance
    #         nearest_player = {
    #             'jersey_number': player.jersey_number,
    #             'team': player.team,
    #             'position': player.position,
    #             'distance': distance
    #         }
    # 
    # return nearest_player
    
    raise NotImplementedError(
        "You need to implement this function based on your tracking system. "
        "See the docstring and example structure above."
    )


def calculate_pitch_distance(pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance on pitch (in meters).
    
    Args:
        pos1: (x, y) position in pitch coordinates (meters)
        pos2: (x, y) position in pitch coordinates (meters)
    
    Returns:
        Distance in meters
    """
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)


def load_tracklab_results(results_path: Path) -> Any:
    """Load tracking results from TrackLab.
    
    Args:
        results_path: Path to tracker state pickle file
    
    Returns:
        Loaded tracking results
    
    TODO: Implement based on your TrackLab output format
    """
    # Example:
    # import pickle
    # with open(results_path, 'rb') as f:
    #     return pickle.load(f)
    raise NotImplementedError("Implement based on your TrackLab output format")


# Example usage function
def example_integration():
    """Example of complete integration workflow."""
    from sn_gamestate.ball_action import BallActionSpotting
    
    # Initialize ball action module
    ball_action = BallActionSpotting(
        ball_action_repo_path="/Users/eattar/repos/ball-action-spotting"
    )
    
    # Load ball action predictions
    predictions = ball_action.load_predictions(
        "data/ball_action/predictions/experiment/game/results_spotting.json"
    )
    
    # Load your tracking results
    # tracking_results = load_tracklab_results("path/to/tracker_state.pkl")
    
    # Create jersey mapping (integrate with your tracking)
    # jersey_mapping = create_jersey_mapping_from_tracklab(
    #     tracking_results, predictions
    # )
    
    # For now, use a manual mapping as example:
    jersey_mapping = {
        (1, 323000): 10,  # half 1, 323 seconds -> player #10
        (1, 525000): 7,   # half 1, 525 seconds -> player #7
        # ... etc
    }
    
    # Filter by jersey number
    player_actions = ball_action.filter_by_jersey_number(
        predictions,
        jersey_number=10,
        jersey_mapping=jersey_mapping
    )
    
    # Generate report
    report = ball_action.generate_player_report(
        player_actions,
        jersey_number=10,
        output_path="player_10_report.json"
    )
    
    return report


if __name__ == "__main__":
    print("This module provides utilities for integrating ball actions with player tracking.")
    print("See the docstrings and example_integration() function for usage.")
