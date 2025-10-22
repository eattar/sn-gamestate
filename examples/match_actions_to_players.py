"""
Match ball actions with player tracking data to assign jersey numbers.

This script takes:
1. Ball action predictions from ball-action-spotting (results_spotting.json)
2. Player tracking data from your tracking system
3. Matches each action to the nearest player at that timestamp

Output: Enhanced predictions with jersey numbers assigned to each action.

Usage:
    python match_actions_to_players.py \
        --actions /path/to/results_spotting.json \
        --tracking /path/to/tracking_results.json \
        --output /path/to/actions_with_jerseys.json
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import math


def load_json(path: str) -> dict:
    """Load JSON file"""
    with open(path, 'r') as f:
        return json.load(f)


def save_json(data: dict, path: str):
    """Save JSON file"""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def get_bbox_center(bbox: List[float]) -> Tuple[float, float]:
    """
    Get center point of bounding box.
    
    Args:
        bbox: Bounding box in format [x, y, w, h] or [x1, y1, x2, y2]
              Adjust this based on your tracking system's format
    
    Returns:
        (center_x, center_y)
    """
    # Assuming format is [x, y, w, h] - adjust if needed
    if len(bbox) == 4:
        x, y, w, h = bbox
        return (x + w/2, y + h/2)
    else:
        # If format is [x1, y1, x2, y2]
        x1, y1, x2, y2 = bbox[:4]
        return ((x1 + x2)/2, (y1 + y2)/2)


def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two points"""
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def get_players_at_frame(tracking_data: dict, frame_number: int) -> List[dict]:
    """
    Extract all players visible at a specific frame.
    
    Args:
        tracking_data: Your tracking system's output
        frame_number: Frame number from ball action prediction
    
    Returns:
        List of player dictionaries with 'jersey', 'bbox', etc.
    
    TODO: Adapt this to your tracking system's format!
    Examples of different tracking formats:
    
    Format 1 - Frame-based:
        {
            "frames": [
                {
                    "frame_id": 1520,
                    "players": [
                        {"id": 1, "jersey": 10, "bbox": [100, 200, 50, 80]},
                        {"id": 2, "jersey": 7, "bbox": [300, 150, 55, 85]}
                    ]
                }
            ]
        }
    
    Format 2 - Track-based:
        {
            "tracks": [
                {
                    "track_id": 1,
                    "jersey": 10,
                    "detections": [
                        {"frame": 1520, "bbox": [100, 200, 50, 80]},
                        {"frame": 1521, "bbox": [102, 201, 50, 80]}
                    ]
                }
            ]
        }
    """
    players = []
    
    # TODO: Implement based on your tracking format
    # This is a placeholder - adapt to your actual data structure
    
    # Example for Format 1 (frame-based):
    if "frames" in tracking_data:
        for frame_data in tracking_data["frames"]:
            if frame_data.get("frame_id") == frame_number:
                return frame_data.get("players", [])
    
    # Example for Format 2 (track-based):
    elif "tracks" in tracking_data:
        for track in tracking_data["tracks"]:
            for detection in track.get("detections", []):
                if detection.get("frame") == frame_number:
                    players.append({
                        "track_id": track["track_id"],
                        "jersey": track.get("jersey"),
                        "bbox": detection.get("bbox"),
                        "team": track.get("team")
                    })
        return players
    
    return players


def find_nearest_player(
    action_position: Optional[Tuple[float, float]],
    players: List[dict],
    max_distance: float = None
) -> Optional[dict]:
    """
    Find the player nearest to the action location.
    
    Args:
        action_position: (x, y) coordinates of action, or None to use bbox centers
        players: List of player dictionaries with bbox information
        max_distance: Maximum distance to consider (pixels), None for no limit
    
    Returns:
        Player dictionary with minimum distance, or None if no players nearby
    """
    if not players:
        return None
    
    min_distance = float('inf')
    nearest_player = None
    
    for player in players:
        bbox = player.get('bbox')
        if not bbox:
            continue
        
        player_center = get_bbox_center(bbox)
        
        # If action position is not provided, we can't determine nearest player
        # In this case, you might want to use other heuristics
        if action_position is None:
            # Option: Return player with largest bbox (assuming they're most prominent)
            # This is a fallback - ideally you'd have ball position from tracking
            continue
        
        distance = calculate_distance(action_position, player_center)
        
        if distance < min_distance:
            min_distance = distance
            nearest_player = player
    
    # Check max_distance threshold
    if max_distance and min_distance > max_distance:
        return None
    
    return nearest_player


def get_ball_position_at_frame(tracking_data: dict, frame_number: int) -> Optional[Tuple[float, float]]:
    """
    Get ball position at a specific frame from tracking data.
    
    Args:
        tracking_data: Your tracking system's output (should include ball tracking)
        frame_number: Frame number
    
    Returns:
        (x, y) ball position or None if not available
    
    TODO: Implement based on your ball tracking data!
    """
    # TODO: Adapt to your ball tracking format
    # Example:
    if "ball_detections" in tracking_data:
        for detection in tracking_data["ball_detections"]:
            if detection.get("frame") == frame_number:
                return (detection["x"], detection["y"])
    
    return None


def match_actions_to_players(
    actions_data: dict,
    tracking_data: dict,
    use_ball_position: bool = True,
    max_distance: float = 200.0
) -> dict:
    """
    Main function to match ball actions with player jersey numbers.
    
    Args:
        actions_data: Ball action predictions (results_spotting.json)
        tracking_data: Player tracking results
        use_ball_position: If True, use ball position; else use player bboxes
        max_distance: Maximum distance (pixels) to assign action to player
    
    Returns:
        Enhanced actions data with jersey numbers
    """
    predictions = actions_data.get('predictions', [])
    enhanced_predictions = []
    
    stats = {
        'total_actions': len(predictions),
        'matched': 0,
        'unmatched': 0,
        'no_players': 0
    }
    
    for i, action in enumerate(predictions):
        if i % 100 == 0:
            print(f"Processing action {i}/{len(predictions)}...")
        
        frame_number = int(action['position'])
        
        # Get all players visible at this frame
        players = get_players_at_frame(tracking_data, frame_number)
        
        if not players:
            stats['no_players'] += 1
            enhanced_action = action.copy()
            enhanced_action['jersey'] = None
            enhanced_action['match_status'] = 'no_players_visible'
            enhanced_predictions.append(enhanced_action)
            continue
        
        # Get ball position (if available)
        ball_position = None
        if use_ball_position:
            ball_position = get_ball_position_at_frame(tracking_data, frame_number)
        
        # Find nearest player
        nearest = find_nearest_player(ball_position, players, max_distance)
        
        # Create enhanced action
        enhanced_action = action.copy()
        if nearest:
            enhanced_action['jersey'] = nearest.get('jersey')
            enhanced_action['player_track_id'] = nearest.get('track_id')
            enhanced_action['team'] = nearest.get('team')
            enhanced_action['match_status'] = 'matched'
            stats['matched'] += 1
        else:
            enhanced_action['jersey'] = None
            enhanced_action['match_status'] = 'no_nearby_player'
            stats['unmatched'] += 1
        
        enhanced_predictions.append(enhanced_action)
    
    # Create output with stats
    output_data = {
        'UrlLocal': actions_data.get('UrlLocal'),
        'predictions': enhanced_predictions,
        'matching_stats': stats
    }
    
    return output_data


def filter_by_jersey(actions_data: dict, jersey_number: int) -> List[dict]:
    """
    Filter actions for a specific jersey number.
    
    Args:
        actions_data: Enhanced actions with jersey numbers
        jersey_number: Jersey number to filter by
    
    Returns:
        List of actions performed by that player
    """
    return [
        action for action in actions_data.get('predictions', [])
        if action.get('jersey') == jersey_number
    ]


def generate_player_stats(actions: List[dict]) -> dict:
    """
    Generate statistics for a player's actions.
    
    Args:
        actions: List of actions by a single player
    
    Returns:
        Dictionary with player statistics
    """
    from collections import Counter
    
    if not actions:
        return {'total_actions': 0}
    
    label_counts = Counter(action['label'] for action in actions)
    
    return {
        'jersey_number': actions[0].get('jersey'),
        'total_actions': len(actions),
        'actions_by_type': dict(label_counts),
        'actions_by_half': {
            '1': len([a for a in actions if a['half'] == '1']),
            '2': len([a for a in actions if a['half'] == '2'])
        },
        'avg_confidence': sum(float(a['confidence']) for a in actions) / len(actions),
        'action_timeline': [
            {
                'time': a['gameTime'],
                'label': a['label'],
                'confidence': float(a['confidence'])
            }
            for a in actions
        ]
    }


def main():
    parser = argparse.ArgumentParser(description='Match ball actions to players')
    parser.add_argument('--actions', required=True, help='Path to results_spotting.json')
    parser.add_argument('--tracking', required=True, help='Path to tracking results JSON')
    parser.add_argument('--output', required=True, help='Path to save enhanced results')
    parser.add_argument('--max-distance', type=float, default=200.0,
                        help='Maximum distance (pixels) to assign action to player')
    parser.add_argument('--jersey', type=int, help='Filter results for specific jersey number')
    
    args = parser.parse_args()
    
    print(f"Loading ball actions from: {args.actions}")
    actions_data = load_json(args.actions)
    
    print(f"Loading tracking data from: {args.tracking}")
    tracking_data = load_json(args.tracking)
    
    print("Matching actions to players...")
    enhanced_data = match_actions_to_players(
        actions_data,
        tracking_data,
        max_distance=args.max_distance
    )
    
    # Print statistics
    stats = enhanced_data['matching_stats']
    print(f"\nMatching Results:")
    print(f"  Total actions: {stats['total_actions']}")
    print(f"  Matched: {stats['matched']} ({stats['matched']/stats['total_actions']*100:.1f}%)")
    print(f"  Unmatched: {stats['unmatched']}")
    print(f"  No players visible: {stats['no_players']}")
    
    # Save full results
    print(f"\nSaving enhanced results to: {args.output}")
    save_json(enhanced_data, args.output)
    
    # If jersey number specified, generate player-specific report
    if args.jersey:
        player_actions = filter_by_jersey(enhanced_data, args.jersey)
        player_stats = generate_player_stats(player_actions)
        
        player_output = args.output.replace('.json', f'_jersey_{args.jersey}.json')
        save_json(player_stats, player_output)
        
        print(f"\nPlayer #{args.jersey} Statistics:")
        print(f"  Total actions: {player_stats['total_actions']}")
        print(f"  Actions by type: {player_stats.get('actions_by_type', {})}")
        print(f"  Saved to: {player_output}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
