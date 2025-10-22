#!/usr/bin/env python3
"""
Match ball actions to player jersey numbers using tracking data.

This script takes:
1. Ball action predictions (with frame numbers)
2. Player tracking data (with bboxes and jersey numbers)

And outputs ball actions with assigned jersey numbers.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np


def parse_bbox_ltwh(bbox_str: str) -> np.ndarray:
    """Parse bbox string to numpy array [left, top, width, height]."""
    # Remove brackets and parse
    bbox_str = bbox_str.strip('[]').replace('\n', ' ')
    values = [float(x) for x in bbox_str.split() if x]
    return np.array(values)


def get_bbox_center(bbox_ltwh: np.ndarray) -> tuple:
    """Get center point of bounding box."""
    left, top, width, height = bbox_ltwh
    center_x = left + width / 2
    center_y = top + height / 2
    return (center_x, center_y)


def get_bbox_bottom_center(bbox_ltwh: np.ndarray) -> tuple:
    """Get bottom center point of bounding box (feet position)."""
    left, top, width, height = bbox_ltwh
    center_x = left + width / 2
    bottom_y = top + height  # Bottom of bbox
    return (center_x, bottom_y)


def calculate_distance(point1: tuple, point2: tuple) -> float:
    """Calculate Euclidean distance between two points."""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def frame_id_to_number(image_id: str) -> int:
    """Convert image_id like '2021000001' to frame number 1."""
    # Format: 2021NNNNNN where NNNNNN is the frame number
    return int(image_id[-6:])


def get_players_at_frame(tracking_data: List[Dict], frame_num: int) -> List[Dict]:
    """
    Get all player detections at a specific frame.
    
    Args:
        tracking_data: List of tracking records
        frame_num: Frame number (1-indexed)
        
    Returns:
        List of player records at that frame
    """
    players = []
    for record in tracking_data:
        image_id = record['image_id']
        record_frame = frame_id_to_number(image_id)
        
        if record_frame == frame_num:
            # Only include records with jersey numbers
            if record.get('jersey_number') and record['jersey_number'] != 'NaN':
                try:
                    bbox = parse_bbox_ltwh(record['bbox_ltwh'])
                    players.append({
                        'track_id': record['track_id'],
                        'jersey_number': record['jersey_number'],
                        'bbox': bbox,
                        'team': record.get('team'),
                        'bbox_conf': record.get('bbox_conf', 0.0),
                        'jersey_conf': record.get('jersey_number_confidence', 0.0)
                    })
                except Exception as e:
                    # Skip records with parsing errors
                    continue
    
    return players


def find_nearest_player(players: List[Dict], 
                       action_frame: int,
                       max_distance: float = 300.0,
                       use_feet_position: bool = True) -> Optional[Dict]:
    """
    Find the player nearest to the center of the frame (ball action location).
    
    Since we don't have ball position, we use the center of the frame as a proxy
    for the action location, as most actions happen near the ball.
    
    Args:
        players: List of player records at the action frame
        action_frame: Frame number of the action
        max_distance: Maximum distance (pixels) to consider
        use_feet_position: Use bottom center of bbox (feet) instead of center
        
    Returns:
        Nearest player dict or None if no player within max_distance
    """
    if not players:
        return None
    
    # Assume frame size is 1920x1080 (standard HD)
    # Action location is assumed to be near the center
    frame_center = (960, 540)
    
    nearest_player = None
    min_distance = float('inf')
    
    for player in players:
        bbox = player['bbox']
        
        if use_feet_position:
            player_pos = get_bbox_bottom_center(bbox)
        else:
            player_pos = get_bbox_center(bbox)
        
        distance = calculate_distance(player_pos, frame_center)
        
        if distance < min_distance and distance <= max_distance:
            min_distance = distance
            nearest_player = player
    
    if nearest_player:
        nearest_player['distance'] = min_distance
    
    return nearest_player


def match_actions_to_players(ball_actions: List[Dict], 
                             tracking_data: List[Dict],
                             max_distance: float = 300.0) -> List[Dict]:
    """
    Match ball actions to player jersey numbers.
    
    Args:
        ball_actions: List of ball action predictions with 'position' (frame)
        tracking_data: List of tracking records
        max_distance: Maximum distance for matching
        
    Returns:
        List of ball actions with jersey numbers assigned
    """
    print(f"Matching {len(ball_actions)} ball actions to players...")
    print(f"Tracking data has {len(tracking_data)} records")
    
    # Build frame index for faster lookup
    print("Building frame index...")
    frame_index = {}
    for record in tracking_data:
        image_id = record['image_id']
        frame_num = frame_id_to_number(image_id)
        if frame_num not in frame_index:
            frame_index[frame_num] = []
        frame_index[frame_num].append(record)
    
    print(f"Indexed {len(frame_index)} frames")
    
    matched_actions = []
    stats = {
        'total': len(ball_actions),
        'matched': 0,
        'no_players': 0,
        'no_nearby_player': 0,
        'by_action_type': {}
    }
    
    for i, action in enumerate(ball_actions):
        if i % 500 == 0:
            print(f"Processing action {i+1}/{len(ball_actions)}...")
        
        frame_num = int(action['position'])
        action_label = action['label']
        
        # Update stats by action type
        if action_label not in stats['by_action_type']:
            stats['by_action_type'][action_label] = {'total': 0, 'matched': 0}
        stats['by_action_type'][action_label]['total'] += 1
        
        # Get players at this frame
        players_at_frame = []
        if frame_num in frame_index:
            for record in frame_index[frame_num]:
                if record.get('jersey_number') and record['jersey_number'] != 'NaN':
                    try:
                        bbox = parse_bbox_ltwh(record['bbox_ltwh'])
                        players_at_frame.append({
                            'track_id': record['track_id'],
                            'jersey_number': record['jersey_number'],
                            'bbox': bbox,
                            'team': record.get('team'),
                            'bbox_conf': record.get('bbox_conf', 0.0),
                            'jersey_conf': record.get('jersey_number_confidence', 0.0)
                        })
                    except:
                        continue
        
        if not players_at_frame:
            stats['no_players'] += 1
            matched_actions.append({
                **action,
                'jersey_number': None,
                'track_id': None,
                'team': None,
                'match_status': 'no_players',
                'distance': None
            })
            continue
        
        # Find nearest player
        nearest = find_nearest_player(players_at_frame, frame_num, max_distance)
        
        if nearest:
            stats['matched'] += 1
            stats['by_action_type'][action_label]['matched'] += 1
            matched_actions.append({
                **action,
                'jersey_number': nearest['jersey_number'],
                'track_id': int(nearest['track_id']),
                'team': nearest['team'],
                'match_status': 'matched',
                'distance': nearest['distance'],
                'bbox_conf': nearest['bbox_conf'],
                'jersey_conf': nearest['jersey_conf']
            })
        else:
            stats['no_nearby_player'] += 1
            matched_actions.append({
                **action,
                'jersey_number': None,
                'track_id': None,
                'team': None,
                'match_status': 'no_nearby_player',
                'distance': None
            })
    
    # Print statistics
    print("\n" + "="*60)
    print("MATCHING STATISTICS")
    print("="*60)
    print(f"Total actions: {stats['total']}")
    print(f"✅ Matched: {stats['matched']} ({stats['matched']/stats['total']*100:.1f}%)")
    print(f"❌ No players in frame: {stats['no_players']}")
    print(f"❌ No nearby player: {stats['no_nearby_player']}")
    
    print(f"\nBy action type:")
    for action_type, type_stats in stats['by_action_type'].items():
        match_rate = type_stats['matched'] / type_stats['total'] * 100
        print(f"  {action_type}: {type_stats['matched']}/{type_stats['total']} ({match_rate:.1f}%)")
    
    # Generate player statistics
    print("\n" + "="*60)
    print("PLAYER STATISTICS")
    print("="*60)
    
    player_stats = {}
    for action in matched_actions:
        if action['match_status'] == 'matched':
            jersey = action['jersey_number']
            team = action['team']
            action_type = action['label']
            
            key = f"{team}_{jersey}"
            if key not in player_stats:
                player_stats[key] = {
                    'jersey_number': jersey,
                    'team': team,
                    'total_actions': 0,
                    'by_type': {}
                }
            
            player_stats[key]['total_actions'] += 1
            if action_type not in player_stats[key]['by_type']:
                player_stats[key]['by_type'][action_type] = 0
            player_stats[key]['by_type'][action_type] += 1
    
    # Sort by total actions
    sorted_players = sorted(player_stats.values(), key=lambda x: x['total_actions'], reverse=True)
    
    print(f"Top 20 players by total actions:")
    for i, player in enumerate(sorted_players[:20], 1):
        jersey = player['jersey_number']
        team = player['team']
        total = player['total_actions']
        actions_str = ', '.join([f"{k}: {v}" for k, v in player['by_type'].items()])
        print(f"  {i}. Jersey #{jersey} ({team}): {total} actions ({actions_str})")
    
    return matched_actions


def main():
    if len(sys.argv) != 4:
        print("Usage: python match_ball_actions_to_tracking.py <ball_actions.json> <tracking_results.json> <output.json>")
        print("\nExample:")
        print("  python match_ball_actions_to_tracking.py \\")
        print("    /workspace/ball-action-spotting/data/ball_action/predictions/.../results_spotting.json \\")
        print("    /workspace/tracking_results.json \\")
        print("    /workspace/matched_results.json")
        sys.exit(1)
    
    ball_actions_file = Path(sys.argv[1])
    tracking_file = Path(sys.argv[2])
    output_file = Path(sys.argv[3])
    
    # Load ball actions
    print(f"Loading ball actions from: {ball_actions_file}")
    with open(ball_actions_file, 'r') as f:
        ball_data = json.load(f)
        ball_actions = ball_data.get('predictions', ball_data.get('UrlLocal', []))
    
    print(f"Loaded {len(ball_actions)} ball actions")
    
    # Load tracking data
    print(f"Loading tracking data from: {tracking_file}")
    with open(tracking_file, 'r') as f:
        tracking_data_full = json.load(f)
        tracking_data = tracking_data_full['tracks']
    
    print(f"Loaded {len(tracking_data)} tracking records")
    
    # Match actions to players
    matched_actions = match_actions_to_players(ball_actions, tracking_data, max_distance=300.0)
    
    # Save results
    output_data = {
        'metadata': {
            'ball_actions_source': str(ball_actions_file),
            'tracking_source': str(tracking_file),
            'total_actions': len(matched_actions),
            'matched_actions': sum(1 for a in matched_actions if a['match_status'] == 'matched')
        },
        'actions': matched_actions
    }
    
    print(f"\n💾 Saving matched results to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print("✅ Done!")


if __name__ == "__main__":
    main()
