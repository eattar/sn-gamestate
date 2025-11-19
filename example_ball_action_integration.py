#!/usr/bin/env python3
"""
Simple example script demonstrating the ball-action integration.

This is a minimal working example that shows the core integration logic
without the full pipeline complexity.
"""

import json
from pathlib import Path

def example_usage():
    """
    Example showing how the integration works conceptually
    """
    
    print("="*60)
    print("Ball Action Integration - Simple Example")
    print("="*60)
    
    # Example 1: Basic usage
    print("\n1. Basic Usage:")
    print("-" * 40)
    command = """
    python run_player_ball_actions.py \\
        --video match.mp4 \\
        --team left \\
        --jersey 10
    """
    print(command)
    
    # Example 2: With output file
    print("\n2. With Custom Output:")
    print("-" * 40)
    command = """
    python run_player_ball_actions.py \\
        --video match.mp4 \\
        --team right \\
        --jersey 7 \\
        --output player7_results.json
    """
    print(command)
    
    # Example 3: Using cached state
    print("\n3. Using Cached Tracking State (Faster):")
    print("-" * 40)
    command = """
    # First run - generates cache
    python run_player_ball_actions.py \\
        --video match.mp4 \\
        --team left \\
        --jersey 10
    
    # Subsequent runs - uses cache
    python run_player_ball_actions.py \\
        --video match.mp4 \\
        --team left \\
        --jersey 10 \\
        --state-cache tracking_state_temp.pklz \\
        --skip-tracking
    """
    print(command)
    
    # Example output
    print("\n4. Example Output:")
    print("-" * 40)
    
    example_output = {
        "player": {
            "team": "left",
            "jersey": 10
        },
        "video": "match.mp4",
        "total_actions": 42,
        "action_counts": {
            "PASS": 28,
            "DRIVE": 14
        },
        "actions": [
            {
                "action": "PASS",
                "frame": 3350,
                "confidence": 0.876,
                "player_frame": 3348,
                "frame_offset": 2,
                "time": "2:14"
            },
            {
                "action": "DRIVE",
                "frame": 8300,
                "confidence": 0.823,
                "player_frame": 8305,
                "frame_offset": 5,
                "time": "5:32"
            },
            {
                "action": "PASS",
                "frame": 12450,
                "confidence": 0.891,
                "player_frame": 12447,
                "frame_offset": 3,
                "time": "8:18"
            }
        ]
    }
    
    print(json.dumps(example_output, indent=2))
    
    # Pipeline steps
    print("\n5. Pipeline Steps:")
    print("-" * 40)
    steps = """
    Step 1: Run SN-GameState tracking
            → Detects players, assigns track IDs
            → Performs jersey OCR
            → Clusters into teams (left/right)
            
    Step 2: Filter by team + jersey
            → Finds all detections for specific player
            → Returns: frames where player is visible
            
    Step 3: Run ball-action detection  
            → Loads model (ball_finetune_long_004)
            → Detects PASS and DRIVE actions
            → Returns: frames with actions
            
    Step 4: Match actions to player
            → For each action frame
            → Find player detection within ±50 frames
            → Record closest match
            
    Step 5: Generate JSON output
            → Convert frames to timestamps
            → Count actions by type
            → Save results
    """
    print(steps)
    
    print("\n" + "="*60)
    print("For full documentation, see: BALL_ACTION_INTEGRATION.md")
    print("="*60)


def mock_integration_test():
    """
    Mock test showing the data flow without running actual models
    """
    print("\n" + "="*60)
    print("Mock Integration Test (No Models Required)")
    print("="*60)
    
    # Mock tracking data
    print("\n[Step 1] Mock Tracking Data:")
    tracking_data = {
        'frames': [100, 101, 102, 150, 151],
        'track_ids': [5, 5, 5, 5, 5],
        'jerseys': [10, 10, 10, 10, 10],
        'teams': ['left', 'left', 'left', 'left', 'left'],
        'bboxes': [
            [120, 150, 80, 120],
            [125, 155, 80, 120],
            [130, 160, 80, 120],
            [200, 200, 85, 125],
            [205, 205, 85, 125]
        ]
    }
    print(f"  Tracked frames: {tracking_data['frames']}")
    print(f"  Player: Team={tracking_data['teams'][0]}, Jersey={tracking_data['jerseys'][0]}")
    
    # Mock ball actions
    print("\n[Step 2] Mock Ball Actions:")
    ball_actions = [
        {'frame': 102, 'action': 'PASS', 'confidence': 0.87},
        {'frame': 148, 'action': 'DRIVE', 'confidence': 0.82},
        {'frame': 250, 'action': 'PASS', 'confidence': 0.91}
    ]
    print(f"  Detected actions: {len(ball_actions)}")
    for action in ball_actions:
        print(f"    Frame {action['frame']}: {action['action']} (conf: {action['confidence']:.2f})")
    
    # Mock matching
    print("\n[Step 3] Mock Action-Player Matching:")
    matched = []
    for action in ball_actions:
        # Find closest player frame
        closest_frame = min(tracking_data['frames'], 
                          key=lambda f: abs(f - action['frame']))
        offset = abs(closest_frame - action['frame'])
        
        if offset <= 50:  # Within window
            matched.append({
                **action,
                'player_frame': closest_frame,
                'frame_offset': offset,
                'matched': True
            })
            print(f"  ✓ Frame {action['frame']} → Player frame {closest_frame} (offset: {offset})")
        else:
            print(f"  ✗ Frame {action['frame']} → No player within window")
    
    # Mock output
    print("\n[Step 4] Mock JSON Output:")
    output = {
        'player': {'team': 'left', 'jersey': 10},
        'total_actions': len(matched),
        'actions': matched
    }
    print(json.dumps(output, indent=2))
    
    print("\n✓ Mock test complete - Integration logic verified!")


if __name__ == '__main__':
    example_usage()
    mock_integration_test()
