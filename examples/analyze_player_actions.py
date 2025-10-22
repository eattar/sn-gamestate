#!/usr/bin/env python3
"""
Simple utility script to analyze ball actions for a specific player.

This script provides a simple command-line interface to:
- Load existing ball action predictions
- Filter by jersey number
- Generate a report

Usage:
    python examples/analyze_player_actions.py \
        --predictions /path/to/results_spotting.json \
        --jersey-number 10 \
        --output player_10_stats.json
"""

import argparse
import json
from pathlib import Path
import sys

# Add sn_gamestate to path if running as standalone script
sys.path.insert(0, str(Path(__file__).parent.parent))

from sn_gamestate.ball_action import BallActionSpotting


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze ball actions for a specific player from existing predictions"
    )
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to results_spotting.json file",
    )
    parser.add_argument(
        "--jersey-number",
        type=int,
        required=True,
        help="Jersey number of the player to analyze",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="player_report.json",
        help="Output file path for the report",
    )
    parser.add_argument(
        "--ball-action-repo",
        type=str,
        default="/Users/eattar/repos/ball-action-spotting",
        help="Path to ball-action-spotting repository",
    )
    return parser.parse_args()


def create_example_jersey_mapping(predictions_dict, target_jersey):
    """Create an example jersey mapping.
    
    In production, you would integrate this with your tracking system
    to get the actual jersey numbers for each action.
    """
    jersey_mapping = {}
    
    # Example: Assign jersey numbers to actions
    # This is where you'd integrate with your tracking/reid system
    for idx, pred in enumerate(predictions_dict.get("predictions", [])):
        half = int(pred["half"])
        position = int(pred["position"])
        
        # Mock assignment for demonstration
        jersey_mapping[(half, position)] = (idx % 15) + 1
        
        # Assign some actions to target player
        if idx % 4 == 0:
            jersey_mapping[(half, position)] = target_jersey
    
    return jersey_mapping


def main():
    """Main execution function."""
    args = parse_args()
    
    print("\n" + "="*70)
    print(" "*20 + "Player Ball Action Analysis")
    print("="*70)
    
    # Initialize API
    try:
        ball_action = BallActionSpotting(
            ball_action_repo_path=args.ball_action_repo,
        )
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nMake sure the ball-action-spotting repository is available at:")
        print(f"  {args.ball_action_repo}")
        return 1
    
    # Load predictions
    print(f"\n📂 Loading predictions from:")
    print(f"   {args.predictions}")
    
    try:
        predictions = ball_action.load_predictions(args.predictions)
        total_actions = len(predictions.get("predictions", []))
        print(f"\n✓ Loaded {total_actions} total ball actions")
    except Exception as e:
        print(f"\n✗ Error loading predictions: {e}")
        return 1
    
    # Show overall statistics
    print("\n" + "-"*70)
    print("Overall Action Statistics:")
    print("-"*70)
    
    overall_stats = ball_action.get_player_stats(predictions)
    for action, count in sorted(overall_stats.items(), key=lambda x: x[1], reverse=True):
        print(f"  {action:25s}: {count:3d} actions")
    
    # Create jersey mapping (integrate with your tracking system here)
    print("\n" + "="*70)
    print("⚠️  NOTE: Using example jersey mapping for demonstration")
    print("    In production, integrate with your tracking system to get")
    print("    accurate player-action associations")
    print("="*70)
    
    jersey_mapping = create_example_jersey_mapping(predictions, args.jersey_number)
    
    # Filter by jersey number
    print(f"\n🔍 Filtering actions for player #{args.jersey_number}...")
    
    try:
        filtered_predictions = ball_action.filter_by_jersey_number(
            predictions,
            jersey_number=args.jersey_number,
            jersey_mapping=jersey_mapping,
        )
        filtered_count = len(filtered_predictions.get("predictions", []))
        print(f"✓ Found {filtered_count} actions by player #{args.jersey_number}")
    except Exception as e:
        print(f"✗ Error: {e}")
        return 1
    
    # Generate report
    print(f"\n📊 Generating detailed report...")
    
    try:
        report = ball_action.generate_player_report(
            filtered_predictions,
            jersey_number=args.jersey_number,
            output_path=args.output,
        )
    except Exception as e:
        print(f"✗ Error generating report: {e}")
        return 1
    
    print("\n" + "="*70)
    print("✅ Analysis Complete!")
    print("="*70)
    print(f"Report saved to: {args.output}")
    print("\nYou can now use this JSON file for further analysis or visualization.")
    print("="*70 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
