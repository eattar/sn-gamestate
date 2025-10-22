#!/usr/bin/env python3
"""
Demo script for ball action spotting with sample data.

This script demonstrates the ball action module functionality using sample data,
so you can test it without needing actual predictions from a game.

Usage:
    python examples/demo_ball_action.py --jersey-number 10
"""

import argparse
import json
from pathlib import Path
import sys

# Add sn_gamestate to path if running as standalone script
sys.path.insert(0, str(Path(__file__).parent.parent))

from sn_gamestate.ball_action import BallActionSpotting


def create_sample_predictions():
    """Create sample ball action predictions for demonstration."""
    return {
        "UrlLocal": "demo_game/2024-10-22 - Team A vs Team B",
        "predictions": [
            {
                "gameTime": "1 - 05:23",
                "label": "PASS",
                "position": "323000",
                "half": "1",
                "confidence": "0.95"
            },
            {
                "gameTime": "1 - 08:45",
                "label": "DRIVE",
                "position": "525000",
                "half": "1",
                "confidence": "0.89"
            },
            {
                "gameTime": "1 - 12:10",
                "label": "SHOT",
                "position": "730000",
                "half": "1",
                "confidence": "0.92"
            },
            {
                "gameTime": "1 - 15:30",
                "label": "PASS",
                "position": "930000",
                "half": "1",
                "confidence": "0.87"
            },
            {
                "gameTime": "1 - 18:20",
                "label": "CROSS",
                "position": "1100000",
                "half": "1",
                "confidence": "0.91"
            },
            {
                "gameTime": "1 - 22:45",
                "label": "HEADER",
                "position": "1365000",
                "half": "1",
                "confidence": "0.85"
            },
            {
                "gameTime": "1 - 28:10",
                "label": "PASS",
                "position": "1690000",
                "half": "1",
                "confidence": "0.93"
            },
            {
                "gameTime": "1 - 32:55",
                "label": "DRIVE",
                "position": "1975000",
                "half": "1",
                "confidence": "0.88"
            },
            {
                "gameTime": "1 - 38:40",
                "label": "PASS",
                "position": "2320000",
                "half": "1",
                "confidence": "0.90"
            },
            {
                "gameTime": "1 - 42:15",
                "label": "SHOT",
                "position": "2535000",
                "half": "1",
                "confidence": "0.94"
            },
            {
                "gameTime": "2 - 03:20",
                "label": "PASS",
                "position": "200000",
                "half": "2",
                "confidence": "0.89"
            },
            {
                "gameTime": "2 - 07:45",
                "label": "TACKLE",
                "position": "465000",
                "half": "2",
                "confidence": "0.86"
            },
            {
                "gameTime": "2 - 12:30",
                "label": "PASS",
                "position": "750000",
                "half": "2",
                "confidence": "0.91"
            },
            {
                "gameTime": "2 - 18:55",
                "label": "CROSS",
                "position": "1135000",
                "half": "2",
                "confidence": "0.87"
            },
            {
                "gameTime": "2 - 25:10",
                "label": "GOAL",
                "position": "1510000",
                "half": "2",
                "confidence": "0.98"
            },
        ]
    }


def create_sample_jersey_mapping(predictions_dict, target_jersey):
    """Create sample jersey mapping for demonstration.
    
    This assigns jersey numbers to actions. In a real scenario,
    this would come from your tracking system.
    """
    jersey_mapping = {}
    
    # Assign jerseys: every 3rd action goes to target player
    for idx, pred in enumerate(predictions_dict.get("predictions", [])):
        half = int(pred["half"])
        position = int(pred["position"])
        
        if idx % 3 == 0:
            # Assign to target player
            jersey_mapping[(half, position)] = target_jersey
        elif idx % 3 == 1:
            # Assign to player #7
            jersey_mapping[(half, position)] = 7
        else:
            # Assign to player #23
            jersey_mapping[(half, position)] = 23
    
    return jersey_mapping


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Demo ball action spotting with sample data"
    )
    parser.add_argument(
        "--jersey-number",
        type=int,
        default=10,
        help="Jersey number to analyze (default: 10)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="demo_output",
        help="Directory to save demo results (default: demo_output)",
    )
    parser.add_argument(
        "--ball-action-repo",
        type=str,
        default="/Users/eattar/repos/ball-action-spotting",
        help="Path to ball-action-spotting repository",
    )
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()
    
    print("\n" + "="*70)
    print(" "*15 + "Ball Action Spotting - DEMO MODE")
    print("="*70)
    print("\n⚠️  NOTE: This is a demonstration using sample data")
    print("   For real predictions, use ball_action_example.py with actual game videos\n")
    print("="*70)
    
    # Initialize API
    try:
        ball_action = BallActionSpotting(
            ball_action_repo_path=args.ball_action_repo,
        )
        print("✓ Ball action API initialized")
    except Exception as e:
        print(f"\n⚠️  Warning: Could not initialize full API: {e}")
        print("   Continuing with demo using mock API...")
        ball_action = None
    
    # Create sample predictions
    print("\n📊 Creating sample ball action predictions...")
    predictions = create_sample_predictions()
    total_actions = len(predictions.get("predictions", []))
    print(f"✓ Generated {total_actions} sample ball actions")
    
    # Show overall statistics
    print("\n" + "-"*70)
    print("Overall Action Statistics (All Players):")
    print("-"*70)
    
    action_counts = {}
    for pred in predictions["predictions"]:
        label = pred["label"]
        action_counts[label] = action_counts.get(label, 0) + 1
    
    for action, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {action:25s}: {count:3d} actions")
    
    # Create jersey mapping
    print("\n" + "="*70)
    print("🔍 Assigning actions to players...")
    print("="*70)
    
    jersey_mapping = create_sample_jersey_mapping(predictions, args.jersey_number)
    
    # Count actions per player
    player_action_counts = {}
    for (half, pos), jersey in jersey_mapping.items():
        player_action_counts[jersey] = player_action_counts.get(jersey, 0) + 1
    
    print("\nActions per player:")
    for jersey in sorted(player_action_counts.keys()):
        count = player_action_counts[jersey]
        marker = " ← TARGET PLAYER" if jersey == args.jersey_number else ""
        print(f"  Player #{jersey:2d}: {count:2d} actions{marker}")
    
    # Filter by jersey number
    print(f"\n" + "="*70)
    print(f"🎯 Filtering actions for player #{args.jersey_number}...")
    print("="*70)
    
    filtered_predictions = {
        "UrlLocal": predictions["UrlLocal"],
        "predictions": []
    }
    
    for pred in predictions["predictions"]:
        half = int(pred["half"])
        position = int(pred["position"])
        
        if jersey_mapping.get((half, position)) == args.jersey_number:
            filtered_predictions["predictions"].append(pred)
    
    filtered_count = len(filtered_predictions["predictions"])
    print(f"✓ Found {filtered_count} actions by player #{args.jersey_number}")
    
    # Generate statistics
    print(f"\n" + "-"*70)
    print(f"Player #{args.jersey_number} - Action Breakdown:")
    print("-"*70)
    
    player_stats = {}
    for pred in filtered_predictions["predictions"]:
        label = pred["label"]
        player_stats[label] = player_stats.get(label, 0) + 1
    
    for action, count in sorted(player_stats.items(), key=lambda x: x[1], reverse=True):
        print(f"  {action:25s}: {count:3d} actions")
    
    # Show detailed actions
    print(f"\n" + "-"*70)
    print(f"Player #{args.jersey_number} - Detailed Actions:")
    print("-"*70)
    
    for pred in filtered_predictions["predictions"]:
        print(f"  {pred['gameTime']:15s} | {pred['label']:20s} | Confidence: {pred['confidence']}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    print(f"\n📁 Saving results to {output_dir}/...")
    
    # Save filtered predictions
    filtered_path = output_dir / f"player_{args.jersey_number}_actions.json"
    with open(filtered_path, "w") as f:
        json.dump(filtered_predictions, f, indent=4)
    print(f"✓ Filtered actions: {filtered_path}")
    
    # Save report
    report = {
        "jersey_number": args.jersey_number,
        "game": predictions["UrlLocal"],
        "total_actions": filtered_count,
        "action_statistics": player_stats,
        "detailed_actions": filtered_predictions["predictions"],
    }
    
    report_path = output_dir / f"player_{args.jersey_number}_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
    print(f"✓ Player report: {report_path}")
    
    # Save all predictions for reference
    all_predictions_path = output_dir / "all_predictions_sample.json"
    with open(all_predictions_path, "w") as f:
        json.dump(predictions, f, indent=4)
    print(f"✓ All predictions (sample): {all_predictions_path}")
    
    print("\n" + "="*70)
    print("✅ Demo Complete!")
    print("="*70)
    print(f"\nResults saved to: {output_dir}/")
    print(f"  - player_{args.jersey_number}_report.json")
    print(f"  - player_{args.jersey_number}_actions.json")
    print(f"  - all_predictions_sample.json")
    print("\n" + "="*70)
    print("Next Steps:")
    print("  1. Examine the generated JSON files")
    print("  2. For real predictions, use: python examples/ball_action_example.py")
    print("  3. Integrate with your tracking system for accurate jersey mapping")
    print("="*70 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
