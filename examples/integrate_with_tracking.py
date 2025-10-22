"""
Complete end-to-end example integrating ball actions with game state tracking.

This script demonstrates how to:
1. Load game state tracking results (from sn-gamestate/TrackLab)
2. Load ball action predictions (from ball-action-spotting)
3. Map ball actions to specific players using tracking data
4. Generate player-specific action reports

Usage:
    python examples/integrate_with_tracking.py \
        --tracking-results /path/to/tracker_state.pkl \
        --ball-predictions /path/to/results_spotting.json \
        --jersey-number 10 \
        --output player_10_full_report.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sn_gamestate.ball_action import BallActionSpotting


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Integrate ball actions with player tracking"
    )
    parser.add_argument(
        "--tracking-results",
        type=str,
        help="Path to tracking results (TrackLab tracker state)",
    )
    parser.add_argument(
        "--ball-predictions",
        type=str,
        required=True,
        help="Path to ball action predictions (results_spotting.json)",
    )
    parser.add_argument(
        "--jersey-number",
        type=int,
        required=True,
        help="Jersey number to analyze",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="integrated_report.json",
        help="Output file path",
    )
    parser.add_argument(
        "--ball-action-repo",
        type=str,
        default="/Users/eattar/repos/ball-action-spotting",
        help="Path to ball-action-spotting repository",
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=2.0,
        help="Distance threshold in meters (default: 2.0)",
    )
    return parser.parse_args()


def create_jersey_mapping_from_tracking(tracking_results, ball_actions, threshold=2.0):
    """Create jersey mapping from tracking results.
    
    TODO: Implement this based on your tracking system structure.
    
    For now, this is a placeholder that shows the expected structure.
    """
    print("\n⚠️  Using placeholder jersey mapping")
    print("   TODO: Implement actual tracking integration")
    print("   See: sn_gamestate/ball_action/tracking_integration.py\n")
    
    jersey_mapping = {}
    
    # Placeholder logic - replace with actual tracking integration
    for idx, action in enumerate(ball_actions.get("predictions", [])):
        half = int(action["half"])
        position_ms = int(action["position"])
        
        # Mock: assign jersey numbers based on action type
        # In reality, you'd look up the nearest player from tracking_results
        if action["label"] in ["PASS", "DRIVE"]:
            jersey_mapping[(half, position_ms)] = 10  # Assume midfielder
        elif action["label"] in ["SHOT", "GOAL"]:
            jersey_mapping[(half, position_ms)] = 9   # Assume forward
        elif action["label"] in ["TACKLE", "BLOCK"]:
            jersey_mapping[(half, position_ms)] = 4   # Assume defender
        else:
            jersey_mapping[(half, position_ms)] = 7   # Default
    
    return jersey_mapping


def main():
    """Main execution function."""
    args = parse_args()
    
    print("\n" + "="*70)
    print("Ball Action + Player Tracking Integration")
    print("="*70)
    
    # Initialize ball action module
    print("\n📦 Initializing ball action module...")
    try:
        ball_action = BallActionSpotting(
            ball_action_repo_path=args.ball_action_repo
        )
        print("✓ Module initialized")
    except Exception as e:
        print(f"✗ Error: {e}")
        return 1
    
    # Load ball predictions
    print(f"\n📂 Loading ball action predictions...")
    print(f"   {args.ball_predictions}")
    try:
        predictions = ball_action.load_predictions(args.ball_predictions)
        total_actions = len(predictions.get("predictions", []))
        print(f"✓ Loaded {total_actions} ball actions")
    except Exception as e:
        print(f"✗ Error: {e}")
        return 1
    
    # Load tracking results (if provided)
    tracking_results = None
    if args.tracking_results:
        print(f"\n📊 Loading tracking results...")
        print(f"   {args.tracking_results}")
        try:
            # TODO: Implement actual loading
            # import pickle
            # with open(args.tracking_results, 'rb') as f:
            #     tracking_results = pickle.load(f)
            print("⚠️  Tracking integration not yet implemented")
            print("   Using placeholder mapping for demonstration")
        except Exception as e:
            print(f"✗ Error: {e}")
            print("   Continuing with placeholder mapping...")
    
    # Create jersey mapping
    print(f"\n🔗 Creating jersey mapping...")
    print(f"   Distance threshold: {args.distance_threshold}m")
    
    jersey_mapping = create_jersey_mapping_from_tracking(
        tracking_results,
        predictions,
        threshold=args.distance_threshold
    )
    
    print(f"✓ Mapped {len(jersey_mapping)} actions to jersey numbers")
    
    # Show mapping statistics
    player_counts = {}
    for jersey in jersey_mapping.values():
        player_counts[jersey] = player_counts.get(jersey, 0) + 1
    
    print("\n" + "-"*70)
    print("Actions per player:")
    for jersey in sorted(player_counts.keys()):
        count = player_counts[jersey]
        marker = " ← TARGET" if jersey == args.jersey_number else ""
        print(f"  Player #{jersey:2d}: {count:3d} actions{marker}")
    print("-"*70)
    
    # Filter by jersey number
    print(f"\n🎯 Filtering actions for player #{args.jersey_number}...")
    
    try:
        filtered_predictions = ball_action.filter_by_jersey_number(
            predictions,
            jersey_number=args.jersey_number,
            jersey_mapping=jersey_mapping
        )
        filtered_count = len(filtered_predictions.get("predictions", []))
        print(f"✓ Found {filtered_count} actions by player #{args.jersey_number}")
    except Exception as e:
        print(f"✗ Error: {e}")
        return 1
    
    # Generate comprehensive report
    print(f"\n📊 Generating integrated report...")
    
    try:
        report = ball_action.generate_player_report(
            filtered_predictions,
            jersey_number=args.jersey_number,
            output_path=args.output
        )
    except Exception as e:
        print(f"✗ Error: {e}")
        return 1
    
    # Add tracking metadata to report
    report["integration_info"] = {
        "tracking_results_provided": args.tracking_results is not None,
        "distance_threshold_meters": args.distance_threshold,
        "total_mapped_actions": len(jersey_mapping),
        "total_players_involved": len(player_counts),
    }
    
    # Save updated report
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=4)
    
    print("\n" + "="*70)
    print("✅ Integration Complete!")
    print("="*70)
    print(f"\nReport saved to: {args.output}")
    print(f"\nPlayer #{args.jersey_number} Summary:")
    print(f"  Total Actions: {report['total_actions']}")
    if report['total_actions'] > 0:
        print(f"  Action Types: {len(report['action_statistics'])}")
        print(f"  Most Common Action: {max(report['action_statistics'], key=report['action_statistics'].get)}")
    
    print("\n" + "="*70)
    print("Next Steps:")
    print("  1. Implement actual tracking integration in:")
    print("     sn_gamestate/ball_action/tracking_integration.py")
    print("  2. Test with real tracking results from TrackLab")
    print("  3. Tune distance_threshold based on your data")
    print("="*70 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
