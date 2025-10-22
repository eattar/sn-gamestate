"""Example script demonstrating ball action spotting integration.

This script shows how to:
1. Run ball action predictions on a game
2. Filter actions by jersey number
3. Generate a player-specific report with JSON output

Usage:
    python examples/ball_action_example.py --jersey-number 10 --game-path /path/to/game
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
        description="Run ball action spotting for a specific player"
    )
    parser.add_argument(
        "--jersey-number",
        type=int,
        required=True,
        help="Jersey number of the player to analyze",
    )
    parser.add_argument(
        "--game-path",
        type=str,
        required=True,
        help="Path to the game video or game directory",
    )
    parser.add_argument(
        "--ball-action-repo",
        type=str,
        default="/Users/eattar/repos/ball-action-spotting",
        help="Path to ball-action-spotting repository",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="ball_tuning_001",
        help="Experiment name/model to use",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/ball_action_results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="GPU device ID to use",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Cross-validation fold (optional)",
    )
    parser.add_argument(
        "--challenge",
        action="store_true",
        help="Run in challenge mode",
    )
    return parser.parse_args()


def create_mock_jersey_mapping(predictions_dict, target_jersey):
    """Create a mock jersey mapping for demonstration.
    
    In a real scenario, this mapping would come from the game state
    tracking system which tracks players and their jersey numbers.
    
    For this example, we randomly assign the target jersey number to
    some of the detected actions.
    """
    print("\n" + "="*60)
    print("NOTE: Creating mock jersey mapping for demonstration")
    print("In production, this should come from your tracking system")
    print("="*60 + "\n")
    
    jersey_mapping = {}
    
    # Assign target jersey to every 3rd action (mock example)
    for idx, pred in enumerate(predictions_dict.get("predictions", [])):
        half = int(pred["half"])
        position = int(pred["position"])
        
        # Mock logic: assign target jersey to some actions
        if idx % 3 == 0:
            jersey_mapping[(half, position)] = target_jersey
        else:
            # Assign random other jersey numbers
            jersey_mapping[(half, position)] = (idx % 20) + 1
    
    return jersey_mapping


def main():
    """Main execution function."""
    args = parse_args()
    
    print("="*60)
    print("Ball Action Spotting - Player Analysis")
    print("="*60)
    print(f"Player Jersey Number: {args.jersey_number}")
    print(f"Game Path: {args.game_path}")
    print(f"Output Directory: {args.output_dir}")
    print("="*60 + "\n")
    
    # Initialize ball action spotting
    try:
        ball_action = BallActionSpotting(
            ball_action_repo_path=args.ball_action_repo,
            experiment=args.experiment,
            gpu_id=args.gpu_id,
        )
        print("✓ Ball action spotting initialized successfully\n")
    except Exception as e:
        print(f"✗ Error initializing ball action spotting: {e}")
        return 1
    
    # Run prediction
    print("Running ball action prediction...")
    print("This may take several minutes depending on video length...\n")
    
    try:
        results_path = ball_action.predict_game(
            game_path=args.game_path,
            output_dir=args.output_dir,
            fold=args.fold,
            challenge=args.challenge,
        )
        print(f"✓ Predictions completed")
        print(f"  Results directory: {results_path}\n")
    except Exception as e:
        print(f"✗ Error during prediction: {e}")
        return 1
    
    # Load predictions
    print("Loading prediction results...")
    try:
        # Find results_spotting.json in the output directory
        results_json = None
        for json_file in results_path.rglob("results_spotting.json"):
            results_json = json_file
            break
        
        if results_json is None:
            print("✗ Could not find results_spotting.json")
            return 1
        
        predictions = ball_action.load_predictions(results_json)
        total_actions = len(predictions.get("predictions", []))
        print(f"✓ Loaded {total_actions} ball actions\n")
    except Exception as e:
        print(f"✗ Error loading predictions: {e}")
        return 1
    
    # Create jersey mapping
    # NOTE: In production, this should come from your tracking system
    jersey_mapping = create_mock_jersey_mapping(predictions, args.jersey_number)
    
    # Filter by jersey number
    print(f"Filtering actions for player #{args.jersey_number}...")
    try:
        filtered_predictions = ball_action.filter_by_jersey_number(
            predictions,
            jersey_number=args.jersey_number,
            jersey_mapping=jersey_mapping,
        )
        filtered_count = len(filtered_predictions.get("predictions", []))
        print(f"✓ Found {filtered_count} actions by player #{args.jersey_number}\n")
    except Exception as e:
        print(f"✗ Error filtering predictions: {e}")
        return 1
    
    # Generate player report
    print("Generating player report...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        report = ball_action.generate_player_report(
            filtered_predictions,
            jersey_number=args.jersey_number,
            output_path=output_dir / f"player_{args.jersey_number}_report.json",
        )
        print("\n✓ Report generated successfully!")
    except Exception as e:
        print(f"✗ Error generating report: {e}")
        return 1
    
    # Save filtered predictions as well
    try:
        ball_action.save_filtered_results(
            filtered_predictions,
            output_path=output_dir / f"player_{args.jersey_number}_actions.json",
        )
    except Exception as e:
        print(f"✗ Error saving filtered results: {e}")
        return 1
    
    print("\n" + "="*60)
    print("Ball Action Analysis Complete!")
    print("="*60)
    print(f"Results saved to: {output_dir}")
    print(f"  - player_{args.jersey_number}_report.json")
    print(f"  - player_{args.jersey_number}_actions.json")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
