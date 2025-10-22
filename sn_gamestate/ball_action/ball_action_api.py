"""API wrapper for ball-action-spotting integration.

This module provides an interface to the ball-action-spotting model,
allowing users to detect ball actions and filter results by jersey number.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union
import subprocess


class BallActionSpotting:
    """Wrapper for ball-action-spotting model to detect player-specific ball actions.
    
    This class provides an interface to run ball-action-spotting predictions
    and filter results by jersey number to show actions made by specific players.
    
    Attributes:
        ball_action_repo_path: Path to the ball-action-spotting repository
        model_config: Configuration for the ball-action-spotting model
    """
    
    def __init__(
        self,
        ball_action_repo_path: Union[str, Path],
        experiment: str = "ball_tuning_001",
        gpu_id: int = 0,
    ):
        """Initialize BallActionSpotting wrapper.
        
        Args:
            ball_action_repo_path: Path to the ball-action-spotting repository
            experiment: Name of the experiment/model to use
            gpu_id: GPU device ID to use for inference
        """
        self.ball_action_repo_path = Path(ball_action_repo_path)
        self.experiment = experiment
        self.gpu_id = gpu_id
        
        # Validate repository path
        if not self.ball_action_repo_path.exists():
            raise ValueError(
                f"Ball action spotting repository not found at: {self.ball_action_repo_path}"
            )
        
        # Add ball-action-spotting to Python path
        if str(self.ball_action_repo_path) not in sys.path:
            sys.path.insert(0, str(self.ball_action_repo_path))
    
    def predict_game(
        self,
        game_path: Union[str, Path],
        output_dir: Union[str, Path],
        fold: Optional[Union[int, str]] = None,
        challenge: bool = False,
    ) -> Path:
        """Run ball action prediction on a game video.
        
        Args:
            game_path: Path to the game video or game directory
            output_dir: Directory to save prediction results
            fold: Fold number for cross-validation (optional)
            challenge: Whether to run in challenge mode
            
        Returns:
            Path to the results_spotting.json file
        """
        predict_script = self.ball_action_repo_path / "scripts" / "ball_action" / "predict.py"
        
        if not predict_script.exists():
            raise FileNotFoundError(
                f"Prediction script not found at: {predict_script}"
            )
        
        # Build command
        cmd = [
            sys.executable,
            str(predict_script),
            "--experiment", self.experiment,
            "--gpu_id", str(self.gpu_id),
        ]
        
        if fold is not None:
            cmd.extend(["--folds", str(fold)])
        
        if challenge:
            cmd.append("--challenge")
        
        # Run prediction
        result = subprocess.run(
            cmd,
            cwd=str(self.ball_action_repo_path),
            capture_output=True,
            text=True,
        )
        
        if result.returncode != 0:
            raise RuntimeError(
                f"Ball action prediction failed with error:\n{result.stderr}"
            )
        
        # Determine output path
        data_split = "challenge" if challenge else "cv"
        fold_dir = f"fold_{fold}" if fold is not None else "fold_0"
        
        results_path = (
            self.ball_action_repo_path
            / "data"
            / "ball_action"
            / "predictions"
            / self.experiment
            / data_split
            / fold_dir
        )
        
        return results_path
    
    def load_predictions(self, results_json_path: Union[str, Path]) -> Dict:
        """Load ball action predictions from JSON file.
        
        Args:
            results_json_path: Path to results_spotting.json file
            
        Returns:
            Dictionary containing all predictions
        """
        results_json_path = Path(results_json_path)
        
        if not results_json_path.exists():
            raise FileNotFoundError(
                f"Results file not found at: {results_json_path}"
            )
        
        with open(results_json_path, "r") as f:
            results = json.load(f)
        
        return results
    
    def filter_by_jersey_number(
        self,
        predictions: Dict,
        jersey_number: int,
        jersey_mapping: Optional[Dict[str, int]] = None,
    ) -> Dict:
        """Filter ball action predictions for a specific player by jersey number.
        
        Note: This function requires additional tracking/jersey information to map
        ball actions to specific players. The jersey_mapping should provide the
        jersey number for each detection at each frame.
        
        Args:
            predictions: Dictionary containing all ball action predictions
            jersey_number: Jersey number to filter by
            jersey_mapping: Mapping from (half, frame, position) to jersey numbers
                            This needs to be provided from the game state tracking
            
        Returns:
            Filtered dictionary containing only actions by the specified player
        """
        if jersey_mapping is None:
            raise ValueError(
                "jersey_mapping is required to filter by jersey number. "
                "This should be derived from the game state tracking results."
            )
        
        filtered_predictions = {
            "UrlLocal": predictions.get("UrlLocal", ""),
            "predictions": [],
        }
        
        # Filter predictions based on jersey mapping
        for pred in predictions.get("predictions", []):
            half = int(pred["half"])
            position = int(pred["position"])
            
            # Check if this action corresponds to the target jersey number
            key = (half, position)
            if key in jersey_mapping and jersey_mapping[key] == jersey_number:
                filtered_predictions["predictions"].append(pred)
        
        return filtered_predictions
    
    def get_player_stats(
        self,
        predictions: Dict,
        jersey_number: Optional[int] = None,
    ) -> Dict[str, int]:
        """Get statistics of ball actions.
        
        Args:
            predictions: Dictionary containing ball action predictions
            jersey_number: If provided, filter stats for this jersey number
            
        Returns:
            Dictionary with action type counts
        """
        stats = {}
        
        for pred in predictions.get("predictions", []):
            action_label = pred["label"]
            stats[action_label] = stats.get(action_label, 0) + 1
        
        return stats
    
    def save_filtered_results(
        self,
        filtered_predictions: Dict,
        output_path: Union[str, Path],
    ) -> None:
        """Save filtered predictions to JSON file.
        
        Args:
            filtered_predictions: Dictionary containing filtered predictions
            output_path: Path where to save the JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(filtered_predictions, f, indent=4)
        
        print(f"Filtered results saved to: {output_path}")
    
    def generate_player_report(
        self,
        predictions: Dict,
        jersey_number: int,
        output_path: Union[str, Path],
    ) -> Dict:
        """Generate a comprehensive report of ball actions for a specific player.
        
        Args:
            predictions: Dictionary containing filtered predictions for the player
            jersey_number: Jersey number of the player
            output_path: Path where to save the report JSON file
            
        Returns:
            Dictionary containing the complete report
        """
        stats = self.get_player_stats(predictions, jersey_number)
        
        report = {
            "jersey_number": jersey_number,
            "game": predictions.get("UrlLocal", ""),
            "total_actions": len(predictions.get("predictions", [])),
            "action_statistics": stats,
            "detailed_actions": predictions.get("predictions", []),
        }
        
        # Save report
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(report, f, indent=4)
        
        print(f"Player report saved to: {output_path}")
        print(f"\nPlayer #{jersey_number} Ball Action Summary:")
        print(f"  Total Actions: {report['total_actions']}")
        print(f"  Action Breakdown:")
        for action, count in stats.items():
            print(f"    {action}: {count}")
        
        return report
