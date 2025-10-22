# Ball Action Spotting Module

This module integrates the [ball-action-spotting](https://github.com/recokick/ball-action-spotting) model into the SoccerNet Game State Reconstruction framework. It enables detection and analysis of ball-related actions performed by specific players.

## Overview

The Ball Action Spotting module detects various ball-related actions in soccer videos including:
- **PASS**: Standard passes between players
- **DRIVE**: Ball drives/dribbles
- **HEADER**: Headers
- **HIGH PASS**: Long/high passes
- **OUT**: Ball going out of play
- **CROSS**: Crossing passes
- **THROW IN**: Throw-ins
- **SHOT**: Shot attempts
- **BALL PLAYER BLOCK**: Ball blocks
- **PLAYER SUCCESSFUL TACKLE**: Successful tackles
- **FREE KICK**: Free kicks
- **GOAL**: Goals scored

## Features

- 🎯 **Player-specific action detection**: Filter ball actions by jersey number
- 📊 **Statistical analysis**: Generate comprehensive reports with action counts
- 💾 **JSON output**: Export results in easy-to-parse JSON format
- 🔧 **Modular design**: Easy integration with existing tracking pipelines
- ⚡ **GPU acceleration**: Fast inference using CUDA

## Installation

### Prerequisites

1. **Clone the ball-action-spotting repository**:
   ```bash
   cd /Users/eattar/repos
   git clone https://github.com/recokick/ball-action-spotting.git
   ```

2. **Install ball-action-spotting dependencies**:
   ```bash
   cd ball-action-spotting
   pip install -r requirements.txt
   ```

3. **Download pretrained models**:
   Download the model weights from [Google Drive](https://drive.google.com/drive/folders/1mIu62cIdsRn3W4o1E5vRR8V5Q1B6HHoz?usp=sharing) and place them in the `data/` directory of the ball-action-spotting repository following the structure:
   ```
   data/
   ├── action/
   │   └── experiments/
   │       └── action_sampling_weights_002/
   └── ball_action/
       └── experiments/
           └── sampling_weights_001/
   ```

### Install sn-gamestate with ball_action module

The ball_action module is already included in the sn-gamestate installation.

## Configuration

Configure the module by editing `sn_gamestate/configs/modules/ball_action/default.yaml`:

```yaml
# Path to ball-action-spotting repository (absolute path)
ball_action_repo_path: "/Users/eattar/repos/ball-action-spotting"

# Model experiment name
experiment: "ball_tuning_001"

# GPU device ID
gpu_id: 0

# Cross-validation fold (null for all folds)
fold: null

# Challenge mode
challenge: false

# Output directory
output_dir: "output/ball_action"
```

## Usage

### Method 1: Using the Simple Analysis Script

Analyze ball actions from existing predictions:

```bash
python examples/analyze_player_actions.py \
    --predictions /path/to/results_spotting.json \
    --jersey-number 10 \
    --output player_10_report.json
```

**Arguments:**
- `--predictions`: Path to the ball action prediction JSON file
- `--jersey-number`: Jersey number of the player to analyze
- `--output`: Output file path for the report
- `--ball-action-repo`: Path to ball-action-spotting repository (optional)

### Method 2: Full Pipeline with Prediction

Run the complete pipeline including prediction and analysis:

```bash
python examples/ball_action_example.py \
    --jersey-number 10 \
    --game-path /path/to/game/video \
    --output-dir output/ball_action_results \
    --experiment ball_tuning_001 \
    --gpu-id 0
```

**Arguments:**
- `--jersey-number`: Jersey number of the player to analyze (required)
- `--game-path`: Path to the game video or directory (required)
- `--ball-action-repo`: Path to ball-action-spotting repository
- `--experiment`: Model experiment name (default: ball_tuning_001)
- `--output-dir`: Directory to save results
- `--gpu-id`: GPU device ID (default: 0)
- `--fold`: Cross-validation fold (optional)
- `--challenge`: Run in challenge mode

### Method 3: Using the Python API

```python
from sn_gamestate.ball_action import BallActionSpotting
from pathlib import Path

# Initialize the ball action module
ball_action = BallActionSpotting(
    ball_action_repo_path="/Users/eattar/repos/ball-action-spotting",
    experiment="ball_tuning_001",
    gpu_id=0,
)

# Run predictions on a game
results_path = ball_action.predict_game(
    game_path="/path/to/game",
    output_dir="output/predictions",
    fold=None,
    challenge=False,
)

# Load predictions
predictions = ball_action.load_predictions(
    results_path / "game_name/results_spotting.json"
)

# Create jersey mapping (integrate with your tracking system)
# This maps (half, position) tuples to jersey numbers
jersey_mapping = {
    (1, 45000): 10,  # Player #10 at position 45000ms in half 1
    (1, 67000): 10,  # Player #10 at position 67000ms in half 1
    # ... etc
}

# Filter actions for a specific player
filtered_predictions = ball_action.filter_by_jersey_number(
    predictions,
    jersey_number=10,
    jersey_mapping=jersey_mapping,
)

# Generate a comprehensive report
report = ball_action.generate_player_report(
    filtered_predictions,
    jersey_number=10,
    output_path="player_10_report.json",
)

# Get action statistics
stats = ball_action.get_player_stats(filtered_predictions)
print(f"Player #10 performed {sum(stats.values())} actions")
```

## Output Format

### Player Report JSON

```json
{
  "jersey_number": 10,
  "game": "england_efl/2019-2020/2019-10-01 - Leeds United - West Bromwich",
  "total_actions": 42,
  "action_statistics": {
    "PASS": 18,
    "DRIVE": 12,
    "SHOT": 3,
    "CROSS": 5,
    "HEADER": 2,
    "TACKLE": 2
  },
  "detailed_actions": [
    {
      "gameTime": "1 - 12:34",
      "label": "PASS",
      "position": "754000",
      "half": "1",
      "confidence": "0.95"
    },
    // ... more actions
  ]
}
```

### Filtered Actions JSON

```json
{
  "UrlLocal": "england_efl/2019-2020/2019-10-01 - Leeds United - West Bromwich",
  "predictions": [
    {
      "gameTime": "1 - 12:34",
      "label": "PASS",
      "position": "754000",
      "half": "1",
      "confidence": "0.95"
    },
    // ... more actions
  ]
}
```

## Integration with Tracking System

To get accurate player-action associations, you need to integrate the ball action module with your tracking system. The `jersey_mapping` parameter connects ball actions (identified by time/position) to specific players (identified by jersey numbers).

### Example Integration Flow:

1. **Run player tracking** (using existing sn-gamestate pipeline)
2. **Run ball action detection** (using this module)
3. **Create jersey mapping** by:
   - For each ball action at time `t`:
     - Find the nearest player to the ball location
     - Use that player's jersey number from your tracking results
4. **Filter actions** by jersey number using the mapping

```python
def create_jersey_mapping_from_tracking(tracking_results, ball_actions):
    """Create jersey mapping from tracking results."""
    jersey_mapping = {}
    
    for action in ball_actions["predictions"]:
        half = int(action["half"])
        position = int(action["position"])
        
        # Find the closest player at this timestamp
        closest_player = find_closest_player(
            tracking_results, half, position
        )
        
        if closest_player:
            jersey_mapping[(half, position)] = closest_player.jersey_number
    
    return jersey_mapping
```

## API Reference

### `BallActionSpotting`

Main class for ball action detection and analysis.

#### `__init__(ball_action_repo_path, experiment='ball_tuning_001', gpu_id=0)`

Initialize the ball action spotting module.

#### `predict_game(game_path, output_dir, fold=None, challenge=False)`

Run ball action prediction on a game video.

**Returns:** Path to prediction results directory.

#### `load_predictions(results_json_path)`

Load ball action predictions from JSON file.

**Returns:** Dictionary containing all predictions.

#### `filter_by_jersey_number(predictions, jersey_number, jersey_mapping)`

Filter predictions for a specific player by jersey number.

**Returns:** Filtered predictions dictionary.

#### `get_player_stats(predictions, jersey_number=None)`

Get statistics of ball actions.

**Returns:** Dictionary with action type counts.

#### `generate_player_report(predictions, jersey_number, output_path)`

Generate a comprehensive report for a specific player.

**Returns:** Report dictionary.

#### `save_filtered_results(filtered_predictions, output_path)`

Save filtered predictions to JSON file.

## Troubleshooting

### Issue: "Ball action spotting repository not found"

**Solution:** Make sure the ball-action-spotting repository is cloned and the path in the configuration is correct.

### Issue: "Module not found" errors

**Solution:** Make sure all dependencies for ball-action-spotting are installed:
```bash
cd /Users/eattar/repos/ball-action-spotting
pip install -r requirements.txt
```

### Issue: CUDA out of memory

**Solution:** Reduce batch size or use a different GPU:
```yaml
gpu_id: 1  # Use a different GPU
```

### Issue: Model weights not found

**Solution:** Download the pretrained model weights from the [Google Drive link](https://drive.google.com/drive/folders/1mIu62cIdsRn3W4o1E5vRR8V5Q1B6HHoz?usp=sharing) and place them in the correct directory structure.

## Citation

If you use this module, please cite both the SoccerNet Game State Reconstruction paper and the Ball Action Spotting challenge:

```bibtex
@inproceedings{Somers2024SoccerNetGameState,
    title = {{SoccerNet} Game State Reconstruction: End-to-End Athlete Tracking and Identification on a Minimap},
    author = {Somers, Vladimir and others},
    booktitle = {CVPRW},
    year = {2024},
}
```

## License

This module is part of the sn-gamestate project and follows the same license. The ball-action-spotting repository has its own MIT license.

## Support

For issues related to:
- **This integration module**: Open an issue on the [sn-gamestate repository](https://github.com/SoccerNet/sn-gamestate)
- **Ball action spotting model**: Open an issue on the [ball-action-spotting repository](https://github.com/recokick/ball-action-spotting)

Join the [SoccerNet Discord](https://discord.com/invite/cPbqf2mAwF) for community support.
