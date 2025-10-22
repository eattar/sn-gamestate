# Ball Action Spotting TrackLab Module

This directory contains TrackLab modules for integrating ball action spotting into the sn-gamestate pipeline.

## Overview

The ball action integration consists of two TrackLab modules:

1. **BallActionSpottingModule**: Detects ball actions (PASS, DRIVE, etc.) in the video
2. **ActionPlayerMatchingModule**: Matches detected actions to player jersey numbers using tracking data

## Setup

### 1. Install ball-action-spotting

```bash
cd /workspace
git clone https://github.com/recokick/ball-action-spotting.git
cd ball-action-spotting

# Create conda environment
conda create -n ball-action-spotting python=3.10 -y
conda activate ball-action-spotting

# Install dependencies
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install numpy scipy opencv-python albumentations segmentation-models-pytorch timm SoccerNet pytorch-argus kornia av joblib

# Configure paths (update these for your system)
# Edit src/constants.py - set data_dir
# Edit src/ball_action/constants.py - set soccernet_dir

# Download model weights
# Follow instructions in ball-action-spotting README
```

### 2. Update sn-gamestate Configuration

Edit `sn_gamestate/configs/soccernet.yaml`:

```yaml
# Add ball action modules to defaults
defaults:
  # ... existing modules ...
  - modules/ball_action: ball_action_spotting
  - modules/ball_action: action_player_matching

# Add to pipeline (after tracking and jersey detection)
pipeline:
  - bbox_detector
  - reid
  - track
  - pitch
  - calibration
  - jersey_number_detect
  - tracklet_agg
  - team
  - team_side
  - ball_action_spotting      # NEW: Detect ball actions
  - action_player_matching    # NEW: Match to jersey numbers

# Configure ball action module
modules:
  ball_action_spotting:
    ball_action_repo_path: "/workspace/ball-action-spotting"
    experiment_name: "sampling_weights_001"
    fold: 0
    use_cached: true
```

### 3. Run the Pipeline

```bash
cd /workspace/sn-gamestate
conda activate sn-gamestate

# Run with ball action detection
tracklab -cn soccernet
```

## Module Details

### BallActionSpottingModule

**What it does:**
- Runs ball-action-spotting model on the entire video
- Detects PASS and DRIVE actions with timestamps and confidence
- Caches results to avoid re-running expensive prediction

**Input:** Video file
**Output:** List of ball actions with:
- `gameTime`: Time in game (e.g., "1 - 00:15")
- `label`: Action type (PASS, DRIVE)
- `position`: Frame number
- `half`: Game half (1 or 2)
- `confidence`: Prediction confidence (0-1)

**Configuration:**
```yaml
ball_action_repo_path: "/workspace/ball-action-spotting"
experiment_name: "sampling_weights_001"
fold: 0
use_cached: true
```

### ActionPlayerMatchingModule

**What it does:**
- Takes ball actions and player tracking data
- Assigns each action to the nearest player at that frame
- Adds jersey numbers to actions

**Input:** 
- Ball actions from BallActionSpottingModule
- Player bounding boxes from tracking
- Jersey numbers from jersey detection
- Track IDs from tracking

**Output:** Enhanced ball actions with:
- `jersey`: Player's jersey number
- `track_id`: Player's track ID
- `match_status`: "matched", "no_players", or "no_nearby_player"

**Configuration:**
```yaml
max_distance: 200.0  # Max pixels to assign action to player
use_ball_position: false  # Use ball tracking if available
```

## Output Format

After running the full pipeline, you'll have ball actions matched to jersey numbers:

```json
{
  "ball_actions_with_jerseys": [
    {
      "gameTime": "1 - 00:15",
      "label": "PASS",
      "position": "1520",
      "half": "1",
      "confidence": "0.92",
      "jersey": 10,
      "track_id": 5,
      "match_status": "matched"
    }
  ]
}
```

## Analysis

After running the pipeline, analyze results:

```python
from sn_gamestate.ball_action import BallActionSpotting

api = BallActionSpotting()

# Load matched results from tracker state
# (TrackLab saves state.ball_actions_with_jerseys)

# Filter by player
player_10_actions = [a for a in actions if a['jersey'] == 10]

# Generate stats
stats = api.get_player_stats(player_10_actions)
print(f"Player #10: {stats['total_actions']} actions")
```

## Advantages of TrackLab Integration

✅ **Single Pipeline**: Run everything in one command
✅ **Automatic Caching**: Predictions cached, no re-computation
✅ **State Management**: All results in one tracker state object
✅ **Visualization**: Integrate with TrackLab's visualization tools
✅ **Modularity**: Easy to swap components or add new modules

## Limitations

⚠️ **Current Status**: The modules are implemented but need TrackLab state integration
⚠️ **TODO**: Complete `_get_players_at_frame()` and `_find_nearest_player()` methods
⚠️ **Manual Matching**: For now, use `examples/match_actions_to_players.py` standalone

## Next Steps

1. Complete the TrackLab module implementation
2. Test with full sn-gamestate pipeline
3. Add visualization of actions on minimap
4. Support more action types when available

## Questions?

See:
- `sn_gamestate/ball_action/README.md` - API documentation
- `TRACKING_INTEGRATION.md` - Manual integration guide
- `examples/match_actions_to_players.py` - Standalone matching script
