# Getting Started with Ball Action Spotting

This guide explains how to use the ball action spotting module to analyze player-specific actions.

## Quick Understanding

The ball action spotting workflow has **two stages**:

### Stage 1: Generate Predictions (Ball Action Spotting Model)
Run the ball-action-spotting model on a game video to detect all ball actions.

### Stage 2: Filter & Analyze (This Module)
Use this module to filter actions by jersey number and generate player reports.

## Option 1: Test with Demo Data (Recommended First)

Test the module with sample data to understand how it works:

```bash
# Run the demo
./demo_ball_action.sh

# Or directly with Python
python examples/demo_ball_action.py --jersey-number 10
```

This will:
- Create sample ball action data
- Filter by jersey number
- Generate JSON reports in `demo_output/`

**Output files:**
- `player_10_report.json` - Comprehensive player stats
- `player_10_actions.json` - Detailed action list
- `all_predictions_sample.json` - All sample predictions

## Option 2: Use Real Game Predictions

### Prerequisites

1. **Ball-action-spotting repository** cloned at `/Users/eattar/repos/ball-action-spotting`
2. **Pretrained models** downloaded (see [ball-action-spotting setup](https://github.com/recokick/ball-action-spotting))
3. **Game video** available

### Step 2.1: Run Ball Action Spotting Model

First, generate predictions using the ball-action-spotting model:

```bash
cd /Users/eattar/repos/ball-action-spotting

# Run prediction on your game
python scripts/ball_action/predict.py \
    --experiment ball_tuning_001 \
    --folds 0 \
    --gpu_id 0
```

This creates `results_spotting.json` in:
```
data/ball_action/predictions/ball_tuning_001/cv/fold_0/[game_name]/results_spotting.json
```

### Step 2.2: Analyze Player Actions

Now use this module to filter by jersey number:

```bash
cd /Users/eattar/repos/sn-gamestate

python examples/analyze_player_actions.py \
    --predictions /path/to/results_spotting.json \
    --jersey-number 10 \
    --output player_10_report.json
```

## Option 3: Full Integrated Pipeline

Run everything in one go:

```bash
python examples/ball_action_example.py \
    --jersey-number 10 \
    --game-path /path/to/game/video \
    --ball-action-repo /Users/eattar/repos/ball-action-spotting \
    --output-dir output/results
```

This will:
1. Run ball action prediction
2. Filter by jersey number
3. Generate comprehensive reports

## Understanding the Output

### Player Report JSON
```json
{
  "jersey_number": 10,
  "game": "england_efl/2019-2020/...",
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
    }
  ]
}
```

## Troubleshooting

### "File not found: results_spotting.json"

**Problem**: The prediction file doesn't exist yet.

**Solution**: 
1. Run the demo first: `./demo_ball_action.sh`
2. Or generate real predictions using ball-action-spotting model (see Step 2.1 above)

### "Ball action spotting repository not found"

**Problem**: The ball-action-spotting repository is not at the expected location.

**Solution**: 
1. Clone the repository:
   ```bash
   cd /Users/eattar/repos
   git clone https://github.com/recokick/ball-action-spotting.git
   ```
2. Or update the path in config: `sn_gamestate/configs/modules/ball_action/default.yaml`

### "Module not found" errors

**Problem**: Dependencies not installed.

**Solution**:
```bash
cd /Users/eattar/repos/sn-gamestate
uv pip install -e .
```

## Current Limitations

### Jersey Number Mapping

The current implementation uses **example jersey mapping** for demonstration. 

**For production use**, you need to integrate with your tracking system:

```python
def create_jersey_mapping_from_tracking(tracking_results, ball_actions):
    """Map ball actions to jersey numbers using tracking data."""
    jersey_mapping = {}
    
    for action in ball_actions["predictions"]:
        half = int(action["half"])
        position = int(action["position"])  # milliseconds
        
        # Find player closest to ball at this time
        closest_player = find_nearest_player(tracking_results, half, position)
        
        if closest_player:
            jersey_mapping[(half, position)] = closest_player.jersey_number
    
    return jersey_mapping
```

## Scripts Reference

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `demo_ball_action.sh` | Test with sample data | First time testing |
| `ball_action_example.py` | Full pipeline | When you have game videos |
| `analyze_player_actions.py` | Analyze existing predictions | When you already have results_spotting.json |
| `quick_start_ball_action.sh` | Quick analysis | Simple analysis of existing predictions |

## Next Steps

1. ✅ **Test the demo**: `./demo_ball_action.sh`
2. 📚 **Read the docs**: `sn_gamestate/ball_action/README.md`
3. 🎮 **Try with real data**: Follow Option 2 or 3 above
4. 🔧 **Integrate tracking**: Connect with your player tracking system

## Support

- 📖 Full documentation: `sn_gamestate/ball_action/README.md`
- 🐛 Issues: https://github.com/eattar/sn-gamestate/issues
- 💬 Discord: https://discord.com/invite/cPbqf2mAwF
