# Tracking Integration Guide

This guide explains how to connect ball action predictions with your player tracking system to assign jersey numbers to each action.

## Overview

The ball-action-spotting model outputs:
- ✅ Action type (PASS, DRIVE, etc.)
- ✅ Timestamp/frame number
- ✅ Confidence score
- ❌ **No player identification**

To get player-specific actions (the main goal), you need to:
1. Run your tracking system on the same video
2. Match each action to the nearest player at that frame
3. Assign jersey numbers from tracking data

## Step-by-Step Integration

### Step 1: Prepare Your Tracking Data

Make sure your tracking system outputs JSON with player positions and jersey numbers. Common formats:

**Option A: Frame-based format**
```json
{
  "frames": [
    {
      "frame_id": 1520,
      "players": [
        {
          "id": 1,
          "jersey": 10,
          "team": "home",
          "bbox": [100, 200, 50, 80]
        }
      ]
    }
  ]
}
```

**Option B: Track-based format**
```json
{
  "tracks": [
    {
      "track_id": 1,
      "jersey": 10,
      "team": "home",
      "detections": [
        {"frame": 1520, "bbox": [100, 200, 50, 80]},
        {"frame": 1521, "bbox": [102, 201, 50, 80]}
      ]
    }
  ]
}
```

### Step 2: Adapt the Template Script

Edit `match_actions_to_players.py` to work with your tracking format:

1. **Update `get_players_at_frame()`** - Extract players at specific frame
2. **Update `get_ball_position_at_frame()`** - Get ball position (if available)
3. **Adjust `get_bbox_center()`** - Match your bbox format ([x,y,w,h] vs [x1,y1,x2,y2])

### Step 3: Run the Matching

```bash
python examples/match_actions_to_players.py \
    --actions /path/to/results_spotting.json \
    --tracking /path/to/tracking_results.json \
    --output /path/to/actions_with_jerseys.json
```

Optional parameters:
- `--max-distance 200` - Maximum pixels between player and action
- `--jersey 10` - Generate report for specific player

### Step 4: Analyze Player Actions

Once you have `actions_with_jerseys.json`, you can filter by jersey:

```bash
python examples/analyze_player_actions.py \
    --results /path/to/actions_with_jerseys.json \
    --jersey 10
```

Or use the API programmatically:

```python
from sn_gamestate.ball_action import BallActionSpotting

api = BallActionSpotting()

# Load enhanced results
results = api.load_predictions("actions_with_jerseys.json")

# Get actions for player #10
player_actions = api.filter_by_jersey_number(results, jersey_number=10)

# Generate stats
stats = api.get_player_stats(player_actions)
print(f"Player #10 made {stats['total_actions']} actions")
print(f"  PASS: {stats['actions_by_type'].get('PASS', 0)}")
print(f"  DRIVE: {stats['actions_by_type'].get('DRIVE', 0)}")
```

## Matching Strategies

### Strategy 1: Ball Position (Recommended)

If your tracking includes ball detection:
- Use ball position at each action frame
- Find nearest player to the ball
- Most accurate for determining who performed the action

```python
ball_position = get_ball_position_at_frame(tracking_data, frame_number)
nearest = find_nearest_player(ball_position, players)
```

### Strategy 2: Player Bounding Boxes Only

If you don't have ball tracking:
- Assume the action happens at the center of the frame
- Find players in the action region (center third of frame)
- Choose the most prominent player (largest bbox)

```python
# Filter to central region
central_players = [p for p in players if is_in_center_region(p['bbox'])]
# Pick largest
nearest = max(central_players, key=lambda p: bbox_area(p['bbox']))
```

### Strategy 3: Temporal Smoothing

For better accuracy:
- Look at player positions in a window (±5 frames)
- Use the player who's consistently closest to the ball
- Helps with occlusions and tracking errors

## Troubleshooting

### Low Matching Rate

If many actions are unmatched:
- **Increase `max_distance`** - Try 300-400 pixels
- **Check bbox format** - Ensure coordinates are correct
- **Verify frame alignment** - Tracking and video must be synchronized

### Wrong Player Assignments

If actions assigned to wrong players:
- **Add ball tracking** - Much more accurate than bbox proximity
- **Use team filtering** - Only consider players from the team in possession
- **Temporal consistency** - A player can't perform actions 50m apart in 1 second

### Missing Jersey Numbers

If tracking doesn't have jerseys:
- Run jersey number detection first (see `sn_gamestate/jersey/`)
- Assign jerseys to tracks before matching
- Or match to track IDs first, then add jerseys later

## Example Workflow

Complete pipeline from video to player stats:

```bash
# 1. Run ball action predictions (already done)
cd /workspace/ball-action-spotting
python scripts/ball_action/predict.py --experiment sampling_weights_001 --folds 0

# 2. Run your tracking system
cd /workspace/sn-gamestate
python run_tracking.py --video /path/to/game.mp4 --output tracking_results.json

# 3. Match actions to players
python examples/match_actions_to_players.py \
    --actions /workspace/ball-action-spotting/data/.../results_spotting.json \
    --tracking tracking_results.json \
    --output actions_with_jerseys.json

# 4. Analyze specific player
python examples/analyze_player_actions.py \
    --results actions_with_jerseys.json \
    --jersey 10
```

## Next Steps

1. ✅ **Adapt the template** - Update functions for your tracking format
2. ✅ **Test on sample data** - Verify matching accuracy
3. ✅ **Integrate with sn-gamestate** - Use in your full pipeline
4. ✅ **Build UI/reporting** - Present results to end users

## Need Help?

Check these files for reference:
- `sn_gamestate/ball_action/README.md` - API documentation
- `examples/integrate_with_tracking.py` - Integration example
- `PRODUCTION_ROADMAP.md` - Full deployment guide
