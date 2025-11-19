# Ball Action Spotting Integration

This integration combines **SN-GameState** player tracking with **ball-action-spotting** to extract ball actions performed by a specific player identified by team and jersey number.

## Features

- 🎯 **Player-Specific Actions**: Filter actions by team (left/right) and jersey number
- ⚽ **Ball Action Detection**: Detects PASS and DRIVE actions (2 out of 12 classes)
- ⏱️ **Timestamped Output**: Actions with MM:SS timestamps
- 💾 **State Caching**: Save tracking state to avoid re-processing
- 🚀 **Optimized Pipeline**: Skips unnecessary modules (visualization, calibration)

## Requirements

### 1. SN-GameState Installation
```bash
# Already installed in this repository
cd /Users/eattar/dfki_project/sn-gamestate
```

### 2. Ball-Action-Spotting Setup
```bash
cd /Users/eattar/dfki_project/ball-action-spotting

# Download model weights from Google Drive
# Place in: data/ball_action/experiments/ball_finetune_long_004/fold_5/
```

Required model structure:
```
ball-action-spotting/
└── data/
    └── ball_action/
        └── experiments/
            └── ball_finetune_long_004/
                └── fold_5/
                    └── model-006-0.901643.pth  (90.1% accuracy)
```

## Usage

### Basic Usage

```bash
python run_player_ball_actions.py \
  --video match.mp4 \
  --team left \
  --jersey 10
```

### With Output File

```bash
python run_player_ball_actions.py \
  --video match.mp4 \
  --team right \
  --jersey 7 \
  --output player7_actions.json
```

### Using Cached Tracking State

```bash
# First run - saves tracking state
python run_player_ball_actions.py \
  --video match.mp4 \
  --team left \
  --jersey 10

# Second run - uses cached state (much faster)
python run_player_ball_actions.py \
  --video match.mp4 \
  --team left \
  --jersey 10 \
  --state-cache tracking_state_temp.pklz \
  --skip-tracking
```

### Custom Model

```bash
python run_player_ball_actions.py \
  --video match.mp4 \
  --team left \
  --jersey 10 \
  --experiment ball_finetune_long_004 \
  --fold 5
```

## Command Line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--video` | ✅ | - | Path to input video file |
| `--team` | ✅ | - | Player team: `left` or `right` |
| `--jersey` | ✅ | - | Player jersey number (integer) |
| `--output` | ❌ | `player_<jersey>_<team>_actions.json` | Output JSON file path |
| `--state-cache` | ❌ | - | Path to cached .pklz tracking state |
| `--skip-tracking` | ❌ | False | Skip tracking (requires --state-cache) |
| `--experiment` | ❌ | `ball_finetune_long_004` | Ball-action experiment name |
| `--fold` | ❌ | `5` | Model fold (5 has best accuracy: 90.1%) |
| `--device` | ❌ | `cuda:0` | Device for inference |

## Output Format

The script generates a JSON file with the following structure:

```json
{
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
    }
  ]
}
```

### Field Descriptions

- **player**: Target player identification
  - `team`: "left" or "right"
  - `jersey`: Jersey number
- **video**: Input video path
- **total_actions**: Total number of matched actions
- **action_counts**: Breakdown by action type
- **actions**: List of detected actions
  - `action`: Action type (PASS or DRIVE)
  - `frame`: Frame where action was detected
  - `confidence`: Model confidence (0-1)
  - `player_frame`: Closest frame where player was detected
  - `frame_offset`: Frame difference between action and player
  - `time`: Timestamp in MM:SS format

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Input: Video File                                           │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────┴──────────────┐
        │                            │
        ▼                            ▼
┌───────────────────┐      ┌──────────────────────┐
│ SN-GameState      │      │ Ball-Action-Spotting │
│ Player Tracking   │      │ Action Detection     │
├───────────────────┤      ├──────────────────────┤
│ • YOLO Detection  │      │ • EfficientNetV2     │
│ • PRTReId         │      │ • 3D CNN Temporal    │
│ • StrongSort      │      │ • TTA Augmentation   │
│ • Jersey OCR      │      └──────────┬───────────┘
│ • Team Clustering │                 │
└────────┬──────────┘                 │
         │                            │
         │ TrackerState (.pklz)       │ Actions (frames)
         │ • track_id                 │ • PASS
         │ • jn_tracklet (jersey)     │ • DRIVE
         │ • team (left/right)        │ • confidence
         │ • bbox_ltwh                │
         │ • image_id (frame)         │
         │                            │
         └─────────────┬──────────────┘
                       │
                       ▼
         ┌─────────────────────────┐
         │ Action-Player Matching  │
         ├─────────────────────────┤
         │ 1. Filter by team+jersey│
         │ 2. Temporal matching    │
         │    (±50 frame window)   │
         │ 3. Closest frame select │
         └────────────┬────────────┘
                      │
                      ▼
         ┌─────────────────────────┐
         │ JSON Output             │
         │ • Player info           │
         │ • Timestamped actions   │
         │ • Statistics            │
         └─────────────────────────┘
```

## How It Works

### Step 1: Player Tracking (SN-GameState)

Runs a simplified pipeline:
1. **bbox_detector** (YOLOv11): Detect all players
2. **reid** (PRTReId): Extract appearance features
3. **track** (StrongSort): Track players across frames
4. **jersey_number_detect** (MMOCR): OCR jersey numbers
5. **tracklet_agg**: Vote for most common jersey per track
6. **team**: Cluster players into 2 teams
7. **team_side**: Assign left/right based on position

**Output**: TrackerState with columns:
- `track_id`: Unique player ID
- `jn_tracklet`: Voted jersey number
- `team`: "left" or "right"
- `bbox_ltwh`: Bounding box
- `image_id`: Frame number

### Step 2: Ball Action Detection

1. Load best model from experiment directory
2. Process video with frame stacking (33 frames, step=2)
3. Run EfficientNetV2-B0 + 3D CNN
4. Post-process predictions (gaussian filter, peak detection)

**Output**: List of actions with frame number and confidence

### Step 3: Action-Player Matching

1. Filter tracking results by `team` and `jn_tracklet`
2. For each detected action:
   - Find player detections within ±50 frame window
   - Select closest detection by frame number
   - Record frame offset

### Step 4: JSON Export

Format results with timestamps (frame / fps → MM:SS)

## Current Limitations

### 1. Action Classes
⚠️ **Only 2 out of 12 action types detected**:
- ✅ PASS
- ✅ DRIVE
- ❌ HEADER, HIGH PASS, OUT, CROSS, THROW IN, SHOT, BALL PLAYER BLOCK, PLAYER SUCCESSFUL TACKLE, FREE KICK, GOAL

**Reason**: Available models (`ball_finetune_long_004`) are binary classifiers trained on PASS and DRIVE only.

**Solution**: Train a 12-class model or use ensemble of binary classifiers.

### 2. Video Input Format
Currently requires video in SoccerNetGS dataset structure:
```
SoccerNetGS/
└── [game_name]/
    └── 720p.mkv
```

**Future improvement**: Support arbitrary video files via TrackLab's `ExternalVideo` wrapper.

### 3. Jersey Number Accuracy
Jersey detection depends on:
- OCR quality (MMOCR)
- Player visibility
- Voting across multiple frames

**If no player found**: Check available combinations with error message.

### 4. Spatial Matching
Currently uses **temporal proximity only** (±50 frames).

**Better approach**: Add spatial matching using:
- Bounding box IoU overlap
- Pitch coordinates (`bbox_pitch` from calibration)

## Performance

### Processing Time (720p video, ~90 min match)

| Step | Time | Notes |
|------|------|-------|
| Player Tracking | ~15-20 min | Can be cached |
| Ball Action Detection | ~15-20 min | Depends on GPU |
| Matching | ~1-2 sec | Fast |
| **Total (first run)** | **~30-40 min** | - |
| **Total (cached)** | **~15-20 min** | Skip tracking |

### Hardware Requirements

- **GPU**: NVIDIA with 8GB+ VRAM (tested on RTX 3080)
- **CUDA**: 11.8+
- **RAM**: 16GB+ recommended
- **Storage**: ~100GB for models and datasets

## Troubleshooting

### Error: "No player found with team='left' and jersey=10"

**Cause**: No player tracked with that team+jersey combination.

**Solution**:
1. Check available players in error output
2. Try different jersey number
3. Verify video has players visible

### Error: "No model found in experiments directory"

**Cause**: Ball-action model not downloaded.

**Solution**:
```bash
# Download from Google Drive (see README)
# Place in: ../ball-action-spotting/data/ball_action/experiments/ball_finetune_long_004/fold_5/
```

### Error: "Video file not found"

**Cause**: Incorrect video path or format issue.

**Solution**:
```bash
# Check video exists
ls -lh match.mp4

# Verify format (should be .mp4 or .mkv)
ffprobe match.mp4
```

### Low Action Count

**Possible causes**:
1. Player not visible during actions (off-screen, occluded)
2. Window too narrow (try increasing with code modification)
3. Jersey detection failed for some frames

**Debug**:
- Check `frame_offset` in output (should be < 50)
- Review tracking state: look at detections count for player

## Future Improvements

### Short Term
- [ ] Support arbitrary video input (no dataset structure required)
- [ ] Add spatial matching using bbox IoU
- [ ] Make matching window configurable via CLI
- [ ] Add verbose mode with detailed logging

### Medium Term
- [ ] Support all 12 action classes (train new model)
- [ ] Add video visualization with overlays
- [ ] Batch processing for multiple players
- [ ] Team-wide statistics generation

### Long Term
- [ ] Real-time processing capability
- [ ] Web UI for easy usage
- [ ] Database integration for match analysis
- [ ] Heatmap generation for player actions

## Citation

If you use this integration, please cite both projects:

```bibtex
@inproceedings{soccernet2023,
  title={SoccerNet Game State Reconstruction: End-to-End Multi-Object Tracking},
  author={...},
  booktitle={...},
  year={2023}
}

@inproceedings{ballaction2023,
  title={SoccerNet Ball Action Spotting},
  author={Baikulov, Ruslan and ...},
  year={2023}
}
```

## License

This integration inherits licenses from both projects:
- SN-GameState: MIT License
- Ball-Action-Spotting: Apache 2.0 License

## Contact

For issues specific to this integration, please open an issue in the sn-gamestate repository on the `ball-action-integration` branch.
