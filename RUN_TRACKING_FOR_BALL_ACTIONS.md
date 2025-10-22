# Running Tracking for Ball Action Integration

This guide explains how to run the sn-gamestate tracking pipeline on the same videos where you've detected ball actions, so you can match actions to players by jersey number.

## Goal
Generate player tracking results (with jersey numbers) for the Leeds United - West Bromwich game to match with the ball action predictions.

## Prerequisites

✅ Ball actions already detected (2,198 PASS + 1,855 DRIVE actions)
✅ Video file at: `/netscratch/eattar/ds/SoccetNet/spotting-ball-2024/england_efl/2019-2020/2019-10-01 - Leeds United - West Bromwich/720p.mp4`

⏳ Need: Player tracking with jersey numbers for the same video

## Step 1: Check if SoccerNet-GS Dataset Format Needed

The sn-gamestate pipeline expects SoccerNet Game State dataset format. Your video is from the ball-action-spotting dataset, which has a different structure.

**Two options:**

### Option A: Adapt sn-gamestate to work with your video format
Create a custom dataset config that points to your video directory structure.

### Option B: Run tracking directly using TrackLab components
Use the individual TrackLab modules to process your video without requiring the full SoccerNet-GS dataset format.

## Step 2: Install sn-gamestate on VM (if not already done)

```bash
cd /workspace
git clone https://github.com/eattar/sn-gamestate.git
cd sn-gamestate
git checkout feature/ball-action-spotting

# Install using conda (recommended for VM)
conda create -n sn-gamestate pip python=3.9 pytorch==1.13.1 torchvision==0.14.1 pytorch-cuda=11.7 -c pytorch -c nvidia -y
conda activate sn-gamestate

# Install dependencies
pip install -e .
mim install mmcv==2.0.1
```

## Step 3: Create Custom Config for Your Video

Create a new config file that points to your ball-action-spotting video:

```yaml
# sn_gamestate/configs/dataset/ball_action_video.yaml
defaults:
  - soccernet_gs

# Override the dataset path to point to your videos
dataset_path: "/netscratch/eattar/ds/SoccetNet/spotting-ball-2024"

# Specify the specific video to process
eval_set: "england_efl"
vids_dict:
  england_efl:
    - "2019-2020/2019-10-01 - Leeds United - West Bromwich"

# Video file naming (720p.mp4 instead of default)
video_file: "720p.mp4"
```

## Step 4: Modify soccernet.yaml for Your Use Case

```yaml
# Update these paths in soccernet.yaml
data_dir: "/netscratch/eattar/ds/SoccetNet/spotting-ball-2024"

# Use only the modules needed for tracking + jersey detection
pipeline:
  - bbox_detector  # Detect players
  - reid          # Re-identification
  - track         # Track players across frames
  - jersey_number_detect  # Recognize jersey numbers

# Don't need these for ball action matching:
# - pitch         # (optional - only if you want pitch coordinates)
# - calibration   # (optional - only if you want pitch coordinates)
# - team          # (optional - team clustering)
# - team_side     # (optional - left/right team)
# - tracklet_agg  # (optional - role voting)

# Save the tracker state
state:
  save_file: "states/ball_action_tracking.pklz"
  load_file: null

# Process just this one video
dataset:
  nvid: 1
  eval_set: "england_efl"
```

## Step 5: Run Tracking

```bash
cd /workspace/sn-gamestate
conda activate sn-gamestate

# Run tracking on your video
tracklab -cn soccernet dataset=ball_action_video
```

This will:
1. Detect all players in each frame
2. Track them across frames (assign track IDs)
3. Extract jersey numbers for each player
4. Save results to `states/ball_action_tracking.pklz`

## Step 6: Export Tracking Results to JSON

The tracker state is in pickle format. You need to export it to JSON for the matching script:

```python
# export_tracking_to_json.py
import pickle
import json
from pathlib import Path

# Load tracker state
with open("states/ball_action_tracking.pklz", "rb") as f:
    tracker_state = pickle.load(f)

# Extract tracking data
tracking_data = {
    "tracks": []
}

for track_id, track in tracker_state.tracks.items():
    track_info = {
        "track_id": track_id,
        "jersey": track.get("jersey_number"),
        "team": track.get("team"),
        "detections": []
    }
    
    for frame_num, detection in track.detections.items():
        track_info["detections"].append({
            "frame": frame_num,
            "bbox": detection.bbox.tolist(),  # [x, y, w, h]
            "confidence": detection.conf
        })
    
    tracking_data["tracks"].append(track_info)

# Save to JSON
with open("tracking_results.json", "w") as f:
    json.dump(tracking_data, f, indent=2)

print(f"Exported {len(tracking_data['tracks'])} tracks")
```

## Step 7: Match Ball Actions with Tracking

Now you can use the matching script:

```bash
python examples/match_actions_to_players.py \
    --actions /workspace/ball-action-spotting/data/.../results_spotting.json \
    --tracking tracking_results.json \
    --output actions_with_jerseys.json \
    --jersey 10
```

## Alternative: Simpler Approach Using Just YOLO + OCR

If the full sn-gamestate pipeline is too complex, you can create a simpler tracking script:

```python
# simple_tracking.py - Minimal tracking for ball action matching
import cv2
from ultralytics import YOLO
import json

# Load YOLO model
model = YOLO('yolov8x.pt')

video_path = "/netscratch/eattar/ds/SoccetNet/spotting-ball-2024/england_efl/2019-2020/2019-10-01 - Leeds United - West Bromwich/720p.mp4"
cap = cv2.VideoCapture(video_path)

tracking_results = {"tracks": []}

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect and track persons
    results = model.track(frame, persist=True, classes=[0])  # class 0 = person
    
    # TODO: Add jersey number recognition
    # TODO: Add to tracking_results

# Save results
with open("tracking_results.json", "w") as f:
    json.dump(tracking_results, f)
```

## Troubleshooting

### Issue: sn-gamestate expects different dataset structure
**Solution**: Create a symbolic link or adapt the dataset loader

### Issue: Out of memory during tracking
**Solution**: Reduce batch sizes in `soccernet.yaml`:
```yaml
modules:
  bbox_detector: {batch_size: 2}
  reid: {batch_size: 16}
```

### Issue: Jersey number detection not working
**Solution**: Jersey detection is challenging. You may need to:
- Use the tracklet aggregation voting method
- Manually label a few key players
- Use team affiliation as fallback

## Next Steps

Once you have `tracking_results.json`:
1. Adapt `match_actions_to_players.py` for your tracking format
2. Run the matching to assign jersey numbers to ball actions
3. Analyze player-specific actions with `analyze_player_actions.py`

## Key Files

- `/workspace/sn-gamestate/sn_gamestate/configs/soccernet.yaml` - Main config
- `/workspace/ball-action-spotting/data/.../results_spotting.json` - Ball actions
- `tracking_results.json` - Your tracking output (to be created)
- `actions_with_jerseys.json` - Final matched results
