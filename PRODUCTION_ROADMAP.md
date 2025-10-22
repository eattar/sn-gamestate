# Ball Action Spotting - Production Roadmap

Now that the demo works, here's your path to production deployment.

## ✅ Phase 1: COMPLETED - Demo & Foundation
- [x] Ball action module structure created
- [x] API wrapper implemented
- [x] Configuration system set up
- [x] Demo with sample data working
- [x] Documentation written
- [x] Code pushed to `feature/ball-action-spotting` branch

**Status**: ✅ DONE - Demo works!

---

## 🔄 Phase 2: CURRENT - Real Model Integration

### 2.1 Set Up Ball-Action-Spotting Model

**Tasks:**
1. ✅ Clone repository (already at `/Users/eattar/repos/ball-action-spotting`)
2. Install dependencies:
   ```bash
   cd /Users/eattar/repos/ball-action-spotting
   pip install -r requirements.txt
   ```
3. Download pretrained models:
   - Get weights from [Google Drive](https://drive.google.com/drive/folders/1mIu62cIdsRn3W4o1E5vRR8V5Q1B6HHoz)
   - Place in `data/ball_action/experiments/` following structure in README

4. Test model on a sample game:
   ```bash
   python scripts/ball_action/predict.py \
       --experiment ball_tuning_001 \
       --folds 0 \
       --gpu_id 0
   ```

**Expected Output**: `data/ball_action/predictions/.../results_spotting.json`

### 2.2 Test with Real Predictions

Once you have `results_spotting.json`:

```bash
cd /Users/eattar/repos/sn-gamestate
python examples/analyze_player_actions.py \
    --predictions /path/to/results_spotting.json \
    --jersey-number 10 \
    --output real_player_report.json
```

**Status**: 🔄 IN PROGRESS

---

## 🎯 Phase 3: CRITICAL - Tracking Integration

### The Problem
Right now, the module uses **mock jersey mapping**. It randomly assigns jersey numbers to actions for demonstration. For production, you need **real player-to-action association**.

### The Solution
Connect ball actions with your player tracking system.

### 3.1 Understand Your Tracking Data Structure

First, examine your TrackLab tracking output:

```python
# Load your tracking state
import pickle
with open('/path/to/tracker_state.pkl', 'rb') as f:
    tracking = pickle.load(f)

# Explore the structure
print(type(tracking))
print(dir(tracking))

# Find:
# - How to get detections at a specific frame
# - How to access jersey numbers
# - How to get pitch coordinates
# - How players are identified across frames (track IDs)
```

### 3.2 Implement Jersey Mapping Function

Edit: `sn_gamestate/ball_action/tracking_integration.py`

Replace `find_nearest_player_at_timestamp()` with your logic:

```python
def find_nearest_player_at_timestamp(tracking_results, half, frame_idx, distance_threshold=2.0):
    """Find nearest player to ball at timestamp."""
    
    # 1. Get all players at this frame
    players = tracking_results.get_frame_detections(half, frame_idx)
    
    # 2. Get ball position (if tracked) or estimate from action location
    ball_pos = tracking_results.get_ball_position(half, frame_idx)
    
    # 3. Find nearest player
    nearest = None
    min_dist = float('inf')
    
    for player in players:
        if player.role not in ['player', 'goalkeeper']:
            continue
            
        distance = euclidean_distance(player.pitch_position, ball_pos)
        
        if distance < min_dist and distance < distance_threshold:
            min_dist = distance
            nearest = {
                'jersey_number': player.jersey_number,
                'team': player.team,
                'distance': distance
            }
    
    return nearest
```

### 3.3 Test Integration

```bash
python examples/integrate_with_tracking.py \
    --tracking-results /path/to/tracker_state.pkl \
    --ball-predictions /path/to/results_spotting.json \
    --jersey-number 10 \
    --output integrated_report.json
```

**Validation**:
- Check if jersey numbers make sense
- Verify action counts per player are reasonable
- Compare with manual video inspection for a few actions

**Status**: ⏳ TO DO

---

## 🚀 Phase 4: Production Deployment

### 4.1 Create Full Pipeline Script

Combine everything into one workflow:

```python
# examples/production_pipeline.py
1. Load game video
2. Run player tracking (TrackLab)
3. Run ball action detection
4. Create jersey mapping
5. Generate player reports
6. Save all outputs
```

### 4.2 Batch Processing

Process multiple games:

```bash
# Process all games in a dataset
python examples/batch_process_games.py \
    --games-dir /path/to/games/ \
    --output-dir /path/to/reports/ \
    --jersey-numbers 7,9,10,11
```

### 4.3 Merge with Main Branch

Once everything works:

```bash
# Create pull request
# Go to: https://github.com/eattar/sn-gamestate
# Create PR from feature/ball-action-spotting to main
```

**Status**: ⏳ TO DO

---

## 📊 Phase 5: Advanced Features (Optional)

### 5.1 Visualization
- Add video overlay showing detected actions
- Create heatmaps of player actions on pitch
- Generate timeline visualizations

### 5.2 Advanced Analytics
- Action success rate analysis
- Pass network visualization
- Action context analysis (game situation)

### 5.3 Real-time Processing
- Stream processing for live games
- Real-time player statistics

**Status**: 💡 IDEAS

---

## 🔧 Immediate Action Items

### Right Now (Today):
1. **Test real model** - Run ball-action-spotting on one game
   ```bash
   cd /Users/eattar/repos/ball-action-spotting
   python scripts/ball_action/predict.py --experiment ball_tuning_001 --folds 0
   ```

2. **Analyze real predictions** - Use the generated results
   ```bash
   cd /Users/eattar/repos/sn-gamestate
   python examples/analyze_player_actions.py \
       --predictions [path from step 1] \
       --jersey-number 10
   ```

### This Week:
3. **Examine tracking data structure** - Understand your TrackLab output
4. **Implement jersey mapping** - Connect actions to real players
5. **Test integrated pipeline** - Verify everything works together

### Next Week:
6. **Create production pipeline** - Combine all steps
7. **Test on multiple games** - Validate robustness
8. **Merge to main** - Make it official

---

## 📚 Key Files Reference

| File | Purpose |
|------|---------|
| `GETTING_STARTED_BALL_ACTION.md` | How to get started |
| `sn_gamestate/ball_action/README.md` | Full API documentation |
| `sn_gamestate/ball_action/tracking_integration.py` | **← IMPLEMENT THIS NEXT** |
| `examples/integrate_with_tracking.py` | Test tracking integration |
| `examples/demo_ball_action.py` | ✅ Working demo |
| `examples/analyze_player_actions.py` | Analyze existing predictions |

---

## 🆘 Getting Help

**Questions about:**
- Ball action module: Check `sn_gamestate/ball_action/README.md`
- Model setup: Check [ball-action-spotting README](https://github.com/recokick/ball-action-spotting)
- TrackLab: Check [TrackLab docs](https://github.com/TrackingLaboratory/tracklab)

**Issues:**
- GitHub: https://github.com/eattar/sn-gamestate/issues
- Discord: https://discord.com/invite/cPbqf2mAwF

---

## ✨ Success Criteria

You'll know you're production-ready when:
- ✅ Real ball action predictions work
- ✅ Jersey mapping connects actions to correct players
- ✅ Reports show realistic player statistics
- ✅ Pipeline runs on multiple games without errors
- ✅ Results validate against manual video inspection

**Current Progress**: 📊 30% Complete (Demo phase done, tracking integration pending)
