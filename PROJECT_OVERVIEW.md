# SoccerNet Game State Reconstruction - Project Overview

**Generated**: 2025-11-24  
**Location**: `/Users/eattar/dfki_project/sn-gamestate`

---

## 🎯 Project Purpose

This project is the **SoccerNet Game State Reconstruction (GSR)** baseline implementation - a computer vision system that tracks and identifies soccer players from broadcast video to create a game-like minimap visualization.

### Core Capabilities
- **Player Tracking**: Multi-object tracking of all players on the field
- **Player Identification**: Jersey number recognition via OCR
- **Team Assignment**: Automatic clustering into left/right teams
- **Pitch Calibration**: Camera calibration and field localization
- **Ball Action Integration**: NEW - Detect ball actions (PASS, DRIVE) for specific players

---

## 📁 Project Structure

```
sn-gamestate/
├── README.md                          # Main documentation
├── pyproject.toml                     # Python dependencies (uv/pip)
├── sn_gamestate/                      # Main source code
│   ├── configs/                       # Hydra configuration files
│   │   ├── soccernet.yaml            # Main config (pipeline, paths, modules)
│   │   ├── modules/                  # Module-specific configs
│   │   │   ├── bbox_detector/        # YOLO detection configs
│   │   │   ├── reid/                 # Re-identification configs
│   │   │   ├── track/                # Tracking algorithm configs
│   │   │   ├── jersey_number_detect/ # OCR configs
│   │   │   ├── team/                 # Team clustering configs
│   │   │   ├── calibration/          # Camera calibration configs
│   │   │   └── pitch/                # Pitch localization configs
│   │   └── dataset/                  # Dataset configs
│   ├── calibration/                   # Camera calibration modules
│   ├── jersey/                        # Jersey number detection
│   ├── reid/                          # Re-identification features
│   ├── team/                          # Team assignment
│   └── visualization/                 # Minimap visualization
├── plugins/                           # Plugin modules
│   └── calibration/                   # Calibration plugin (61 files)
├── pretrained_models/                 # Model weights directory
├── run_player_ball_actions.py        # ⭐ Ball action integration script
├── example_ball_action_integration.py # Example usage
├── finetune_yolo.py                  # YOLO fine-tuning script
├── auto_annotate_balls.py            # Ball annotation helper
├── test_footandball_model.py         # Model testing
└── tracknet_detector.py              # TrackNet ball detector

Documentation Files:
├── BALL_ACTION_INTEGRATION.md        # ⭐ Ball action integration guide
├── QUICK_START.md                    # Quick start for ball actions
├── QUICK_START_BALL_DETECTION.md     # Ball detection guide
├── VM_SETUP_INTEGRATION.md           # VM setup instructions
├── VM_QUICK_COMMANDS.md              # Quick command reference
├── EXECUTE_ON_VM.md                  # VM execution guide
├── YOLO_FINETUNING_GUIDE.md         # YOLO training guide
├── PRETRAINED_MODELS_GUIDE.md        # Model comparison
├── TRACKNET_USAGE.md                 # TrackNet usage
├── COLOR_FILTER_USAGE.md             # Color filtering guide
└── ChallengeRules.md                 # Challenge submission rules
```

---

## 🔧 Technology Stack

### Core Framework
- **TrackLab**: Modular multi-object tracking framework (parent framework)
- **Python**: 3.9 (required)
- **Package Manager**: `uv` (recommended) or `conda`

### Key Dependencies
- **PyTorch**: 1.13.1 (deep learning)
- **Ultralytics YOLO**: Player/ball detection
- **PRTReId**: Multi-task re-identification (player appearance + team + role)
- **MMOCR**: Jersey number OCR
- **TVCalib/NBJW**: Camera calibration
- **SoccerNet**: Dataset utilities
- **Hydra**: Configuration management

### Models Used
1. **YOLOv11**: Bounding box detection (players, referees)
2. **PRTReId**: Player re-identification + team affiliation
3. **StrongSort**: Multi-object tracking
4. **MMOCR**: Jersey number recognition
5. **K-means**: Team clustering
6. **Ball-Action Model**: EfficientNetV2 + 3D CNN (PASS/DRIVE detection)

---

## 🚀 Main Features

### 1. Player Tracking Pipeline
```
Video Input → YOLO Detection → PRTReId Features → StrongSort Tracking
          → Jersey OCR → Team Clustering → Minimap Output
```

**Modules in Pipeline** (from `soccernet.yaml`):
1. `bbox_detector`: Detect player bounding boxes
2. `reid`: Extract appearance features
3. `track`: Track players across frames
4. `pitch`: Localize pitch in image
5. `calibration`: Calibrate camera
6. `jersey_number_detect`: OCR jersey numbers
7. `tracklet_agg`: Vote for consistent jersey per track
8. `team`: Cluster into 2 teams
9. `team_side`: Assign left/right based on position

### 2. Ball Action Integration (NEW)
**Script**: `run_player_ball_actions.py`

**Purpose**: Extract ball actions (PASS, DRIVE) performed by a specific player

**Workflow**:
```
1. Run SN-GameState tracking → Get player detections with jersey numbers
2. Run ball-action detection → Get action frames (PASS/DRIVE)
3. Match actions to player → Filter by team + jersey + temporal proximity
4. Output JSON → Timestamped actions for the player
```

**Usage**:
```bash
python run_player_ball_actions.py \
  --game SNGS-001 \
  --split valid \
  --team left \
  --jersey 10 \
  --state-cache tracking_state.pklz
```

**Output Format**:
```json
{
  "player": {"team": "left", "jersey": 10},
  "total_actions": 42,
  "action_counts": {"PASS": 28, "DRIVE": 14},
  "actions": [
    {
      "action": "PASS",
      "frame": 3350,
      "time": "2:14",
      "confidence": 0.876,
      "player_frame": 3348,
      "frame_offset": 2
    }
  ]
}
```

---

## 📊 Dataset

### SoccerNet Game State Dataset
- **Location**: `/netscratch/eattar/ds/SoccerNet/2024/data/SoccerNetGS`
- **Splits**: train, valid, test, challenge
- **Format**: Image frames (720p) + JSON annotations
- **Version**: 1.3 (latest)

### Dataset Structure
```
SoccerNetGS/
├── train/
│   ├── SNGS-001/
│   │   ├── 000001.jpg
│   │   ├── 000002.jpg
│   │   ├── ...
│   │   └── Labels-GameState.json
│   ├── SNGS-002/
│   └── ...
├── valid/
├── test/
└── challenge/
```

### Annotations (Labels-GameState.json)
- **Bounding boxes**: Player locations in image
- **Jersey numbers**: Ground truth jersey numbers
- **Teams**: Left/right team assignment
- **Roles**: Player, goalkeeper, referee, other
- **Pitch coordinates**: 2D field positions

---

## ⚙️ Configuration System

### Main Config: `sn_gamestate/configs/soccernet.yaml`

**Key Settings**:
```yaml
# Paths
data_dir: "/netscratch/eattar/ds/SoccerNet/2024/data"
output_dir: "/netscratch/eattar/SoccerNet/outputs"
model_dir: "${project_dir}/pretrained_models"

# Pipeline (order matters!)
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

# Dataset
dataset:
  eval_set: "valid"  # train/valid/test/challenge
  nvid: -1           # -1 = all videos, 1 = first video only
  dataset_path: "${data_dir}/SoccerNetGS"

# State Caching (save/load tracking results)
state:
  save_file: "states/${experiment_name}.pklz"
  load_file: null  # Set to path to skip tracking

# Module Batch Sizes
modules:
  bbox_detector: {batch_size: 8}
  reid: {batch_size: 64}
  track: {batch_size: 64}
  jersey_number_detect: {batch_size: 8}
```

### Hydra Configuration
- **Framework**: Facebook Hydra (compositional configs)
- **Override syntax**: `key=value` on command line
- **Example**: `uv run tracklab -cn soccernet dataset.eval_set=test`

---

## 🎮 Running the System

### 1. Basic Tracking (Single Video)
```bash
# Run on first validation video
uv run tracklab -cn soccernet

# Run on specific video
uv run tracklab -cn soccernet dataset.vids_dict.valid=[SNGS-021]

# Run on test set
uv run tracklab -cn soccernet dataset.eval_set=test
```

### 2. Ball Action Integration
```bash
# Step 1: Generate tracking state (if not cached)
uv run tracklab -cn soccernet dataset.eval_set=valid dataset.vids_dict.valid=[SNGS-001]

# Step 2: Extract player actions
python run_player_ball_actions.py \
  --game SNGS-001 \
  --split valid \
  --team left \
  --jersey 10 \
  --state-cache outputs/.../states/sn-gamestate.pklz
```

### 3. Using Cached States (Fast)
```bash
# Save state on first run
uv run tracklab -cn soccernet state.save_file=states/my_state.pklz

# Load state on subsequent runs (skip tracking)
uv run tracklab -cn soccernet \
  state.load_file=states/my_state.pklz \
  pipeline=[]  # Empty pipeline = only load
```

---

## 📈 Evaluation Metric: GS-HOTA

### What is GS-HOTA?
Game State HOTA - extends standard HOTA metric for GSR task

**Formula**:
```
Sim_GS-HOTA(P, G) = LocSim(P, G) × IdSim(P, G)

LocSim(P, G) = exp(ln(0.05) × ||P - G||²/τ²)  # τ=5m tolerance
IdSim(P, G) = 1 if all attributes match, else 0
```

**Attributes Checked**:
- Role (player/goalkeeper/referee/other)
- Team (left/right)
- Jersey number

**Strict Constraint**: All attributes must match exactly, or detection is False Positive

---

## 🔬 Ball Action Integration Details

### Current Capabilities
- ✅ Detects **PASS** and **DRIVE** actions (2/12 classes)
- ✅ Matches actions to specific player by team + jersey
- ✅ Temporal matching (±50 frame window)
- ✅ JSON output with timestamps
- ✅ State caching for performance

### Limitations
- ⚠️ Only 2 action classes (model limitation)
- ⚠️ Temporal matching only (no spatial IoU yet)
- ⚠️ Requires SoccerNetGS dataset structure

### Future Improvements
- [ ] Support all 12 action classes (train new model)
- [ ] Add spatial matching (bbox IoU)
- [ ] Support arbitrary video input
- [ ] Real-time processing
- [ ] Web UI

---

## 🖥️ VM Setup

### VM Details
- **Address**: 134.96.204.42
- **User**: eattar
- **Dataset**: `/netscratch/eattar/ds/SoccerNet/2024/data/SoccerNetGS`
- **Output**: `/netscratch/eattar/SoccerNet/outputs`

### Environment Strategy
```bash
# Option 1: Use conda env (for ball-action integration)
conda activate ball-action-spotting
cd ~/sn-gamestate
pip install -e .

# Option 2: Use uv (for standard tracking)
cd ~/sn-gamestate
uv venv --python 3.9
uv pip install -e .
```

### Quick Commands
```bash
# SSH to VM
ssh eattar@134.96.204.42

# Pull latest code
cd ~/sn-gamestate
git pull origin ball-action-integration

# Run tracking
uv run tracklab -cn soccernet

# Run ball action integration
python run_player_ball_actions.py --game SNGS-001 --split valid --team left --jersey 10
```

---

## 📚 Key Documentation Files

### Getting Started
1. **README.md**: Main project documentation
2. **QUICK_START.md**: Ball action quick start
3. **QUICK_START_BALL_DETECTION.md**: Ball detection guide

### Ball Action Integration
1. **BALL_ACTION_INTEGRATION.md**: Complete integration guide (396 lines)
2. **example_ball_action_integration.py**: Usage examples
3. **run_player_ball_actions.py**: Main integration script (1933 lines)

### VM Deployment
1. **VM_SETUP_INTEGRATION.md**: VM setup for ball actions
2. **VM_QUICK_COMMANDS.md**: Quick command reference
3. **EXECUTE_ON_VM.md**: Training execution guide

### Advanced Topics
1. **YOLO_FINETUNING_GUIDE.md**: Fine-tune YOLO on soccer balls
2. **PRETRAINED_MODELS_GUIDE.md**: Model comparison
3. **TRACKNET_USAGE.md**: TrackNet ball detector
4. **COLOR_FILTER_USAGE.md**: Color-based filtering

---

## 🧪 Testing & Validation

### Test Scripts
```bash
# Test ball detection model
python test_footandball_model.py

# Run example integration (no models needed)
python example_ball_action_integration.py

# Validate tracking on single video
uv run tracklab -cn soccernet dataset.nvid=1
```

### Validation Workflow
1. Run tracking on validation set
2. Compare with ground truth (Labels-GameState.json)
3. Compute GS-HOTA metric
4. Analyze errors (jersey OCR, team assignment, etc.)

---

## 🔗 Related Projects

### Dependencies
- **TrackLab**: https://github.com/TrackingLaboratory/tracklab
- **PRTReId**: https://github.com/VlSomers/prtreid
- **BPBreID**: https://github.com/VlSomers/bpbreid
- **TVCalib**: https://github.com/MM4SPA/tvcalib

### Ball Action Spotting
- **Repository**: `/Users/eattar/dfki_project/ball-action-spotting`
- **Model**: EfficientNetV2-B0 + 3D CNN
- **Accuracy**: 90.1% (fold 5)
- **Classes**: PASS, DRIVE (2/12)

---

## 🎓 Academic Context

### Paper
**SoccerNet Game State Reconstruction: End-to-End Athlete Tracking and Identification on a Minimap**
- Conference: CVPRW 2024
- Authors: Somers et al.
- arXiv: 2404.11335

### Challenge
- **Platform**: Codabench
- **Evaluation**: GS-HOTA metric
- **Deadline**: May 30, 2024 (past)

### Citation
```bibtex
@inproceedings{Somers2024SoccerNetGameState,
  title = {SoccerNet Game State Reconstruction: End-to-End Athlete Tracking and Identification on a Minimap},
  author = {Somers, Vladimir and Joos, Victor and ...},
  booktitle = {CVPRW},
  year = {2024}
}
```

---

## 🛠️ Development Workflow

### Adding a New Module
1. Create module in `sn_gamestate/` or `plugins/`
2. Add config in `sn_gamestate/configs/modules/`
3. Register in `soccernet.yaml` defaults
4. Add to pipeline list

### Modifying Pipeline
Edit `soccernet.yaml`:
```yaml
pipeline:
  - bbox_detector
  - reid
  - track
  # - pitch  # Comment out to skip
  # - calibration
  - jersey_number_detect
  - tracklet_agg
  - team
  - team_side
```

### Debugging
```bash
# Print full config
uv run tracklab -cn soccernet print_config=True

# Run on single frame
uv run tracklab -cn soccernet dataset.nframes=1

# Enable verbose logging
uv run tracklab -cn soccernet use_rich=True
```

---

## 📊 Performance Benchmarks

### Processing Time (720p video, ~90 min match)
| Module | Time | Notes |
|--------|------|-------|
| YOLO Detection | ~5 min | GPU-dependent |
| PRTReId | ~8 min | Batch size 64 |
| Tracking | ~2 min | CPU-based |
| Jersey OCR | ~10 min | MMOCR |
| Team Clustering | ~1 min | K-means |
| **Total** | **~30 min** | With GPU |

### Ball Action Integration
| Step | Time | Notes |
|------|------|-------|
| Player Tracking | ~15-20 min | Can be cached |
| Ball Action Detection | ~15-20 min | GPU-dependent |
| Matching | ~1-2 sec | Fast |
| **Total (first run)** | **~30-40 min** | - |
| **Total (cached)** | **~15-20 min** | Skip tracking |

---

## 🚨 Common Issues & Solutions

### Issue: CUDA Out of Memory
```bash
# Reduce batch sizes in soccernet.yaml
modules:
  bbox_detector: {batch_size: 4}  # Was 8
  reid: {batch_size: 32}          # Was 64
```

### Issue: Jersey Detection Errors
- **Cause**: OCR quality, player visibility
- **Solution**: Check `tracklet_agg` voting, increase frames

### Issue: Team Assignment Wrong
- **Cause**: K-means clustering on appearance features
- **Solution**: Use `team_side` module (position-based)

### Issue: No Player Found (Ball Actions)
- **Cause**: Jersey number not detected
- **Solution**: Check available combinations in error output

---

## 📞 Support & Resources

### Documentation
- **Main README**: Complete setup and usage
- **TrackLab Docs**: https://github.com/TrackingLaboratory/tracklab
- **SoccerNet Website**: https://www.soccer-net.org/tasks/new-game-state-reconstruction

### Community
- **Discord**: https://discord.com/invite/cPbqf2mAwF
- **GitHub Issues**: https://github.com/SoccerNet/sn-gamestate/issues

### Videos
- **Demo**: https://www.youtube.com/watch?v=0JRB7hjyOOk
- **Tutorial**: https://www.youtube.com/watch?v=Ir-6D3j_lkA
- **Intro**: https://www.youtube.com/watch?v=UDeSdOR9Ing

---

## ✅ Quick Reference Checklist

### First-Time Setup
- [ ] Clone repository
- [ ] Install dependencies (`uv pip install -e .`)
- [ ] Download dataset (auto or manual)
- [ ] Download model weights (auto or manual)
- [ ] Configure paths in `soccernet.yaml`
- [ ] Test on single video

### Running Ball Action Integration
- [ ] Ensure ball-action-spotting repo exists
- [ ] Download ball-action model weights
- [ ] Generate tracking state (or use cached)
- [ ] Run `run_player_ball_actions.py`
- [ ] Verify JSON output

### VM Deployment
- [ ] SSH to VM
- [ ] Pull latest code
- [ ] Activate conda/uv environment
- [ ] Verify dataset paths
- [ ] Run training/inference
- [ ] Download results

---

## 🎯 Project Status

**Current Branch**: `ball-action-integration`

**Completed**:
- ✅ Full tracking pipeline (YOLO + PRTReId + StrongSort)
- ✅ Jersey number recognition (MMOCR)
- ✅ Team assignment (K-means + position)
- ✅ Ball action integration (PASS/DRIVE)
- ✅ State caching for performance
- ✅ Comprehensive documentation

**In Progress**:
- 🔄 YOLO fine-tuning on soccer balls
- 🔄 TrackNet integration for ball detection
- 🔄 Color filtering for false positives

**Future Work**:
- 📋 Support all 12 action classes
- 📋 Spatial matching (bbox IoU)
- 📋 Arbitrary video input
- 📋 Real-time processing
- 📋 Web UI

---

**Last Updated**: 2025-11-24  
**Maintainer**: eattar  
**License**: MIT (SN-GameState) + Apache 2.0 (Ball-Action)
