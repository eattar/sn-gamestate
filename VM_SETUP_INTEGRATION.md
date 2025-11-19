# VM Setup - Ball Action Integration

Quick guide for running the integration on VM using `uv` (sn-gamestate) + conda (ball-action-spotting).

## Prerequisites

- ✅ `uv` installed for sn-gamestate
- ✅ conda environment for ball-action-spotting
- ✅ Both repos cloned on VM
- ✅ Ball-action model weights downloaded

## Setup Steps

### 1. SSH to VM
```bash
ssh your-vm-address
```

### 2. Pull Latest Integration Code
```bash
# Pull sn-gamestate integration branch
cd /workspace/sn-gamestate
git fetch origin
git checkout ball-action-integration
git pull origin ball-action-integration

# Verify files are there
ls -la run_player_ball_actions.py
```

### 3. Setup Environment Strategy

Since you use different package managers, use this approach:

```bash
# Activate ball-action conda environment
cd /workspace/ball-action-spotting
conda activate ball-action-spotting  # or your env name

# Install sn-gamestate in the conda env (so both work together)
cd /workspace/sn-gamestate
pip install -e .

# Verify both work
python -c "import tracklab; print('✓ SN-GameState OK')"
python -c "from src.predictors import MultiDimStackerPredictor; print('✓ Ball-Action OK')"
```

## Running the Integration

### Basic Usage
```bash
# Make sure conda env is active
conda activate ball-action-spotting

cd /workspace/sn-gamestate
python run_player_ball_actions.py \
    --video /path/to/video.mp4 \
    --team left \
    --jersey 10
```

### With All Options
```bash
python run_player_ball_actions.py \
    --video /path/to/match.mp4 \
    --team right \
    --jersey 7 \
    --output player7_actions.json \
    --experiment ball_finetune_long_004 \
    --fold 5 \
    --device cuda:0
```

### Using Cached State (Fast Re-runs)
```bash
# First run - creates cache
python run_player_ball_actions.py \
    --video video.mp4 \
    --team left \
    --jersey 10

# Second run - uses cache (skips tracking, ~2x faster)
python run_player_ball_actions.py \
    --video video.mp4 \
    --team right \
    --jersey 7 \
    --state-cache tracking_state_temp.pklz \
    --skip-tracking
```

## Alternative: Run with uv directly

If you want to use `uv` to run the script:

```bash
cd /workspace/sn-gamestate

# Use uv run (will use sn-gamestate's environment)
uv run python run_player_ball_actions.py \
    --video video.mp4 \
    --team left \
    --jersey 10
```

**Note**: This requires ball-action-spotting to be accessible from uv's environment.

## Recommended: Create Convenience Script

Create `/workspace/run_ball_action.sh`:
```bash
#!/bin/bash
# Convenience script to run ball action integration

# Activate conda env
source ~/miniconda3/etc/profile.d/conda.sh  # Adjust path if needed
conda activate ball-action-spotting

# Run the integration
cd /workspace/sn-gamestate
python run_player_ball_actions.py "$@"
```

Make it executable:
```bash
chmod +x /workspace/run_ball_action.sh
```

Usage:
```bash
/workspace/run_ball_action.sh --video video.mp4 --team left --jersey 10
```

## Directory Structure on VM

```
/workspace/
├── ball-action-spotting/
│   ├── data/
│   │   └── ball_action/
│   │       └── experiments/
│   │           └── ball_finetune_long_004/
│   │               └── fold_5/
│   │                   └── model-006-0.901643.pth
│   ├── src/
│   └── (conda env: ball-action-spotting)
│
└── sn-gamestate/
    ├── run_player_ball_actions.py ← Main script
    ├── example_ball_action_integration.py
    ├── BALL_ACTION_INTEGRATION.md
    ├── pretrained_models/
    └── (uses uv, but script runs in conda env)
```

## Quick Test

```bash
# 1. Activate conda env
conda activate ball-action-spotting

# 2. Test example script (no models needed)
cd /workspace/sn-gamestate
python example_ball_action_integration.py

# 3. Check help
python run_player_ball_actions.py --help

# 4. Run on video
python run_player_ball_actions.py \
    --video /path/to/video.mp4 \
    --team left \
    --jersey 10
```

## Troubleshooting

### Issue: Import errors from ball-action-spotting
```bash
# Solution: Make sure conda env is active
conda activate ball-action-spotting
python -c "from src.predictors import MultiDimStackerPredictor; print('OK')"
```

### Issue: Import errors from sn-gamestate
```bash
# Solution: Install sn-gamestate in conda env
cd /workspace/sn-gamestate
pip install -e .
python -c "import tracklab; print('OK')"
```

### Issue: CUDA/GPU not detected
```bash
# Check GPU
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Issue: Model weights not found
```bash
# Check model exists
ls -lh /workspace/ball-action-spotting/data/ball_action/experiments/ball_finetune_long_004/fold_5/*.pth

# If missing, download from Google Drive (see ball-action-spotting README)
```

## Performance Tips

1. **Use tmux for long runs**:
   ```bash
   tmux new -s integration
   conda activate ball-action-spotting
   python run_player_ball_actions.py --video video.mp4 --team left --jersey 10
   # Ctrl+B, D to detach
   tmux attach -t integration  # to resume
   ```

2. **Cache tracking state** to analyze multiple players from same video:
   ```bash
   # Run once, saves tracking_state_temp.pklz
   python run_player_ball_actions.py --video video.mp4 --team left --jersey 10
   
   # Reuse for other players (much faster)
   python run_player_ball_actions.py --video video.mp4 --team left --jersey 7 \
       --state-cache tracking_state_temp.pklz --skip-tracking
   ```

3. **Process multiple players** with a simple loop:
   ```bash
   for jersey in 7 9 10 11; do
       python run_player_ball_actions.py \
           --video video.mp4 \
           --team left \
           --jersey $jersey \
           --state-cache tracking_state_temp.pklz \
           --skip-tracking \
           --output "player_${jersey}_actions.json"
   done
   ```

## Auto-Activation on VM Login

Add to `~/.bashrc`:
```bash
# Auto-activate ball-action-spotting environment
conda activate ball-action-spotting 2>/dev/null || true

# Add convenience alias
alias ball-action='cd /workspace/sn-gamestate && python run_player_ball_actions.py'
```

Then:
```bash
source ~/.bashrc

# Now you can just run
ball-action --video video.mp4 --team left --jersey 10
```

## Summary

**Recommended Workflow:**
1. ✅ Keep conda env for ball-action-spotting active
2. ✅ Install sn-gamestate in that conda env with `pip install -e .`
3. ✅ Run the integration script from conda env
4. ✅ Use state caching for analyzing multiple players

This avoids environment conflicts and keeps everything simple!
