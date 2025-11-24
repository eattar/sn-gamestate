# VM Setup for YOLO Fine-Tuning on SoccerNet v3 Dataset

Complete instructions for setting up and training YOLO on the VM.

---

## Prerequisites

- VM with GPU access
- CUDA installed
- SSH access to VM
- ~20 GB free disk space (2.5 GB dataset + 5 GB for training artifacts)

---

## Step 1: Pull Latest Code on VM

```bash
# SSH into VM
ssh eattar@134.96.204.42

# Navigate to project
cd ~/sn-gamestate

# Pull latest changes
git fetch origin
git checkout ball-action-integration
git pull origin ball-action-integration

# Verify new files are present
ls -la QUICK_START_BALL_DETECTION.md
ls -la PRETRAINED_MODELS_GUIDE.md
ls -la finetune_yolo.py
```

---

## Step 2: Verify/Download SoccerNet v3 H250 Dataset

**Note**: The dataset should already be at `/netscratch/eattar/ds/YOLO` (2.5 GB).

```bash
# Check if dataset already exists
ls -la /netscratch/eattar/ds/YOLO

# If it exists, skip to Step 3
# If not, download it:
cd /netscratch/eattar/ds

# Download dataset (2.5 GB, takes ~5-10 minutes)
wget https://zenodo.org/record/7808511/files/YOLO.zip

# Check download completed
ls -lh YOLO.zip
# Should show: ~2.5G

# Extract dataset
unzip YOLO.zip

# Verify structure
ls -R YOLO/
# Should show:
# YOLO/
#   train/
#     images/
#     labels/
#   valid/
#     images/
#     labels/
#   test/
#     images/
#     labels/

# Check sample counts
echo "Train images: $(ls YOLO/train/images/*.jpg | wc -l)"
echo "Train labels: $(ls YOLO/train/labels/*.txt | wc -l)"
echo "Valid images: $(ls YOLO/valid/images/*.jpg | wc -l)"
echo "Valid labels: $(ls YOLO/valid/labels/*.txt | wc -l)"
echo "Test images: $(ls YOLO/test/images/*.jpg | wc -l)"
echo "Test labels: $(ls YOLO/test/labels/*.txt | wc -l)"

# Expected output:
# Train images: 14368
# Train labels: 14368
# Valid images: 2726
# Valid labels: 2726
# Test images: 2692
# Test labels: 2692

# Clean up zip file (optional, saves 2.5 GB)
rm YOLO.zip
```

---

## Step 3: Create Dataset Configuration

```bash
# Create dataset.yaml file (in the YOLO directory on netscratch)
cat > /netscratch/eattar/ds/YOLO/dataset.yaml << 'EOF'
# SoccerNet v3 H250 Dataset Configuration
# Long-shot broadcast footage with ball annotations
# Person height <= 250px filter applied

path: /netscratch/eattar/ds/YOLO
train: train/images
val: valid/images
test: test/images

# Classes
nc: 2
names: ['ball', 'person']
EOF

# Verify the file
cat /netscratch/eattar/ds/YOLO/dataset.yaml
```

---

## Step 4: Set Up Python Environment

```bash
# Activate conda environment (if using conda)
conda activate soccernet  # or your environment name

# OR create new environment
# conda create -n yolo_training python=3.10 -y
# conda activate yolo_training

# Install/upgrade required packages
pip install --upgrade ultralytics torch torchvision pyyaml

# Verify installations
python -c "from ultralytics import YOLO; print('Ultralytics OK')"
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Expected output:
# Ultralytics OK
# PyTorch 2.x.x
# CUDA available: True
```

---

## Step 5: Download Base YOLO Model

```bash
# The training script will auto-download yolov8n.pt, but you can pre-download:
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# This downloads ~6 MB yolov8n.pt to ~/.ultralytics/
```

---

## Step 6: Start Training

### Quick Start (Recommended)

```bash
cd ~/sn-gamestate

# Start training with optimized configuration (dataset on netscratch)
python finetune_yolo.py \
  --dataset-yaml /netscratch/eattar/ds/YOLO/dataset.yaml \
  --base-model yolov8n.pt \
  --epochs 100 \
  --imgsz 1280 \
  --batch -1 \
  --optimizer SGD \
  --output-name soccer_ball_soccernet_v3
```

### Run in Background with Logging

```bash
# Run in background with nohup
nohup python finetune_yolo.py \
  --dataset-yaml /netscratch/eattar/ds/YOLO/dataset.yaml \
  --base-model yolov8n.pt \
  --epochs 100 \
  --imgsz 1280 \
  --batch -1 \
  --optimizer SGD \
  --output-name soccer_ball_soccernet_v3 \
  > training.log 2>&1 &

# Save the process ID
echo $! > training.pid

# Monitor training progress
tail -f training.log

# Or use screen/tmux
screen -S yolo_training
python finetune_yolo.py --dataset-yaml /netscratch/eattar/ds/YOLO/dataset.yaml ...
# Detach: Ctrl+A, D
# Reattach: screen -r yolo_training
```

---

## Step 7: Monitor Training

### Watch Training Logs

```bash
# Real-time log monitoring
tail -f training.log

# Or if running in foreground, output will show:
# - Epoch progress
# - Loss values (box_loss, cls_loss, dfl_loss)
# - Metrics (precision, recall, mAP50, mAP50-95)
# - Validation results every epoch
```

### TensorBoard (Optional)

```bash
# In a separate terminal/screen session
conda activate yolo_training
tensorboard --logdir runs/train --host 0.0.0.0 --port 6006

# Access from your local machine:
# ssh -L 6006:localhost:6006 eattar@134.96.204.42
# Then open: http://localhost:6006
```

### Check GPU Usage

```bash
# Monitor GPU utilization
watch -n 1 nvidia-smi

# Or single check
nvidia-smi
```

---

## Step 8: Training Progress Checkpoints

Training saves checkpoints automatically:

```bash
# Check training progress
ls -lh runs/train/soccer_ball_soccernet_v3/weights/

# Files created during training:
# - last.pt       # Latest checkpoint (auto-saved each epoch)
# - best.pt       # Best model based on validation mAP
# - epoch_10.pt   # Saved every 10 epochs

# Check results
cat runs/train/soccer_ball_soccernet_v3/results.csv

# View training curves
ls runs/train/soccer_ball_soccernet_v3/*.png
# - results.png       # Training curves
# - confusion_matrix.png
# - PR_curve.png
# - F1_curve.png
```

---

## Step 9: Verify Training Completion

```bash
# Training is complete when you see:
# - "Training complete (X.XX hours)"
# - "Results saved to runs/train/soccer_ball_soccernet_v3"
# - "Best model: runs/train/soccer_ball_soccernet_v3/weights/best.pt"

# Check the best model exists
ls -lh runs/train/soccer_ball_soccernet_v3/weights/best.pt

# Expected size: ~12-15 MB for YOLOv8n
```

---

## Step 10: Test the Fine-Tuned Model

### Quick Validation Test

```bash
# Run validation on test set
python -c "
from ultralytics import YOLO
model = YOLO('runs/train/soccer_ball_soccernet_v3/weights/best.pt')
results = model.val(data='YOLO/dataset.yaml', split='test')
print(f'Test mAP50: {results.box.map50:.3f}')
print(f'Test mAP50-95: {results.box.map:.3f}')
"
```

### Test on Real SoccerNet GameState Data

```bash
# Test on a validation game
python run_player_ball_actions.py \
  --game SNGS-021 \
  --split valid \
  --team right \
  --jersey 22 \
  --use-ground-truth-detections \
  --labels-path /netscratch/eattar/ds/SoccerNet/2025/data/SoccerNetGS/valid/SNGS-021/Labels-GameState.json \
  --frames-dir /netscratch/eattar/ds/SoccerNet/2025/data/SoccerNetGS/valid/SNGS-021/img1 \
  --ball-model runs/train/soccer_ball_soccernet_v3/weights/best.pt \
  --filter-dark-colors \
  --min-brightness 60 \
  --min-saturation 30 \
  --max-ball-distance 70 \
  --ball-search-window 10
```

---

## Step 11: Compare Before/After Results

### Before Fine-Tuning (Generic YOLOv8x)

```bash
# Run with default COCO-trained model
python run_player_ball_actions.py \
  --game SNGS-021 \
  --split valid \
  --team right \
  --jersey 22 \
  --use-ground-truth-detections \
  --labels-path /netscratch/eattar/ds/SoccerNet/2025/data/SoccerNetGS/valid/SNGS-021/Labels-GameState.json \
  --frames-dir /netscratch/eattar/ds/SoccerNet/2025/data/SoccerNetGS/valid/SNGS-021/img1 \
  --filter-dark-colors \
  > results_before.txt 2>&1

# Count detections
grep "Ball detected" results_before.txt | wc -l
```

### After Fine-Tuning (SoccerNet v3)

```bash
# Run with fine-tuned model
python run_player_ball_actions.py \
  --game SNGS-021 \
  --split valid \
  --team right \
  --jersey 22 \
  --use-ground-truth-detections \
  --labels-path /netscratch/eattar/ds/SoccerNet/2025/data/SoccerNetGS/valid/SNGS-021/Labels-GameState.json \
  --frames-dir /netscratch/eattar/ds/SoccerNet/2025/data/SoccerNetGS/valid/SNGS-021/img1 \
  --ball-model runs/train/soccer_ball_soccernet_v3/weights/best.pt \
  --filter-dark-colors \
  > results_after.txt 2>&1

# Count detections
grep "Ball detected" results_after.txt | wc -l
```

---

## Expected Training Timeline

| Phase | Time | Details |
|-------|------|---------|
| Dataset download | 5-10 min | 2.5 GB from Zenodo |
| Dataset extraction | 2-3 min | Unzip 14K+ images |
| Environment setup | 2-5 min | Install packages |
| **Training** | **4-6 hours** | **100 epochs with early stopping** |
| Validation testing | 5-10 min | Test on validation set |
| Real-world testing | 10-15 min | Test on GameState data |
| **Total** | **~5-7 hours** | |

---

## Expected Metrics

### Target Performance (from kmouts research)

```
Training metrics after ~50-100 epochs:
- mAP50: 0.80-0.90
- mAP50-95: 0.60-0.70
- Precision: 0.85-0.92
- Recall: 0.78-0.88
- Ball class mAP50: 0.75-0.85
- Person class mAP50: 0.90-0.95
```

### Real-World Performance

```
Before fine-tuning:
- Ball detection rate: ~50-60%
- False positives: High (shoes, logos, shin guards)
- Small ball misses: Common

After fine-tuning:
- Ball detection rate: ~80-90%
- False positives: Reduced
- Small ball detection: Improved
```

---

## Troubleshooting

### Out of Memory Error

```bash
# Reduce batch size manually (instead of -1 auto)
python finetune_yolo.py --batch 4 --dataset-yaml /netscratch/eattar/ds/YOLO/dataset.yaml
```

### Dataset Not Found

```bash
# Verify paths in dataset.yaml
cat /netscratch/eattar/ds/YOLO/dataset.yaml

# Update path to absolute path if needed
path: /netscratch/eattar/ds/YOLO
```

### CUDA Out of Memory

```bash
# Use smaller image size (less accurate but works)
python finetune_yolo.py --imgsz 640 --dataset-yaml /netscratch/eattar/ds/YOLO/dataset.yaml

# Or use CPU (very slow, not recommended)
python finetune_yolo.py --device cpu --dataset-yaml /netscratch/eattar/ds/YOLO/dataset.yaml
```

### Training Stuck/Slow

```bash
# Check GPU usage
nvidia-smi

# Check if other processes are using GPU
fuser -v /dev/nvidia*

# Reduce workers if I/O is bottleneck
python finetune_yolo.py --workers 4 --dataset-yaml /netscratch/eattar/ds/YOLO/dataset.yaml
```

### Resume Training After Interruption

```bash
# Resume from last checkpoint
python finetune_yolo.py \
  --dataset-yaml /netscratch/eattar/ds/YOLO/dataset.yaml \
  --resume runs/train/soccer_ball_soccernet_v3/weights/last.pt
```

---

## Post-Training: Download Model to Local

```bash
# On your local machine
scp eattar@134.96.204.42:~/sn-gamestate/runs/train/soccer_ball_soccernet_v3/weights/best.pt \
    ~/dfki_project/sn-gamestate/pretrained_models/yolov8n_soccernet_v3_best.pt

# Or download entire results folder
scp -r eattar@134.96.204.42:~/sn-gamestate/runs/train/soccer_ball_soccernet_v3/ \
    ~/dfki_project/sn-gamestate/training_results/
```

---

## Quick Command Reference

```bash
# Dataset download
wget https://zenodo.org/record/7808511/files/YOLO.zip && unzip YOLO.zip

# Start training
python finetune_yolo.py --dataset-yaml /netscratch/eattar/ds/YOLO/dataset.yaml

# Monitor
tail -f training.log
watch -n 1 nvidia-smi

# Test
python -c "from ultralytics import YOLO; YOLO('runs/train/soccer_ball_soccernet_v3/weights/best.pt').val(data='/netscratch/eattar/ds/YOLO/dataset.yaml')"

# Use in production
python run_player_ball_actions.py --ball-model runs/train/soccer_ball_soccernet_v3/weights/best.pt ...
```

---

## Summary Checklist

- [ ] SSH into VM
- [ ] Pull latest code (`git pull origin ball-action-integration`)
- [ ] Download SoccerNet v3 H250 dataset (2.5 GB)
- [ ] Extract and verify dataset structure
- [ ] Create `/netscratch/eattar/ds/YOLO/dataset.yaml` configuration
- [ ] Activate Python environment
- [ ] Start training (`python finetune_yolo.py ...`)
- [ ] Monitor training progress (4-6 hours)
- [ ] Verify best model created
- [ ] Test on validation set
- [ ] Test on real GameState data
- [ ] Download trained model to local machine
- [ ] Update production scripts to use fine-tuned model

---

## Support Files

- `QUICK_START_BALL_DETECTION.md` - Fast-track guide
- `PRETRAINED_MODELS_GUIDE.md` - All available models overview
- `YOLO_FINETUNING_GUIDE.md` - Manual annotation guide (if needed)
- `finetune_yolo.py` - Training script with optimized config

---

**Estimated Total Time: 5-7 hours (mostly training)**  
**Expected Accuracy Improvement: 50-60% → 80-90%**
