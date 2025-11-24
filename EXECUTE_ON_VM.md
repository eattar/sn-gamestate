# 🚀 Execute Training on VM - Final Instructions

**Status**: All code pushed to `ball-action-integration` branch with correct dataset paths

## Prerequisites ✅

- **VM**: 134.96.204.42
- **Dataset**: Already exists at `/netscratch/eattar/ds/YOLO` (14,368 train images)
- **Expected Time**: 4-6 hours training

---

## Step-by-Step Execution

### 1. SSH to VM and Pull Latest Code (2 minutes)

```bash
ssh eattar@134.96.204.42

cd ~/sn-gamestate
git fetch origin
git checkout ball-action-integration
git pull origin ball-action-integration
```

### 2. Verify Dataset Exists (30 seconds)

```bash
ls -lh /netscratch/eattar/ds/YOLO

# Should see:
# train/  (14,368 images)
# valid/  (2,726 images)
# test/   (2,692 images)
```

### 4. Start Training (4-6 hours) ⏳

**Option A: Foreground (stay logged in)**
```bash
cd ~/sn-gamestate
python finetune_yolo.py --dataset-dir /netscratch/eattar/ds/YOLO
```

**Option B: Background (recommended, can disconnect)**
```bash
cd ~/sn-gamestate
nohup python finetune_yolo.py --dataset-dir /netscratch/eattar/ds/YOLO > training.log 2>&1 &

# Monitor progress
tail -f training.log

# Or check periodically
tail -n 50 training.log
```

### 5. Monitor Training Progress

```bash
# Check GPU usage
nvidia-smi

# Watch log file
tail -f training.log

# Check if training is still running
ps aux | grep finetune_yolo
```

**Expected Output:**
- Epoch 1/100 will take ~3-4 minutes
- Early stopping will trigger if no improvement after 5 epochs
- Best model saved to: `runs/train/soccer_ball_soccernet_v3/weights/best.pt`

### 6. Test Fine-Tuned Model

```bash
cd ~/sn-gamestate

# Quick validation test
python -c "
from ultralytics import YOLO
model = YOLO('runs/train/soccer_ball_soccernet_v3/weights/best.pt')
results = model.val(data='/netscratch/eattar/ds/YOLO/dataset.yaml')
print(f'mAP50: {results.box.map50:.4f}')
print(f'mAP50-95: {results.box.map:.4f}')
print(f'Precision: {results.box.mp:.4f}')
print(f'Recall: {results.box.mr:.4f}')
"
```

**Expected Performance:**
- **mAP50**: 0.80-0.90 (vs 0.50-0.60 with generic YOLO)
- **Precision**: 0.85-0.95
- **Recall**: 0.75-0.90

### 7. Download Model to Local Machine

```bash
# On your LOCAL machine (not VM)
scp eattar@134.96.204.42:~/sn-gamestate/runs/train/soccer_ball_soccernet_v3/weights/best.pt \
    ~/dfki_project/sn-gamestate/pretrained_models/yolov8n_soccernet_v3_best.pt

# Test locally
cd ~/dfki_project/sn-gamestate
python run_player_ball_actions.py \
  --game SNGS-021 --split valid --team right --jersey 22 \
  --ball-model pretrained_models/yolov8n_soccernet_v3_best.pt \
  --output-csv output/jersey_22_finetuned.csv
```

---

## ⚠️ Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
python finetune_yolo.py \
  --dataset-dir /netscratch/eattar/ds/YOLO \
  --batch 4
```

### Training Interrupted (Power/Network Issue)
```bash
# Resume from last checkpoint
python finetune_yolo.py \
  --dataset-dir /netscratch/eattar/ds/YOLO \
  --resume runs/train/soccer_ball_soccernet_v3/weights/last.pt
```

### Dataset Not Found
```bash
# Verify path
ls /netscratch/eattar/ds/YOLO

# Check dataset.yaml path
cat /netscratch/eattar/ds/YOLO/dataset.yaml

# Ensure path is: /netscratch/eattar/ds/YOLO (NOT ~/sn-gamestate/YOLO)
```

---

## 📊 Expected Timeline

| Step | Duration | Status |
|------|----------|--------|
| SSH + Pull Code | 2 min | ⏱️ |
| Verify Dataset | 30 sec | ⏱️ |
| **Training** | **4-6 hours** | ⏱️ |
| Validation Test | 5 min | ⏱️ |
| Download Model | 2 min | ⏱️ |
| **Total** | **~5-7 hours** | |

---

## 📁 Output Files

After training completes, you'll have:

```
runs/train/soccer_ball_soccernet_v3/
├── weights/
│   ├── best.pt           # Best model (use this!)
│   └── last.pt           # Last checkpoint (for resuming)
├── results.png           # Training curves
├── confusion_matrix.png  # Confusion matrix
├── F1_curve.png          # F1 score curve
└── args.yaml             # Training parameters
```

---

## 🎯 Success Criteria

✅ Training completes without OOM errors  
✅ mAP50 > 0.80 (improvement from ~0.55)  
✅ Precision > 0.85  
✅ Model weights saved to `best.pt`  
✅ Local testing shows fewer false positives  

---

## 📚 Additional Documentation

- **Complete Setup**: `VM_YOLO_TRAINING_SETUP.md` (11 detailed steps)
- **Quick Commands**: `VM_QUICK_COMMANDS.md` (copy/paste reference)
- **Model Info**: `PRETRAINED_MODELS_GUIDE.md` (FootAndBall vs YOLO)
- **Fast Track**: `QUICK_START_BALL_DETECTION.md` (30-minute overview)

---

## 🔗 Key Paths Reference

```bash
# Dataset
/netscratch/eattar/ds/YOLO

# Training code
~/sn-gamestate/finetune_yolo.py

# Output model
~/sn-gamestate/runs/train/soccer_ball_soccernet_v3/weights/best.pt
```

---

**Ready to start?** Run Step 1 above! 🚀
