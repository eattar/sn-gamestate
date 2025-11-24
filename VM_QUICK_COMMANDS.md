# Quick Reference: VM YOLO Training

## 🚀 Fast Track Commands

### 1. Initial Setup (5 minutes)
```bash
ssh eattar@134.96.204.42
cd ~/sn-gamestate
git fetch origin
git checkout ball-action-integration
git pull origin ball-action-integration
```

### 2. Verify Dataset (2 minutes)
```bash
# Dataset should already exist at netscratch location
ls /netscratch/eattar/ds/YOLO

# If missing, download it:
# cd /netscratch/eattar/ds
# wget https://zenodo.org/record/7808511/files/YOLO.zip
# unzip YOLO.zip
# rm YOLO.zip

# Create config file
cat > /netscratch/eattar/ds/YOLO/dataset.yaml << 'EOF'
path: /netscratch/eattar/ds/YOLO
train: train/images
val: valid/images
test: test/images
nc: 2
names: ['ball', 'person']
EOF
```

### 3. Start Training (4-6 hours)
```bash
cd ~/sn-gamestate

# Option A: Run in foreground (stay logged in)
python finetune_yolo.py --dataset-dir /netscratch/eattar/ds/YOLO

# Option B: Run in background (can disconnect)
nohup python finetune_yolo.py --dataset-dir /netscratch/eattar/ds/YOLO > training.log 2>&1 &
tail -f training.log  # Monitor progress
```

### 4. Test When Complete
```bash
# Quick test
python -c "from ultralytics import YOLO; YOLO('runs/train/soccer_ball_soccernet_v3/weights/best.pt').val(data='/netscratch/eattar/ds/YOLO/dataset.yaml')"

# Real-world test
python run_player_ball_actions.py \
  --game SNGS-021 --split valid --team right --jersey 22 \
  --use-ground-truth-detections \
  --labels-path /netscratch/eattar/ds/SoccerNet/2025/data/SoccerNetGS/valid/SNGS-021/Labels-GameState.json \
  --frames-dir /netscratch/eattar/ds/SoccerNet/2025/data/SoccerNetGS/valid/SNGS-021/img1 \
  --ball-model runs/train/soccer_ball_soccernet_v3/weights/best.pt \
  --filter-dark-colors
```

---

## 📊 Expected Results

**Training Time**: 4-6 hours  
**Expected Accuracy**: 80-90% ball detection  
**Improvement**: +30-40% over generic YOLO

---

## 🔍 Monitoring

```bash
# Watch training logs
tail -f training.log

# Check GPU usage
watch -n 1 nvidia-smi

# View progress files
ls runs/train/soccer_ball_soccernet_v3/
cat runs/train/soccer_ball_soccernet_v3/results.csv
```

---

## ⚠️ Troubleshooting

**Out of Memory?**
```bash
python finetune_yolo.py --dataset-dir /netscratch/eattar/ds/YOLO --batch 4
```

**Training Interrupted?**
```bash
python finetune_yolo.py --dataset-dir /netscratch/eattar/ds/YOLO --resume runs/train/soccer_ball_soccernet_v3/weights/last.pt
```

---

## 📥 Download Trained Model

```bash
# On your local machine
scp eattar@134.96.204.42:~/sn-gamestate/runs/train/soccer_ball_soccernet_v3/weights/best.pt \
    ~/dfki_project/sn-gamestate/pretrained_models/yolov8n_soccernet_v3_best.pt
```

---

**Full Instructions**: `VM_YOLO_TRAINING_SETUP.md`
