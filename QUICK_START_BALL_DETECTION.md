# Quick Start: Ball Detection Fine-Tuning

## Status ✅

**FootAndBall pretrained weights downloaded**: `pretrained_models/model_20201019_1416_final.pth` (806 KB)

---

## Option 1: Test FootAndBall Baseline (Optional)

**Purpose**: Establish baseline before fine-tuning  
**Expected Accuracy**: ~20-30% on broadcast footage  
**Time**: 1 hour setup + testing

### Steps

```bash
# 1. Clone FootAndBall repository
cd /Users/eattar/dfki_project
git clone https://github.com/jac99/FootAndBall.git

# 2. Install dependencies (if not already installed)
pip install torch torchvision opencv-python tqdm scipy Pillow

# 3. Copy pretrained weights
cp sn-gamestate/pretrained_models/model_20201019_1416_final.pth \
   FootAndBall/models/

# 4. Test on a video clip
cd FootAndBall
python run_detector.py \
  --path <your_video.mp4> \
  --weights models/model_20201019_1416_final.pth \
  --out_video footandball_results.avi \
  --device cuda
```

**Note**: FootAndBall was trained on close-up sideline footage, so expect poor performance on broadcast long-shots.

---

## Option 2: Fine-Tune YOLO on SoccerNet v3 ⭐ RECOMMENDED

**Purpose**: Production-ready model for broadcast footage  
**Expected Accuracy**: ~80-90%  
**Time**: 4-6 hours training

### Step 1: Download Dataset (2.5 GB)

```bash
cd /Users/eattar/dfki_project/sn-gamestate

# Download SoccerNet v3 H250 dataset
wget https://zenodo.org/record/7808511/files/YOLO.zip

# Extract
unzip YOLO.zip

# Verify structure
ls -R YOLO/
# Should show: train/, valid/, test/ folders with images/ and labels/
```

### Step 2: Prepare Dataset for Training

```bash
# The dataset already has train/valid/test splits
# Just need to create a dataset.yaml file

cat > YOLO/dataset.yaml << EOF
path: YOLO
train: train/images
val: valid/images
test: test/images

nc: 2
names: ['ball', 'person']
EOF
```

### Step 3: Train the Model

```bash
# Fine-tune YOLOv8n on the dataset
# Uses optimized configuration from kmouts/FootAndBall research
python finetune_yolo.py \
  --dataset-yaml YOLO/dataset.yaml \
  --base-model yolov8n.pt \
  --epochs 100 \
  --imgsz 1280 \
  --batch -1 \
  --optimizer SGD \
  --output-name soccer_ball_soccernet_v3
```

**Training Configuration**:
- Base model: YOLOv8n (faster than YOLOv8x, still accurate)
- Image size: 1280px (vs standard 640px - crucial for small balls)
- Batch: Auto-detect (will use maximum GPU allows)
- Optimizer: SGD (better than Adam for this task, per kmouts research)
- Early stopping: patience=5 (stops if no improvement for 5 epochs)
- Expected time: 4-6 hours on GPU

### Step 4: Test the Fine-Tuned Model

```bash
# Test on SoccerNet GameState frames
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

### Step 5: Compare Results

**Before fine-tuning** (generic YOLO):
```bash
# Current system uses YOLOv8x.pt (COCO-trained)
# Accuracy: ~50-60% ball detection
# False positives: shoes, logos, shin guards
```

**After fine-tuning** (SoccerNet v3):
```bash
# Expected accuracy: ~80-90% ball detection
# Reduced false positives (trained on soccer-specific data)
# Better small object detection (1280px images)
```

---

## Expected Timeline

| Phase | Time | Status |
|-------|------|--------|
| FootAndBall baseline test | 1 hour | ✅ Optional |
| Download SoccerNet v3 H250 | 30 min | ⏳ To do |
| Fine-tune YOLO | 4-6 hours | ⏳ To do |
| Test on validation set | 30 min | ⏳ To do |
| **Total** | **5-7 hours** | |

---

## Monitoring Training

During training, monitor these metrics:

```bash
# Watch training progress
tensorboard --logdir runs/train

# Check training logs
tail -f runs/train/soccer_ball_soccernet_v3/results.csv

# Key metrics to watch:
# - mAP50: Should increase to ~0.8-0.9
# - precision: Should increase to ~0.85+
# - recall: Should increase to ~0.80+
# - loss: Should decrease steadily
```

---

## Troubleshooting

### Out of Memory Error
```bash
# Reduce batch size manually
python finetune_yolo.py --batch 4
```

### Slow Training
```bash
# Use smaller model
python finetune_yolo.py --base-model yolov8n.pt  # (already default)

# Reduce image size (not recommended, but faster)
python finetune_yolo.py --imgsz 640
```

### Dataset Not Found
```bash
# Verify dataset structure
ls YOLO/train/images | head
ls YOLO/train/labels | head

# Check dataset.yaml
cat YOLO/dataset.yaml
```

---

## Next Steps After Training

1. **Evaluate on test set**:
   ```bash
   python -c "from ultralytics import YOLO; \
              model = YOLO('runs/train/soccer_ball_soccernet_v3/weights/best.pt'); \
              model.val(data='YOLO/dataset.yaml', split='test')"
   ```

2. **Compare with baseline**:
   - Run on same validation games
   - Compare detection rates
   - Measure false positive reduction

3. **Deploy to production**:
   - Update `run_player_ball_actions.py` default model
   - Test on multiple games
   - Document performance improvements

---

## References

- **FootAndBall pretrained**: `pretrained_models/model_20201019_1416_final.pth`
- **Dataset source**: https://zenodo.org/record/7808511
- **Training config**: Based on kmouts/FootAndBall research
- **Full guide**: See `PRETRAINED_MODELS_GUIDE.md`
