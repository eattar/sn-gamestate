# Fine-Tuning YOLO for Soccer Ball Detection

Complete guide for creating a soccer ball detection dataset and fine-tuning YOLO.

## ⚠️ UPDATE: Pre-Annotated Dataset Available!

**SoccerNet v3 H250 dataset** is now available with 14K+ pre-annotated soccer ball images!

**Skip manual annotation and go straight to training:**
- See `QUICK_START_BALL_DETECTION.md` for the fast path (5-7 hours total)
- See `PRETRAINED_MODELS_GUIDE.md` for all available models

**This guide below** is for custom dataset creation (if needed).

---

## Overview

This process takes ~1 week and significantly improves ball detection accuracy from ~50-60% to ~80-90%.

## Process Flow

```
1. Auto-annotate frames (1 hour)
   ↓
2. Manual review & correction (4-6 hours)
   ↓
3. Fine-tune YOLO (2-4 hours training)
   ↓
4. Test fine-tuned model (1 hour)
```

## Step 1: Auto-Annotate Frames

Use pretrained YOLO to generate initial annotations:

### Option A: From SoccerNet Game Frames

```bash
python auto_annotate_balls.py \
  --frames-dir /netscratch/eattar/ds/SoccerNet/2025/data/SoccerNetGS/valid/SNGS-021/img1 \
  --output-dir ball_dataset \
  --model yolov8x.pt \
  --confidence 0.15 \
  --sample-interval 5
```

**Parameters:**
- `--frames-dir`: Directory with frame images
- `--output-dir`: Where to save annotations (default: ball_dataset)
- `--model`: YOLO model for auto-annotation (yolov8x.pt recommended)
- `--confidence`: Lower = more detections (0.10-0.20 recommended)
- `--sample-interval`: Annotate every Nth frame (5 = every 5th frame)

**Output:**
```
ball_dataset/
├── images/              # Copied frames
│   ├── frame_000000.jpg
│   ├── frame_000001.jpg
│   └── ...
├── labels/              # YOLO format labels
│   ├── frame_000000.txt
│   ├── frame_000001.txt
│   └── ...
├── annotations.json     # Detailed annotations
└── dataset.yaml         # YOLO dataset config
```

### Option B: From Multiple Games (Recommended)

```bash
# Annotate multiple games for diversity
for game in SNGS-021 SNGS-024 SNGS-030; do
  python auto_annotate_balls.py \
    --frames-dir /netscratch/eattar/ds/SoccerNet/2025/data/SoccerNetGS/valid/$game/img1 \
    --output-dir ball_dataset_$game \
    --sample-interval 5
done

# Merge datasets (manual or with script)
```

### Expected Output

```
Auto-Annotation Complete
==================================================
Total frames: 500
Frames with ball detected: 320 (64.0%)
Total detections: 385
Average confidence: 0.428

⚠️  IMPORTANT: These annotations need manual review!
   Run: python review_annotations.py --annotations annotations.json
```

## Step 2: Review and Correct Annotations

Interactive tool to fix false positives and add missed balls:

```bash
python review_annotations.py \
  --annotations ball_dataset/annotations.json
```

### Controls

| Key | Action |
|-----|--------|
| `SPACE` | Next frame |
| `B` | Previous frame |
| `D` | Delete all detections in current frame |
| `A` + Click & Drag | Add new bounding box |
| `Y` | Mark frame as verified (good) |
| `S` | Save and exit |
| `Q` | Quit without saving |
| `H` | Show help |

### Review Strategy

1. **Quick pass (30-60 min)**: Check all frames, delete obvious false positives
2. **Detailed pass (2-3 hours)**: Add missed balls, adjust boxes
3. **Verification pass (1 hour)**: Mark verified frames

**Tips:**
- Focus on frames with actions (PASS, SHOT, DRIVE)
- Ball should be **10-30 pixels** in size
- Delete elongated boxes (shoes) - look for circular shapes
- Add balls even if partially occluded
- Aim for **300-500 good annotations** minimum

### Expected Result

```bash
✅ Saved corrected annotations to: ball_dataset/annotations_reviewed.json
   Frames with ball: 420/500
   Total detections: 450
```

## Step 3: Fine-Tune YOLO

Train YOLO on your corrected dataset:

```bash
python finetune_yolo.py \
  --dataset-dir ball_dataset \
  --base-model yolov8x.pt \
  --epochs 50 \
  --imgsz 1280 \
  --batch 8 \
  --output-name soccer_ball_yolo
```

**Parameters:**
- `--dataset-dir`: Directory with annotations
- `--base-model`: Start from yolov8x.pt (recommended)
- `--epochs`: Training epochs (50-100)
- `--imgsz`: Image size (1280 for high resolution, 640 if GPU memory limited)
- `--batch`: Batch size (reduce if GPU memory error)
- `--train-split`: Train/val ratio (default: 0.8)

### Training Parameters

The script uses optimized settings for fine-tuning:
- **Learning rate**: 0.001 (low for fine-tuning)
- **Optimizer**: AdamW
- **Early stopping**: Patience 15 epochs
- **Augmentation**: Moderate (HSV, rotation, flip, scale)
- **Confidence**: 0.01 during training (low threshold)

### GPU Requirements

| Batch Size | GPU Memory | Training Time (50 epochs) |
|------------|------------|---------------------------|
| 4 | 8 GB | ~3-4 hours |
| 8 | 16 GB | ~2-3 hours |
| 16 | 24 GB | ~1.5-2 hours |

**If GPU memory error:**
```bash
python finetune_yolo.py \
  --dataset-dir ball_dataset \
  --batch 4 \
  --imgsz 640
```

### Expected Output

```
Training Complete!
==================================================
Best model saved to: runs/train/soccer_ball_yolo/weights/best.pt

To use the fine-tuned model:
  python run_player_ball_actions.py \
    --ball-model runs/train/soccer_ball_yolo/weights/best.pt \
    --ball-confidence 0.20 \
    ...
```

## Step 4: Test Fine-Tuned Model

### Quick Test

```bash
python run_player_ball_actions.py \
  --game SNGS-024 \
  --split valid \
  --team right \
  --jersey 22 \
  --use-ground-truth-detections \
  --labels-path /netscratch/eattar/ds/SoccerNet/2025/data/SoccerNetGS/valid/SNGS-024/Labels-GameState.json \
  --frames-dir /netscratch/eattar/ds/SoccerNet/2025/data/SoccerNetGS/valid/SNGS-024/img1 \
  --ball-model runs/train/soccer_ball_yolo/weights/best.pt \
  --ball-confidence 0.20 \
  --filter-dark-colors \
  --ball-search-window 15 \
  --max-ball-distance 80
```

### Compare Results

**Before fine-tuning (yolov8x.pt):**
- Ball detection rate: ~50-60%
- False positives: ~30-40%
- Actions matched: 3-6 per game

**After fine-tuning (expected):**
- Ball detection rate: ~80-90%
- False positives: ~10-15%
- Actions matched: 8-12 per game

### Evaluation Script

```bash
# Test on validation game not used in training
python test_ball_detection.py \
  --model runs/train/soccer_ball_yolo/weights/best.pt \
  --frames-dir /netscratch/eattar/ds/SoccerNet/2025/data/SoccerNetGS/valid/SNGS-050/img1 \
  --confidence 0.20
```

## Tips for Better Results

### 1. Dataset Quality

**Good annotations:**
- ✅ Diverse games (different lighting, camera angles)
- ✅ Mix of close/far balls
- ✅ Include occluded balls
- ✅ Accurate bounding boxes (tight fit)
- ✅ 300-500 quality examples minimum

**Bad annotations:**
- ❌ Single game only
- ❌ All similar frames
- ❌ Loose/oversized boxes
- ❌ Missed balls or false positives

### 2. Data Augmentation

Already included in `finetune_yolo.py`:
- HSV color shifts (different lighting)
- Rotation (±5 degrees)
- Translation & scaling
- Horizontal flip

### 3. Hyperparameter Tuning

If results are poor after 50 epochs:

```bash
# Train longer with more patience
python finetune_yolo.py \
  --dataset-dir ball_dataset \
  --epochs 100 \
  --output-name soccer_ball_yolo_v2
```

### 4. Ensemble Models

Combine multiple models for best results:

```python
models = [
    YOLO('runs/train/soccer_ball_yolo/weights/best.pt'),
    YOLO('yolov8x.pt'),
    YOLO('yolov9e.pt')
]

# Vote on detections
```

## Troubleshooting

### Issue: Low detection rate after training

**Causes:**
- Insufficient training data (< 200 examples)
- All examples from single game (overfitting)
- Confidence threshold too high

**Solutions:**
- Annotate more frames from diverse games
- Lower confidence: `--ball-confidence 0.10`
- Train more epochs

### Issue: High false positive rate

**Causes:**
- Poor annotation quality (included shoes/logos)
- Confidence threshold too low

**Solutions:**
- Re-review annotations, delete false positives
- Increase confidence: `--ball-confidence 0.30`
- Use `--filter-dark-colors` flag

### Issue: GPU out of memory

**Solutions:**
```bash
python finetune_yolo.py \
  --dataset-dir ball_dataset \
  --batch 4 \
  --imgsz 640
```

### Issue: Training not improving

**Check:**
- Validation loss plateau after ~20 epochs? (normal, early stopping will kick in)
- Validation loss increasing? (overfitting, reduce epochs)
- Loss very high? (check dataset quality)

## Advanced: Transfer from Other Datasets

If available, use existing ball tracking datasets:

```bash
# Download tennis/badminton dataset (if available)
# Fine-tune on those first, then on soccer

python finetune_yolo.py \
  --dataset-dir tennis_balls \
  --base-model yolov8x.pt \
  --epochs 30 \
  --output-name ball_detector_tennis

python finetune_yolo.py \
  --dataset-dir soccer_balls \
  --base-model runs/train/ball_detector_tennis/weights/best.pt \
  --epochs 50 \
  --output-name ball_detector_soccer
```

## Summary Checklist

- [ ] Auto-annotate 500-1000 frames from 3-5 games
- [ ] Manual review: 4-6 hours, delete false positives, add missed balls
- [ ] Aim for 300-500 quality annotations
- [ ] Fine-tune with yolov8x.pt base, 50 epochs, batch 8
- [ ] Test on validation game
- [ ] Compare before/after detection rates
- [ ] If < 75% accuracy, annotate more data and retrain

**Expected timeline:** 1 week part-time  
**Expected improvement:** 50-60% → 80-90% detection accuracy
