# Pretrained Models for Soccer Ball Detection

## Overview

This guide covers available pretrained models for soccer ball detection and recommendations for your use case.

---

## 1. FootAndBall Pretrained Model ✅ Downloaded

**Location**: `pretrained_models/model_20201019_1416_final.pth` (806 KB)

### Details
- **Architecture**: FootAndBall (custom FPN-based detector)
- **Trained on**: 
  - ISSIA-CNR Soccer dataset (cameras 1,2,3,4) - close-up sideline footage
  - SoccerPlayerDetection dataset (set 1)
- **Detects**: Both players AND ball (integrated detector)
- **Paper**: [FootAndBall: Integrated Player and Ball Detector](https://www.scitepress.org/Link.aspx?doi=10.5220/0008916000470056) (VISAPP 2020)
- **License**: MIT

### Limitations for Broadcast Footage
❌ **Trained on close-up sideline cameras** (not broadcast long-shots)  
❌ **Ball size mismatch** (10-30px in broadcasts vs 40-60px in training data)  
❌ **Different camera angles** (broadcast overhead vs sideline)  
⚠️ **Likely poor performance** on SoccerNet GameState long-shot footage

### To Use This Model

**Requirements**:
1. Clone the full FootAndBall repository:
   ```bash
   git clone https://github.com/jac99/FootAndBall.git
   ```

2. Install dependencies:
   ```bash
   pip install torch torchvision opencv-python tqdm scipy Pillow
   ```

3. Copy the model to their folder or adapt their code:
   ```bash
   cp pretrained_models/model_20201019_1416_final.pth FootAndBall/models/
   ```

4. Run detection:
   ```bash
   cd FootAndBall
   python run_detector.py \
     --path your_video.mp4 \
     --weights models/model_20201019_1416_final.pth \
     --out_video results.avi \
     --device cuda
   ```

### Quick Test (Optional)
You can test this model to establish a baseline, but **expect low accuracy (~20-30%)** on broadcast footage.

---

## 2. YOLOv8 Fine-tuned on SoccerNet v3 ⭐ RECOMMENDED

**Status**: ❌ NOT available (must train)

### Why This Is Better
✅ **Trained on broadcast footage** (long-shot cameras, person height ≤ 250px)  
✅ **Larger dataset** (14,368 train + 2,726 val images)  
✅ **YOLO architecture** (state-of-the-art, proven on small objects)  
✅ **Image size 1280px** (vs 640px standard, better for small balls)  
✅ **Research-validated** (kmouts paper shows YOLOv8n outperforms FootAndBall)

### Training Configuration (from kmouts/FootAndBall)

```python
from ultralytics.yolo.v8.detect import DetectionTrainer

trainer = DetectionTrainer(overrides={
    "data": "soccernet_v3.yaml",
    "imgsz": 1280,              # Large image size for small balls
    "batch": -1,                # Auto-detect optimal batch size
    "workers": 12,
    "pretrained": True,
    "epochs": 100,
    "patience": 5,              # Early stopping
    "model": "yolov8n.pt",
    "val": True,
    "device": 0,
    "optimizer": "SGD",         # SGD works better than Adam
    "rect": True,               # Rectangular training
    "verbose": True,
    "cache": False
})
trainer.train()
```

### Expected Performance
- **Before fine-tuning**: ~50-60% ball detection (generic YOLO)
- **After fine-tuning**: ~80-90% ball detection (domain-adapted)
- **Training time**: 4-6 hours on GPU (100 epochs with early stopping)

---

## 3. Dataset: SoccerNet v3 H250 ✅ Available

**Download**: https://zenodo.org/record/7808511/files/YOLO.zip (2.5 GB)

### Dataset Details
- **Train**: 14,368 images with ball annotations
- **Validation**: 2,726 images
- **Test**: 2,692 images
- **Format**: YOLO normalized (.txt label files)
- **Classes**: 0=ball, 1=person
- **Filter**: Long-shot frames (person height ≤ 250px)
- **License**: CC BY 4.0 (free to use)

### Download and Setup

```bash
# Download dataset (2.5 GB)
cd /Users/eattar/dfki_project/sn-gamestate
wget https://zenodo.org/record/7808511/files/YOLO.zip

# Extract
unzip YOLO.zip

# Verify structure
ls -R YOLO/
# Expected:
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
```

---

## Recommendation: Decision Tree

```
START: Need ball detection for broadcast footage
  │
  ├─ Want quick baseline test? (Optional)
  │   └─ YES → Test FootAndBall model
  │       ├─ Download FootAndBall repo
  │       └─ Run on sample frames
  │       └─ Expect ~20-30% accuracy
  │
  └─ Want production-ready model? (Recommended)
      └─ YES → Fine-tune YOLO on SoccerNet v3
          ├─ Download SoccerNet v3 H250 dataset (2.5 GB)
          ├─ Train YOLOv8n with kmouts configuration
          ├─ Training time: 4-6 hours
          └─ Expected: ~80-90% accuracy
```

---

## Next Steps

### Option A: Test FootAndBall (Baseline)
```bash
# 1. Clone FootAndBall
git clone https://github.com/jac99/FootAndBall.git

# 2. Test on your frames
cd FootAndBall
python run_detector.py \
  --path /netscratch/eattar/ds/SoccerNet/2025/data/SoccerNetGS/valid/SNGS-021/img1 \
  --weights ../sn-gamestate/pretrained_models/model_20201019_1416_final.pth \
  --out_video footandball_test.avi
```

### Option B: Fine-tune YOLO (Recommended)
```bash
# 1. Download SoccerNet v3 H250 dataset
cd /Users/eattar/dfki_project/sn-gamestate
wget https://zenodo.org/record/7808511/files/YOLO.zip
unzip YOLO.zip

# 2. Update finetune_yolo.py with kmouts configuration
# (Use image size 1280, SGD optimizer, batch=-1)

# 3. Train the model
python finetune_yolo.py \
  --dataset-dir YOLO/ \
  --epochs 100 \
  --imgsz 1280 \
  --batch -1 \
  --optimizer SGD

# 4. Test on your data
python run_player_ball_actions.py \
  --game SNGS-021 \
  --ball-model runs/train/soccer_ball_soccernet_v3/weights/best.pt \
  --filter-dark-colors \
  --min-brightness 60
```

---

## Performance Comparison (Expected)

| Model | Dataset | Accuracy | Training Time | Best For |
|-------|---------|----------|---------------|----------|
| FootAndBall (pretrained) | ISSIA-CNR (sideline) | ~20-30% | 0 (pretrained) | Baseline test |
| YOLOv8x (generic) | COCO (multi-domain) | ~50-60% | 0 (pretrained) | Current system |
| **YOLOv8n (fine-tuned)** | **SoccerNet v3 (broadcast)** | **~80-90%** | **4-6 hours** | **Production** |

---

## References

1. **FootAndBall Paper**: Komorowski et al., "FootAndBall: Integrated Player and Ball Detector", VISAPP 2020
2. **SoccerNet v3 H250 Dataset**: https://zenodo.org/record/7808511
3. **Training Code**: https://github.com/kmouts/FootAndBall (fork with YOLO training)
4. **Original FootAndBall**: https://github.com/jac99/FootAndBall

---

## Summary

✅ **FootAndBall pretrained model downloaded** → Good for baseline comparison  
⭐ **Recommended**: Fine-tune YOLO on SoccerNet v3 H250 → Best for production  
📦 **Dataset ready to download** → 2.5 GB, 14K+ annotated images  
⏱️ **Timeline**: 4-6 hours training → 80-90% accuracy expected
