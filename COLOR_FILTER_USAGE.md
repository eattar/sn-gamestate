# Color-Based Ball Detection Filtering

## Overview

Added HSV color-based filtering to reduce false positives from dark objects like shoes, boots, and shin guards during ball detection.

## How It Works

The filter analyzes the **mean brightness (V)** and **saturation (S)** of detected ball regions in HSV color space:

1. **Brightness Filter**: Rejects dark objects (shoes are typically black/brown)
   - Soccer balls are usually white or brightly colored
   - Shoes have low brightness values (V < 60)

2. **Saturation + Brightness Filter**: Rejects desaturated dark objects
   - Prevents gray/dark objects from being detected as balls
   - Allows white balls (low saturation + high brightness)

## Usage

### Enable Color Filtering

```bash
python run_player_ball_actions.py \
  --game SNGS-024 \
  --split valid \
  --team right \
  --jersey 22 \
  --use-ground-truth-detections \
  --labels-path /path/to/Labels-GameState.json \
  --frames-dir /path/to/frames \
  --filter-dark-colors
```

### Adjust Thresholds

```bash
# More aggressive filtering (exclude more dark objects)
--filter-dark-colors \
--min-brightness 80 \
--min-saturation 40

# Less aggressive (keep more detections, may include some false positives)
--filter-dark-colors \
--min-brightness 50 \
--min-saturation 20
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--filter-dark-colors` | False | Enable color-based filtering |
| `--min-brightness` | 60 | Minimum brightness (V in HSV, 0-255) |
| `--min-saturation` | 30 | Minimum saturation (S in HSV, 0-255) |

## HSV Color Space Reference

- **Hue (H)**: Color type (0-180 in OpenCV)
- **Saturation (S)**: Color intensity (0-255)
  - Low saturation = white/gray
  - High saturation = vivid colors
- **Value (V)**: Brightness (0-255)
  - Low value = dark
  - High value = bright

## Examples

### Soccer Ball (White)
- **Saturation**: 10-30 (low, desaturated)
- **Brightness**: 200-250 (high, bright)
- **Result**: ✅ Passes filter (bright despite low saturation)

### Soccer Ball (Colored, e.g., orange)
- **Saturation**: 150-200 (high)
- **Brightness**: 180-230 (high)
- **Result**: ✅ Passes filter (both high)

### Black Shoe
- **Saturation**: 20-50 (low)
- **Brightness**: 10-40 (very low)
- **Result**: ❌ Filtered out (too dark)

### Brown/Dark Shoe
- **Saturation**: 40-80
- **Brightness**: 30-60
- **Result**: ❌ Filtered out (brightness below threshold)

## Debug Visualization

Color filtering happens BEFORE drawing debug bboxes:
- **Blue bbox**: Detection passed ALL filters (including color)
- **Red bbox**: Detection failed ANY filter (height/size/aspect/color)

Debug images show which detections were excluded due to color:
```
frame_162_PASS_search_f162.jpg
```

## Filter Order

Ball detections are filtered in this sequence:
1. ✅ Height constraint (ball on ground, not in sky)
2. ✅ Size constraint (consistent ball dimensions)
3. ✅ Aspect ratio (circular shape, not elongated like shoes)
4. ✅ **Color filter** (bright objects, not dark shoes) ← NEW
5. Distance to player (proximity matching)

## Recommendations

### When to Use

- ✅ YOLO detects many false positives (shoes, shin guards)
- ✅ Camera has good lighting (color information reliable)
- ✅ Ball is white or brightly colored

### When NOT to Use

- ❌ Poor lighting conditions (shadows, night games)
- ❌ Ball is dark-colored (rare but possible)
- ❌ Already using `--skip-ball-verification` (proximity-only mode)

## Combining with Other Filters

Recommended configuration for maximum accuracy:

```bash
python run_player_ball_actions.py \
  --game SNGS-024 \
  --split valid \
  --team right \
  --jersey 22 \
  --use-ground-truth-detections \
  --labels-path /path/to/Labels-GameState.json \
  --frames-dir /path/to/frames \
  --ball-model yolov8x.pt \
  --ball-confidence 0.25 \
  --ball-max-height 0.55 \
  --ball-min-size 18 \
  --ball-max-size 100 \
  --ball-search-window 10 \
  --filter-dark-colors \
  --min-brightness 60 \
  --min-saturation 30
```

## Performance Impact

- Minimal: Color analysis only runs on detections that already passed geometric filters
- Processes small image regions (ball bboxes only)
- HSV conversion is fast with OpenCV

## Troubleshooting

### Filter too aggressive (real balls rejected)

```bash
# Lower thresholds
--min-brightness 50
--min-saturation 20
```

### Filter too loose (shoes still detected)

```bash
# Raise thresholds
--min-brightness 70
--min-saturation 40
```

### Check filter reasons in output

Look for console output like:
```
Frame 162: 5 detections, all filtered: dark=V45, dark=V38, aspect=0.65
```

This tells you why detections were rejected (e.g., `dark=V45` means brightness was 45, below threshold).
