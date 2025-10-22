"""
Inspect ball action predictions from ball-action-spotting model.

This script analyzes the raw predictions to understand:
- Total number of actions
- Actions per half
- Confidence distribution
- Temporal distribution
"""

import json
from pathlib import Path
from collections import Counter, defaultdict


def load_predictions(json_path: str) -> dict:
    """Load predictions from results_spotting.json"""
    with open(json_path, 'r') as f:
        return json.load(f)


def analyze_predictions(data: dict):
    """Analyze prediction statistics"""
    predictions = data['predictions']
    
    print(f"\n{'='*60}")
    print(f"Ball Action Analysis: {data['UrlLocal']}")
    print(f"{'='*60}\n")
    
    # Basic stats
    print(f"Total predictions: {len(predictions)}")
    
    # Count by label
    label_counts = Counter(p['label'] for p in predictions)
    print(f"\nActions by type:")
    for label, count in label_counts.most_common():
        print(f"  {label:15} {count:5} ({count/len(predictions)*100:.1f}%)")
    
    # Count by half
    half_counts = Counter(p['half'] for p in predictions)
    print(f"\nActions by half:")
    for half, count in sorted(half_counts.items()):
        print(f"  Half {half}: {count} actions")
    
    # Confidence statistics
    confidences = [float(p['confidence']) for p in predictions]
    print(f"\nConfidence scores:")
    print(f"  Mean:   {sum(confidences)/len(confidences):.3f}")
    print(f"  Min:    {min(confidences):.3f}")
    print(f"  Max:    {max(confidences):.3f}")
    
    # High confidence actions (>0.9)
    high_conf = [p for p in predictions if float(p['confidence']) > 0.9]
    print(f"  >0.9:   {len(high_conf)} ({len(high_conf)/len(predictions)*100:.1f}%)")
    
    # Show first 10 predictions
    print(f"\nFirst 10 predictions:")
    print(f"{'Time':12} {'Label':12} {'Frame':8} {'Confidence':12}")
    print(f"{'-'*50}")
    for p in predictions[:10]:
        print(f"{p['gameTime']:12} {p['label']:12} {p['position']:8} {float(p['confidence']):12.3f}")
    
    # Show sample high-confidence predictions
    print(f"\nSample high-confidence predictions (>0.95):")
    print(f"{'Time':12} {'Label':12} {'Frame':8} {'Confidence':12}")
    print(f"{'-'*50}")
    very_high = [p for p in predictions if float(p['confidence']) > 0.95]
    for p in very_high[:10]:
        print(f"{p['gameTime']:12} {p['label']:12} {p['position']:8} {float(p['confidence']):12.3f}")


def main():
    # Example path - update this to your actual results path
    results_path = "/path/to/results_spotting.json"
    
    # Check if running on VM
    vm_path = "/workspace/ball-action-spotting/data/ball_action/predictions/sampling_weights_001/cv/fold_0/england_efl/2019-2020/2019-10-01 - Leeds United - West Bromwich/results_spotting.json"
    
    if Path(vm_path).exists():
        results_path = vm_path
    else:
        print(f"Please provide the path to results_spotting.json")
        print(f"\nOn VM, it should be at:")
        print(f"  {vm_path}")
        return
    
    # Load and analyze
    data = load_predictions(results_path)
    analyze_predictions(data)
    
    print(f"\n{'='*60}")
    print(f"Next steps:")
    print(f"{'='*60}")
    print(f"1. Get player tracking data for the same video")
    print(f"2. For each action at frame F:")
    print(f"   - Get all player bounding boxes at frame F")
    print(f"   - Find the player closest to the ball (or center of action)")
    print(f"   - Assign that player's jersey number to the action")
    print(f"3. Save enhanced results with jersey numbers")
    print(f"\n")


if __name__ == "__main__":
    main()
