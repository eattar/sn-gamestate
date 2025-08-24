#!/usr/bin/env python3
"""
Simple script to switch between unified backbone and separate modules.
This allows you to easily test the unified backbone without permanently changing your configuration.
"""

import os
import shutil
from pathlib import Path

def backup_original_config():
    """Backup the original soccernet.yaml configuration."""
    config_path = Path("sn_gamestate/configs/soccernet.yaml")
    backup_path = Path("sn_gamestate/configs/soccernet.yaml.backup")
    
    if config_path.exists() and not backup_path.exists():
        shutil.copy2(config_path, backup_path)
        print(f"✅ Backed up original config to: {backup_path}")
    elif backup_path.exists():
        print(f"ℹ️  Backup already exists at: {backup_path}")
    else:
        print(f"❌ Original config not found at: {config_path}")
        return False
    
    return True

def restore_original_config():
    """Restore the original soccernet.yaml configuration."""
    config_path = Path("sn_gamestate/configs/soccernet.yaml")
    backup_path = Path("sn_gamestate/configs/soccernet.yaml.backup")
    
    if backup_path.exists():
        shutil.copy2(backup_path, config_path)
        print(f"✅ Restored original config from: {backup_path}")
        return True
    else:
        print(f"❌ Backup not found at: {backup_path}")
        return False

def switch_to_unified_backbone():
    """Switch to unified backbone configuration."""
    unified_config = Path("sn_gamestate/configs/soccernet_unified.yaml")
    target_config = Path("sn_gamestate/configs/soccernet.yaml")
    
    if unified_config.exists():
        shutil.copy2(unified_config, target_config)
        print(f"✅ Switched to unified backbone configuration")
        print(f"   You can now run: uv run tracklab -cn soccernet")
        return True
    else:
        print(f"❌ Unified backbone config not found at: {unified_config}")
        return False

def switch_to_separate_modules():
    """Switch back to separate modules configuration."""
    separate_config = Path("sn_gamestate/configs/soccernet_separate.yaml")
    target_config = Path("sn_gamestate/configs/soccernet.yaml")
    
    if separate_config.exists():
        shutil.copy2(separate_config, target_config)
        print(f"✅ Switched to separate modules configuration")
        print(f"   You can now run: uv run tracklab -cn soccernet")
        return True
    else:
        print(f"❌ Separate modules config not found at: {separate_config}")
        return False

def create_separate_modules_config():
    """Create a configuration file for separate modules (original approach)."""
    separate_config = Path("sn_gamestate/configs/soccernet_separate.yaml")
    
    if not separate_config.exists():
        config_content = """# TrackLab SoccerNet config with Separate Modules (Original Approach)

# The defaults list contains the files that will be used
# to create the final config file. This item *must* be
# the first element in the file.
defaults:
  - dataset: soccernet_gs
  - eval: gs_hota
  - engine: offline
  - visualization: gamestate
  - modules/bbox_detector: yolo_ultralytics
  - modules/reid: prtreid
  - modules/track: bpbreid_strong_sort
  - modules/jersey_number_detect: mmocr
  - modules/team: kmeans_embeddings
  - modules/team_side: mean_position
  - modules/tracklet_agg: voting_role_jn
  - modules/pitch: nbjw_calib
  - modules/calibration: nbjw_calib
  - _self_

# Pipeline definition with separate modules:
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

# Experiment name
experiment_name: "sn-gamestate-separate"

# Path definitions
home_dir: "${oc.env:HOME}"
data_dir: "/netscratch/eattar/ds/SoccerNet/2024/data"
model_dir: "${project_dir}/pretrained_models"

# Machine configuration
num_cores: 1
use_wandb: True
use_rich: True

modules:
  bbox_detector: {batch_size: 8}
  pose_bottomup: {batch_size: 8}
  reid: {batch_size: 64}
  track: {batch_size: 64}
  pitch: {batch_size: 1}
  calibration: {batch_size: 1}
  jersey_number_detect: {batch_size: 8}

# Flags
test_tracking: True
eval_tracking: True
print_config: True

# Dataset
dataset:
  nvid: 1
  eval_set: "valid"
  dataset_path: ${data_dir}/SoccerNetGS
  vids_dict:
    valid: []

# Tracker state
state:
  save_file: "states/${experiment_name}.pklz"
  load_file: null

# Visualization
visualization:
  cfg:
    save_videos: True

# Hydra configuration
project_dir: "${hydra:runtime.cwd}"
hydra:
  output_subdir: "configs"
  job:
    chdir: True
  run:
    dir: "/netscratch/eattar/SoccerNet/outputs/${experiment_name}/${now:%Y-%m-%d}/${now:%H-%M-%S}"
  sweep:
    dir: "multirun_outputs/${experiment_name}/${now:%Y-%m-%d}/${now:%H-%M-%S}"
"""
        
        with open(separate_config, 'w') as f:
            f.write(config_content)
        
        print(f"✅ Created separate modules config at: {separate_config}")

def show_current_config():
    """Show which configuration is currently active."""
    config_path = Path("sn_gamestate/configs/soccernet.yaml")
    
    if not config_path.exists():
        print("❌ No active configuration found")
        return
    
    with open(config_path, 'r') as f:
        content = f.read()
    
    if "unified_backbone" in content:
        print("🔄 Currently using: UNIFIED BACKBONE configuration")
        print("   This replaces bbox_detector + pitch + calibration with a single module")
    elif "bbox_detector" in content and "pitch" in content and "calibration" in content:
        print("🔀 Currently using: SEPARATE MODULES configuration")
        print("   This uses bbox_detector, pitch, and calibration as separate modules")
    else:
        print("❓ Unknown configuration type")

def main():
    """Main function to handle user input."""
    print("🔄 SoccerNet Configuration Switcher")
    print("=====================================")
    print()
    
    # Create separate modules config if it doesn't exist
    create_separate_modules_config()
    
    while True:
        print("\nCurrent status:")
        show_current_config()
        print()
        
        print("Options:")
        print("1. Switch to Unified Backbone (recommended)")
        print("2. Switch to Separate Modules (original)")
        print("3. Backup current configuration")
        print("4. Restore original configuration")
        print("5. Exit")
        print()
        
        choice = input("Enter your choice (1-5): ").strip()
        
        if choice == "1":
            if backup_original_config():
                if switch_to_unified_backbone():
                    print("\n🎉 Successfully switched to Unified Backbone!")
                    print("   You can now run: uv run tracklab -cn soccernet")
                    print("\nWould you like to exit? (y/n): ", end="")
                    exit_choice = input().strip().lower()
                    if exit_choice in ['y', 'yes']:
                        print("👋 Exiting. Happy coding!")
                        break
        elif choice == "2":
            if backup_original_config():
                if switch_to_separate_modules():
                    print("\n🎉 Successfully switched to Separate Modules!")
                    print("   You can now run: uv run tracklab -cn soccernet")
                    print("\nWould you like to exit? (y/n): ", end="")
                    exit_choice = input().strip().lower()
                    if exit_choice in ['y', 'yes']:
                        print("👋 Exiting. Happy coding!")
                        break
        elif choice == "3":
            backup_original_config()
        elif choice == "4":
            restore_original_config()
        elif choice == "5":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1-5.")
        
        print()

if __name__ == "__main__":
    main()
