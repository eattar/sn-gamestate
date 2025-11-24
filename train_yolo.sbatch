#!/bin/bash
#SBATCH --job-name=yolo-finetune
#SBATCH --output=/netscratch/eattar/logs/yolo_training_%j.log
#SBATCH --error=/netscratch/eattar/logs/yolo_training_%j.err
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Print job info
echo "=================================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
echo "Start time: $(date)"
echo "=================================================="

# Create log directory if it doesn't exist
mkdir -p /netscratch/eattar/logs
mkdir -p /netscratch/eattar/yolo-weights

# Navigate to project directory
cd /workspace/sn-gamestate || cd ~/sn-gamestate

# Pull latest code from git
echo "Pulling latest code from git..."
git fetch origin
git checkout ball-action-integration
git pull origin ball-action-integration

# Activate conda environment
echo "Activating conda environment..."
source /root/miniconda/etc/profile.d/conda.sh
conda activate ball-action-spotting

# Verify environment
echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"

# Set PyTorch memory allocation config to avoid fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Print GPU info
echo "=================================================="
nvidia-smi
echo "=================================================="

# Start training
echo "Starting YOLO training..."
python finetune_yolo.py \
  --dataset-dir /netscratch/eattar/ds/YOLO \
  --base-model yolov8x.pt \
  --epochs 50 \
  --imgsz 1280 \
  --batch 16 \
  --output-name soccer_ball_yolo

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo "Training completed successfully!"
    
    # Copy weights to netscratch
    echo "Copying weights to /netscratch/eattar/yolo-weights..."
    cp -r runs/train/soccer_ball_yolo/weights /netscratch/eattar/yolo-weights/soccer_ball_yolo_$(date +%Y%m%d_%H%M%S)
    
    # Also copy the entire results folder for reference
    cp -r runs/train/soccer_ball_yolo /netscratch/eattar/yolo-weights/soccer_ball_yolo_full_$(date +%Y%m%d_%H%M%S)
    
    echo "Weights saved to:"
    ls -lh /netscratch/eattar/yolo-weights/
    
    echo "Best model: /netscratch/eattar/yolo-weights/soccer_ball_yolo_*/best.pt"
else
    echo "Training failed with exit code $?"
fi

echo "=================================================="
echo "End time: $(date)"
echo "=================================================="
