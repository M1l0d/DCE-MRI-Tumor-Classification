#!/bin/bash
#SBATCH --job-name=train_3dcnn
#SBATCH --output=Runs/train_3dcnn_%j.out  # Saves output logs
#SBATCH --error=Errors/train_3dcnn_%j.err   # Saves error logs
#SBATCH --time=2:00:00               # Adjust as needed (4 hours max training)
##SBATCH --partition=accelerated      # Use the GPU partition
#SBATCH --gpus-per-node=V100:1        # Request 1 V100 GPU
##SBATCH --cpus-per-task=8            # Request 8 CPU cores for data loading
##SBATCH --mem=32G                    # Request 32GB RAM
#SBATCH --account=NAISS2024-5-579     # Use the correct project account

# Load necessary modules 
module load Python/3.11.3-GCCcore-12.3.0
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1

# Activate virtual environment
source my_venv/bin/activate

# Run the Python training script
python 3d_cnn_multichannel_new.py
