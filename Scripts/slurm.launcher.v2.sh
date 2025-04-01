#!/bin/bash
#SBATCH --job-name=deepyV2
#SBATCH --output=slurm_logs/deepyV2_seed_%A_%a.out
#SBATCH --error=Errors/deepyV2_seed_%A_%a.err
#SBATCH --time=2:00:00               
#SBATCH --gpus-per-node=V100:1        # Request 1 V100 GPU
#SBATCH --account=NAISS2024-5-579     # Use the correct project account
#SBATCH --array=0-4                  # Array job for different seeds

# Load necessary modules 
module load Python/3.11.3-GCCcore-12.3.0
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1

# Activate virtual environment
source my_venv/bin/activate

# SEEDS
SEEDS=(42 1337 7 123 2025)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

# Run the Python training script
python Models/DeepyNetV2/run_v2.py \
 --input_mode delta \
 --batch_size 32 \
 --epochs 100 \
 --lr 5e-4 \
 --seed $SEED