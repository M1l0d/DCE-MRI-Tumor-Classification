#!/bin/bash
#SBATCH --job-name=TNBCNet
#SBATCH --output=slurm_logs_tnbc/TNBCNet_seed_%A_%a.out
#SBATCH --error=Errors/TNBCNet_seed_%A_%a.err
#SBATCH --time=12:00:00               
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
python Models/TNBCNet/run_tnbc.py \
 --input_mode image \
 --mri_mode delta2 \
 --epochs 100 \
 --lr 1e-7 \
 --weight_decay 5e-3 \
 --batch_size 32 \
 --use_segmentation_channel \
 --seed 2025
#  --use_pe_map \
#  --seed $SEED
#  --use_simple_model \
#  --backbone resnet18 \
#  --debug \