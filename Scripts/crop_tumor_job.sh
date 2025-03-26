#!/bin/bash
#SBATCH --job-name=crop_tumor
#SBATCH --output=crop_tumor_%j.out  # Saves output logs
#SBATCH --error=crop_tumor_%j.err   # Saves error logs
#SBATCH --time=2:00:00              # Set a max runtime (adjust as needed)
##SBATCH --partition=cpu             # Use the CPU partition
#SBATCH --ntasks=1                  # Run a single task
#SBATCH --cpus-per-task=4           # Use 4 CPU cores (adjust as needed)
#SBATCH --account=NAISS2024-5-579   # Use the NAISS2024-5-579 project
#SBATCH -C NOGPU                    # Do not use GPUs

# Load necessary modules 
module load Python/3.11.3-GCCcore-12.3.0
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
# Add more modules as needed...

# Activate virtual environment
source my_venv/bin/activate

# Run your Python script
python crop_tumor.py
