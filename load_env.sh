#!/bin/bash

# to run: source load_env.sh

# Load necessary modules (replace with the actual modules from `module list`)
module load Python/3.11.3-GCCcore-12.3.0
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
# Add more modules as needed...

# Activate virtual environment
source my_venv/bin/activate

echo "Modules loaded and virtual environment activated."
