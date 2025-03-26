#!/bin/bash

# to run: source load_env.sh

# Load necessary modules (replace with the actual modules from `module list`)
module load Python/3.11.3-GCCcore-12.3.0
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
module load scikit-learn/1.3.1-gfbf-2023a
module load matplotlib/3.7.2-gfbf-2023a
# Add more modules as needed...

# Activate virtual environment
source my_venv/bin/activate

echo "Modules loaded and virtual environment activated."
