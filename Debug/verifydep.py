import torch
import monai
import SimpleITK as sitk
import nrrd
import numpy as np

print(f"PyTorch version: {torch.__version__}")
print(f"MONAI version: {monai.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
