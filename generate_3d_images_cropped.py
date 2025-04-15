import os
import torch
import numpy as np
import nrrd
import SimpleITK as sitk

from monai.data import DataLoader, ImageDataset
from monai.networks.nets import DenseNet121, DenseNet169, DenseNet201
from monai.utils import set_determinism
from torchvision.models import vgg19, VGG19_Weights, densenet121

from monai.transforms import (
    EnsureChannelFirst,
    Compose,
    RandRotate90,
    Resize,
    ScaleIntensity,
    SpatialPad,
    RandFlip,
    RandZoom,
    AsDiscrete,
)

import argparse
import utils_BTC
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score
import json
import torch.nn as nn
import sys
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score


parser = argparse.ArgumentParser("HBC25", add_help=False)
parser.add_argument("--data_dir", default="../Dataset/DATA_Test", type=str)
parser.add_argument("--mask_dir", default="../Dataset/MASK_Test", type=str)
args = parser.parse_args()

# Get 3D images
data_dir = args.data_dir
mask_dir = args.mask_dir
data_paths = [
    os.path.join(data_dir, f)
    for f in sorted(os.listdir(data_dir))
    if f.endswith(".nrrd")
]
mask_paths = [
    os.path.join(mask_dir, f)
    for f in sorted(os.listdir(mask_dir))
    if f.endswith(".nrrd")
]

for i in range(len(data_paths)):
    idx = os.path.basename(data_paths[i]).split("_")[1].split(".")[0]
    # print(idx)
    data, head1 = nrrd.read(data_paths[i])
    mask, head2 = nrrd.read(mask_paths[i])
    image3d = data * mask

    # Get mask area
    mask_range_x = [float("inf"), -1]
    for i in range(image3d.shape[0]):
        slice_mask_x = mask.take(indices=i, axis=0)
        mask_area_x = np.sum(slice_mask_x)
        if mask_area_x > 0:
            mask_range_x[0] = min(mask_range_x[0], i)
            mask_range_x[1] = max(mask_range_x[1], i)

    print(mask_range_x)

    mask_range_y = [float("inf"), -1]
    for j in range(image3d.shape[1]):
        slice_mask_y = mask.take(indices=j, axis=1)
        mask_area_y = np.sum(slice_mask_y)
        if mask_area_y > 0:
            mask_range_y[0] = min(mask_range_y[0], j)
            mask_range_y[1] = max(mask_range_y[1], j)

    print(mask_range_y)

    mask_range_z = [float("inf"), -1]
    for k in range(image3d.shape[2]):
        slice_mask_z = mask.take(indices=k, axis=2)
        mask_area_z = np.sum(slice_mask_z)
        if mask_area_z > 0:
            mask_range_z[0] = min(mask_range_z[0], k)
            mask_range_z[1] = max(mask_range_z[1], k)

    print(mask_range_z)

    image3d = image3d[
        mask_range_x[0] : mask_range_x[1] + 1,
        mask_range_y[0] : mask_range_y[1] + 1,
        mask_range_z[0] : mask_range_z[1] + 1,
    ]
    print(f"Image shape: {image3d.shape}")
    image3d = sitk.GetImageFromArray(image3d)
    save_path = f"../Dataset/Image3d_Cropped_Train_Valid/IMAGE{idx}_0.nii.gz"
    sitk.WriteImage(image3d, save_path)
    print(idx + " is processed.")
