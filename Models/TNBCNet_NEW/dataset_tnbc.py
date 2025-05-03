import os
import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
import pandas as pd
import torch.nn.functional as F
import random
from torchvision import transforms
import scipy.ndimage as ndimage

class RandomRotate3D:
    def __init__(self, max_angle=15):
        self.max_angle = max_angle

    def __call__(self, x):
        # Get tensor shape and convert to numpy for rotation
        device = x.device
        x_np = x.cpu().numpy()
        
        # Generate random angles
        angle_x = random.uniform(-self.max_angle, self.max_angle)
        angle_y = random.uniform(-self.max_angle, self.max_angle)
        angle_z = random.uniform(-self.max_angle, self.max_angle)
        
        # Apply rotations
        axes = [(1, 2), (0, 2), (0, 1)]  # Rotation planes
        angles = [angle_x, angle_y, angle_z]
        
        for i, (axis, angle) in enumerate(zip(axes, angles)):
            # For each channel
            for c in range(x_np.shape[0]):
                # scipy expects (z,y,x) order while PyTorch uses (c,z,y,x)
                x_np[c] = ndimage.rotate(x_np[c], angle, axes=axis, reshape=False, 
                                        order=1, mode='nearest')
        
        # Convert back to tensor
        return torch.from_numpy(x_np).to(device)

class RandomGamma:
    def __init__(self, gamma_range=(0.8, 1.2)):
        self.gamma_range = gamma_range
        
    def __call__(self, x):
        gamma = random.uniform(self.gamma_range[0], self.gamma_range[1])
        eps = 1e-8
        x_min = x.min()
        x_range = x.max() - x_min + eps
        
        # Normalize to [0, 1], apply gamma, then restore original range
        x_norm = (x - x_min) / x_range
        x_gamma = x_norm ** gamma
        return x_gamma * x_range + x_min

class TNBCDataset(Dataset):
    def __init__(self, patient_ids, labels, data_dir, mri_mode="fixed_crop_128_padded_matched", 
                 input_mode="delta2", use_segmentation=False, use_pe_map=False, 
                 use_ser_map=False, use_augmentation=False):
        self.patient_ids = patient_ids
        self.labels = labels
        self.data_dir = data_dir
        self.mri_mode = mri_mode
        self.input_mode = input_mode
        self.use_segmentation = use_segmentation
        self.use_pe_map = use_pe_map
        self.use_ser_map = use_ser_map
        self.use_augmentation = use_augmentation

        self.tp_dirs = [
            os.path.join(data_dir, f"{mri_mode}_0000"),
            os.path.join(data_dir, f"{mri_mode}_0001"),
            os.path.join(data_dir, f"{mri_mode}_0002"),
        ]
        self.segmentation_dir = os.path.join(data_dir, "segmentations", "expert")
        
        # Define augmentation pipeline
        if self.use_augmentation:
            self.augmentations = [
                lambda x: x.flip(dims=[random.randint(1, 3)]) if random.random() > 0.5 else x,  # Random flip
                RandomRotate3D(max_angle=15),  # Random rotation
                RandomGamma(gamma_range=(0.8, 1.2)),  # Random gamma correction
            ]
        
    def __len__(self):
        return len(self.patient_ids)
    
    def augment(self, volume):
        """Apply a series of augmentations to the 3D volume"""
        if not self.use_augmentation:
            return volume
            
        for aug_fn in self.augmentations:
            if random.random() > 0.5:  # 50% chance to apply each augmentation
                volume = aug_fn(volume)
                
        return volume

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        label = torch.tensor(float(self.labels[pid]), dtype=torch.float32)

        volumes = []
        for i, dir_path in enumerate(self.tp_dirs):
            nii_path = os.path.join(dir_path, f"{pid}_{str(i).zfill(4)}_cropped.nii.gz")
            vol = nib.load(nii_path).get_fdata()
            vol = (vol - vol.mean()) / (vol.std() + 1e-5)
            vol = torch.tensor(vol, dtype=torch.float32).unsqueeze(0)
            volumes.append(vol)

        t0, t1, t2 = volumes
        delta1 = t2 - t1
        delta2 = t1 - t0

        if self.input_mode == "t2":
            img = t2
        elif self.input_mode == "delta":
            img = torch.cat([t2, delta1], dim=0)
        elif self.input_mode == "delta2":
            img = torch.cat([t2, delta1, delta2], dim=0)
        elif self.input_mode == "t0t2":
            img = torch.cat([t0, t2], dim=0)
        else:
            raise ValueError(f"Invalid input_mode: {self.input_mode}")

        if self.use_segmentation:
            seg_path = os.path.join(self.segmentation_dir, f"{pid}.nii.gz")
            if os.path.exists(seg_path):
                seg = nib.load(seg_path).get_fdata()
                seg = (seg > 0).astype(np.float32)
                seg = F.interpolate(
                    torch.tensor(seg).unsqueeze(0).unsqueeze(0),
                    size=img.shape[1:],
                    mode="nearest"
                ).squeeze(0)
                img = torch.cat([img, seg], dim=0)
            else:
                empty_seg = torch.zeros((1, *img.shape[1:]))
                img = torch.cat([img, empty_seg], dim=0)
                
        # Apply augmentations (only to training data)
        if self.use_augmentation:
            img = self.augment(img)

        return img, label

def get_cross_validation_loaders(data_dir, batch_size, input_mode, mri_mode, label_column, 
                                use_segmentation, use_pe_map, use_ser_map, n_splits=5, 
                                use_augmentation=False):
    from sklearn.model_selection import KFold
    from torch.utils.data import DataLoader

    clinical_path = os.path.join(data_dir, "filtered_clinical_and_imaging_info_subtype.xlsx")
    df = pd.read_excel(clinical_path)
    df[label_column] = (df["tumor_subtype"].str.lower() == "triple_negative").astype(float)
    df = df.dropna(subset=[label_column])

    patient_ids = df["patient_id"].tolist()
    label_dict = dict(zip(df["patient_id"], df[label_column]))

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for train_idx, val_idx in kf.split(patient_ids):
        train_ids = [patient_ids[i] for i in train_idx]
        val_ids = [patient_ids[i] for i in val_idx]

        train_set = TNBCDataset(
            train_ids, label_dict, data_dir, mri_mode, input_mode, 
            use_segmentation, use_pe_map, use_ser_map, use_augmentation=use_augmentation
        )
        
        val_set = TNBCDataset(
            val_ids, label_dict, data_dir, mri_mode, input_mode, 
            use_segmentation, use_pe_map, use_ser_map, use_augmentation=False  # No augmentation for validation
        )

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)

        yield train_loader, val_loader