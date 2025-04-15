import os
import torch
import numpy as np
import nibabel as nib
import pandas as pd
import torch.nn.functional as F
import torchio as tio
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split


def zscore(volume):
    return (volume - volume.mean()) / (volume.std() + 1e-8)


class TNBCDataset(Dataset):
    def __init__(
        self,
        patient_ids,
        labels,
        timepoint_dirs,
        input_mode="delta",
        use_segmentation=False,
        segmentation_dir=None,
        augment=False,
        use_pe_map=False,
    ):
        self.patient_ids = patient_ids
        self.labels = labels
        self.timepoint_dirs = timepoint_dirs
        self.input_mode = input_mode
        self.use_segmentation = use_segmentation
        self.segmentation_dir = segmentation_dir
        self.augment = augment
        self.use_pe_map = use_pe_map

        self.transform = tio.Compose([
            tio.RandomAffine(scales=0.1, degrees=10, translation=5, p=0.5),
            tio.RandomFlip(axes=(0, 1, 2), flip_probability=0.3),
            tio.RandomNoise(mean=0, std=0.01, p=0.3)
        ])

        print(f"✅ Initialized TNBCDataset with {len(self.patient_ids)} patients | mode={input_mode} | segmentation={use_segmentation}")

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        label = torch.tensor(float(self.labels[pid]), dtype=torch.float32)

        # Load t0, t1, t2 volumes
        volumes = []
        for i, tp_dir in enumerate(self.timepoint_dirs):
            nii_path = os.path.join(tp_dir, f"{pid}_{str(i).zfill(4)}_cropped.nii.gz")
            vol = nib.load(nii_path).get_fdata()
            vol = np.clip(vol, *np.percentile(vol, (2, 98)))
            vol = (vol - vol.mean()) / (vol.std() + 1e-5)
            vol = torch.tensor(vol, dtype=torch.float32).unsqueeze(0)
            volumes.append(vol)

        t0, t1, t2 = volumes
        delta1 = t2 - t1
        delta2 = t1 - t0

        img = None
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

        # Add PE map: (t1 - t0) / t0
        if self.use_pe_map:
            # Use simple tanh normalization instead of log
            pe = torch.tanh((t1 - t0) / (torch.abs(t0) + 1e-5))
            # Apply standard normalization
            pe = (pe - pe.mean()) / (pe.std() + 1e-5)
            img = torch.cat([img, pe], dim=0)

        # Add segmentation mask
        if self.use_segmentation:
            mask_path = os.path.join(self.segmentation_dir, f"{pid}.nii.gz")
            if os.path.exists(mask_path):
                seg = nib.load(mask_path).get_fdata()
                seg = torch.tensor((seg > 0).astype(np.float32))
                seg = F.interpolate(seg.unsqueeze(0).unsqueeze(0), size=img.shape[1:], mode="nearest").squeeze(0)
            else:
                seg = torch.zeros_like(t2[:1])
                print(f"⚠️ Missing segmentation for {pid}, using empty mask.")
            img = torch.cat([img, seg], dim=0)

        if self.augment:
            img = self.transform(tio.ScalarImage(tensor=img)).data

        img = F.interpolate(img.unsqueeze(0), size=(64, 64, 64), mode="trilinear", align_corners=False).squeeze(0)

        return img, label


def get_dataloaders(data_dir, batch_size, input_mode, mri_mode, label_column="tnbc", val_split=0.2, use_segmentation=False, use_pe_map=False):
    # clinical_path = os.path.join(data_dir, "filtered_clinical_and_imaging_info_subtype.xlsx")
    clinical_path = os.path.join(data_dir, "filtered_clinical_and_imaging_info_subtype.xlsx")
    seg_dir = os.path.join(data_dir, "segmentations", "expert")

    # timepoint dirs
    timepoint_dirs = [
        os.path.join(data_dir, "cropped_0000"),
        os.path.join(data_dir, "cropped_0001"),
        os.path.join(data_dir, "cropped_0002")
    ]

    df = pd.read_excel(clinical_path)
    df[label_column] = (df["tumor_subtype"].str.lower() == "triple_negative").astype(float)
    df = df.dropna(subset=[label_column])

    patient_ids = df["patient_id"].tolist()
    label_dict = dict(zip(df["patient_id"], df[label_column]))

    # Split
    train_ids, val_ids = train_test_split(patient_ids, test_size=val_split, random_state=42)

    # Datasets
    train_set = TNBCDataset(train_ids, label_dict, timepoint_dirs, mri_mode, use_segmentation, seg_dir, augment=True, use_pe_map=use_pe_map)
    val_set = TNBCDataset(val_ids, label_dict, timepoint_dirs, mri_mode, use_segmentation, seg_dir, augment=False, use_pe_map=use_pe_map)

    if mri_mode == "delta2":
        num_base_channels = 3
    elif mri_mode == "delta":
        num_base_channels = 2
    elif mri_mode == "t0t2":
        num_base_channels = 2
    elif mri_mode == "t2":
        num_base_channels = 1
    else:
        raise ValueError(f"Unsupported mri_mode: {mri_mode}")
    
    in_channels = num_base_channels + int(use_pe_map) + int(use_segmentation)

    # Compute weights
    label_tensor = torch.tensor([label_dict[pid] for pid in train_ids], dtype=torch.long)
    class_sample_counts = torch.tensor([(label_tensor == 0).sum(), (label_tensor == 1).sum()], dtype=torch.float)
    weights = 1. / class_sample_counts
    sample_weights = weights[label_tensor]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    # Use sampler instead of shuffle=True
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, num_workers=4)

    #train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)

    # Compute class imbalance for pos_weight (use in BCEWithLogitsLoss)
    n_pos = sum(label_dict[pid] == 1.0 for pid in train_ids)
    n_neg = sum(label_dict[pid] == 0.0 for pid in train_ids)
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)
    print(f"⚖️ Class counts in train set: pos={n_pos}, neg={n_neg} | pos_weight={pos_weight.item():.4f}")

    return (train_loader, val_loader), in_channels, 0, pos_weight  # tabular_dim=0 for image-only

import matplotlib.pyplot as plt
def visualize_sample(dataset, idx):
    img, label = dataset[idx]
    print(f"Label: {label.item()}")
    for i in range(img.shape[0]):  # Iterate over channels
        plt.figure()
        plt.title(f"Channel {i}")
        plt.imshow(img[i, :, :, img.shape[3] // 2].numpy(), cmap="gray")
        plt.show()


from sklearn.model_selection import KFold
def get_cross_validation_loaders(data_dir, batch_size, input_mode, mri_mode, label_column, use_segmentation, use_pe_map, n_splits=5):
    """Generate cross-validation loaders."""
    clinical_path = os.path.join(data_dir, "filtered_clinical_and_imaging_info_subtype.xlsx")
    seg_dir = os.path.join(data_dir, "segmentations", "expert")
    timepoint_dirs = [
        os.path.join(data_dir, "cropped_0000"),
        os.path.join(data_dir, "cropped_0001"),
        os.path.join(data_dir, "cropped_0002")
    ]

    df = pd.read_excel(clinical_path)
    df[label_column] = (df["tumor_subtype"].str.lower() == "triple_negative").astype(float)
    df = df.dropna(subset=[label_column])

    patient_ids = df["patient_id"].tolist()
    label_dict = dict(zip(df["patient_id"], df[label_column]))

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_idx, val_idx in kf.split(patient_ids):
        train_ids = [patient_ids[i] for i in train_idx]
        val_ids = [patient_ids[i] for i in val_idx]

        train_set = TNBCDataset(train_ids, label_dict, timepoint_dirs, mri_mode, use_segmentation, seg_dir, augment=True, use_pe_map=use_pe_map)
        val_set = TNBCDataset(val_ids, label_dict, timepoint_dirs, mri_mode, use_segmentation, seg_dir, augment=False, use_pe_map=use_pe_map)

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)

        yield train_loader, val_loader