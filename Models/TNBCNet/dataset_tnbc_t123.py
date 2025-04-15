
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
        use_segmentation=False,
        segmentation_dir=None,
        augment=False,
    ):
        self.patient_ids = patient_ids
        self.labels = labels
        self.timepoint_dirs = timepoint_dirs
        self.use_segmentation = use_segmentation
        self.segmentation_dir = segmentation_dir
        self.augment = augment

        self.transform = tio.Compose([
            tio.RandomAffine(scales=0.05, degrees=5, translation=3, p=0.2),
            tio.RandomFlip(axes=(0, 1, 2), flip_probability=0.1),
            tio.RandomNoise(mean=0, std=0.005, p=0.1)
        ])

        print(f"✅ Initialized TNBCDataset with {len(self.patient_ids)} patients | mode=t123 | segmentation={use_segmentation}")

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        label = torch.tensor(float(self.labels[pid]), dtype=torch.float32)

        volumes = []
        for i, tp_dir in enumerate(self.timepoint_dirs):
            nii_path = os.path.join(tp_dir, f"{pid}_{str(i).zfill(4)}_cropped.nii.gz")
            vol = nib.load(nii_path).get_fdata()
            vol = np.clip(vol, *np.percentile(vol, (2, 98)))
            vol = zscore(vol)
            vol = torch.tensor(vol, dtype=torch.float32).unsqueeze(0)
            volumes.append(vol)

        t0, t1, t2 = volumes
        delta1 = t2 - t1
        # img = torch.cat(volumes, dim=0)  # t0 + t1 + t2
        img = torch.cat([t0, t1, t2, delta1], dim=0)

        if self.use_segmentation:
            mask_path = os.path.join(self.segmentation_dir, f"{pid}.nii.gz")
            if os.path.exists(mask_path):
                seg = nib.load(mask_path).get_fdata()
                seg = torch.tensor((seg > 0).astype(np.float32))
                seg = F.interpolate(seg.unsqueeze(0).unsqueeze(0), size=img.shape[1:], mode="nearest").squeeze(0)
            else:
                seg = torch.zeros_like(volumes[0])
                print(f"⚠️ Missing segmentation for {pid}, using empty mask.")
            img = torch.cat([img, seg], dim=0)

        if self.augment:
            img = self.transform(tio.ScalarImage(tensor=img)).data

        img = F.interpolate(img.unsqueeze(0), size=(64, 64, 64), mode="trilinear", align_corners=False).squeeze(0)

        return img, label

def get_dataloaders(data_dir, batch_size, val_split=0.2, use_segmentation=False):
    clinical_path = os.path.join(data_dir, "filtered_clinical_and_imaging_info_subtype.xlsx")
    seg_dir = os.path.join(data_dir, "segmentations", "expert")

    timepoint_dirs = [
        os.path.join(data_dir, "cropped_0000"),
        os.path.join(data_dir, "cropped_0001"),
        os.path.join(data_dir, "cropped_0002")
    ]

    df = pd.read_excel(clinical_path)
    df["tnbc"] = (df["tumor_subtype"].str.lower() == "triple_negative").astype(float)
    df = df.dropna(subset=["tnbc"])

    patient_ids = df["patient_id"].tolist()
    label_dict = dict(zip(df["patient_id"], df["tnbc"]))

    train_ids, val_ids = train_test_split(patient_ids, test_size=val_split, random_state=42)

    train_set = TNBCDataset(train_ids, label_dict, timepoint_dirs, use_segmentation, seg_dir, augment=True)
    val_set = TNBCDataset(val_ids, label_dict, timepoint_dirs, use_segmentation, seg_dir, augment=False)

    in_channels = 4 + int(use_segmentation)

    train_labels = torch.tensor([label_dict[pid] for pid in train_ids])
    class_sample_counts = torch.tensor([(train_labels == 0).sum(), (train_labels == 1).sum()], dtype=torch.float)
    weights = 1. / class_sample_counts
    sample_weights = weights[train_labels.long()]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)

    n_pos = (train_labels == 1.0).sum().item()
    n_neg = (train_labels == 0.0).sum().item()
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)
    print(f"⚖️ Class counts in train set: pos={n_pos}, neg={n_neg} | pos_weight={pos_weight.item():.4f}")

    return (train_loader, val_loader), in_channels, 0, pos_weight
