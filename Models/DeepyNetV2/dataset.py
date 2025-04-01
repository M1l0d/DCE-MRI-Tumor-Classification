import os
import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
import pandas as pd
import torchio as tio
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class MRIDataset(Dataset):
    def __init__(self, timepoint_dirs, labels_df, augment=False, input_mode="delta", clinical_cols=None):
        self.data_dirs = timepoint_dirs
        self.labels_df = labels_df
        self.input_mode = input_mode
        self.augment = augment
        self.clinical_cols = clinical_cols or []
        self.num_clinical_features = len(self.clinical_cols)

        self.patients = labels_df["patient_id"].tolist()
        self.label_map = dict(zip(labels_df["patient_id"], labels_df["pcr"]))

        self.transforms = tio.Compose([
            tio.RandomAffine(scales=0.05, degrees=5, translation=3, p=0.2),
            tio.RandomFlip(axes=(0, 1, 2), flip_probability=0.1),
            tio.RandomNoise(mean=0, std=0.005, p=0.1)
        ])

        print(f"✅ Loaded {len(self.patients)} samples with {len(self.clinical_cols)} clinical features.")

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        pid = self.patients[idx]
        label = float(self.label_map.get(pid, 0.0))

        # --- Load MRI Volumes ---
        vols = []
        for i, path in enumerate(self.data_dirs):
            nii_path = os.path.join(path, f"{pid}_{str(i).zfill(4)}_cropped.nii.gz")
            vol = nib.load(nii_path).get_fdata()
            vol = np.clip(vol, *np.percentile(vol, (2, 98)))
            vol = (vol - vol.mean()) / (vol.std() + 1e-5)
            vol_tensor = torch.tensor(vol, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            vol_tensor = F.interpolate(vol_tensor, size=(64, 64, 64), mode="trilinear", align_corners=False)
            vols.append(vol_tensor.squeeze())

        t0, t1, t2 = vols
        delta1 = t2 - t1
        delta2 = t1 - t0

        if self.input_mode == "t2":
            img = t2.unsqueeze(0)
        elif self.input_mode == "delta":
            img = torch.stack([t2, delta1], dim=0)
        elif self.input_mode == "delta2":
            img = torch.stack([t2, delta1, delta2], dim=0)
        else:
            raise ValueError(f"Invalid input_mode: {self.input_mode}")

        if self.augment:
            img = self.transforms(tio.Image(tensor=img, type=tio.INTENSITY)).data

        # --- Load Clinical Features ---
        clinical = self.labels_df.loc[self.labels_df["patient_id"] == pid, self.clinical_cols].values.astype(np.float32)
        clinical_tensor = torch.tensor(clinical).squeeze()

        return img, clinical_tensor, torch.tensor(label, dtype=torch.float32)
    
def load_clinical_metadata(excel_path):
    df = pd.read_excel(excel_path)
    df = df.dropna(subset=["pcr"])  # keep only labeled samples

    # --- CATEGORICAL ENCODING ---
    # Tumor Subtype (one-hot)
    df["tumor_subtype"] = df["tumor_subtype"].fillna("unknown")
    tumor_subtypes = ["luminal", "luminal_a", "luminal_b", "her2_enriched", "her2_pure", "triple_negative"]
    for subtype in tumor_subtypes:
        df[f"subtype_{subtype}"] = (df["tumor_subtype"] == subtype).astype(int)

    # Nottingham Grade (ordinal)
    grade_map = {"low": 0, "intermediate": 1, "high": 2}
    df["nottingham_grade"] = df["nottingham_grade"].map(grade_map).fillna(-1)

    # BMI Group (ordinal)
    bmi_map = {
        "underweight": 0, "normal": 1, "overweight": 2,
        "obesity_class1": 3, "obesity_class2": 4, "obesity_class3": 5
    }
    df["bmi_group"] = df["bmi_group"].map(bmi_map).fillna(-1)

    # Ethnicity (grouped)
    ethnicity_map = {
        "caucasian": 0,
        "african american": 1,
        "asian": 2,
        "hispanic": 3, "native american": 3, "multiple race": 3,
        "american indian": 3, "pacific islander": 3,
        "american indian/alaskan native": 3,
        "hawaiian": 3, "hawaiian/pacific islander": 3
    }
    df["ethnicity"] = df["ethnicity"].str.lower().map(ethnicity_map).fillna(3)

    # Mammaprint (binary)
    df["mammaprint"] = df["mammaprint"].fillna(df["mammaprint"].mode()[0])

    # --- NUMERIC CONTINUOUS VARIABLES ---
    all_numeric_candidates = ["age", "oncotype", "weight", "patient_size", "her2", "hr", "er", "pr"]
    numeric_cols = [col for col in all_numeric_candidates if col in df.columns]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    # --- COMBINE FEATURES ---
    final_features = (
        numeric_cols +
        [f"subtype_{s}" for s in tumor_subtypes] +
        ["nottingham_grade", "bmi_group", "ethnicity", "mammaprint"]
    )

    # --- NORMALIZE ALL FINAL FEATURES ---
    scaler = StandardScaler()
    df[final_features] = scaler.fit_transform(df[final_features])

    # --- RETURN DICTS ---
    labels_dict = dict(zip(df["patient_id"], df["pcr"]))
    clinical_dict = {
        pid: df.loc[df["patient_id"] == pid, final_features].values.squeeze().astype(np.float32)
        for pid in df["patient_id"]
    }

    return df, labels_dict, clinical_dict, final_features

def collect_image_paths(timepoint_dirs, valid_ids):
    image_paths = {}
    for pid in valid_ids:
        paths = []
        for i, d in enumerate(timepoint_dirs):
            fname = f"{pid}_{str(i).zfill(4)}_cropped.nii.gz"
            paths.append(os.path.join(d, fname))
        image_paths[pid] = paths
    return image_paths

