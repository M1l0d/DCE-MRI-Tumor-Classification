import os
import shutil
import pandas as pd

# === Paths ===
mini_clinical_path = "/mimer/NOBACKUP/groups/biomedicalimaging-kth/miladfa/mama_mia/mini_dummy_dataset/mini_100_clinical_info.xlsx"
src_root = "/mimer/NOBACKUP/groups/biomedicalimaging-kth/miladfa/mama_mia"
dst_root = "mini_dummy_dataset"

# === Load patient IDs ===
df = pd.read_excel(mini_clinical_path)
patient_ids = df["patient_id"].astype(str).tolist()

# === Timepoint folders ===
timepoints = {
    "cropped_0000": "0000",
    "cropped_0001": "0001",
    "cropped_0002": "0002",
}

# === Copy DCE-MRI volumes ===
for folder, suffix in timepoints.items():
    src_dir = os.path.join(src_root, folder)
    dst_dir = os.path.join(dst_root, folder)
    os.makedirs(dst_dir, exist_ok=True)

    for pid in patient_ids:
        filename = f"{pid}_{suffix}_cropped.nii.gz"
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dst_dir, filename)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
        else:
            print(f"❌ Missing file: {src_path}")

# === Optional: Copy segmentations ===
seg_src_dir = os.path.join(src_root, "segmentations", "expert")
seg_dst_dir = os.path.join(dst_root, "segmentations", "expert")
os.makedirs(seg_dst_dir, exist_ok=True)

for pid in patient_ids:
    seg_filename = f"{pid}.nii.gz"
    src_path = os.path.join(seg_src_dir, seg_filename)
    dst_path = os.path.join(seg_dst_dir, seg_filename)
    if os.path.exists(src_path):
        shutil.copy2(src_path, dst_path)
    else:
        print(f"⚠️ Missing segmentation for {pid}")
