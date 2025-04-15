import os
import pandas as pd

# Load Excel file
df = pd.read_excel("/mimer/NOBACKUP/groups/biomedicalimaging-kth/miladfa/mama_mia/filtered_clinical_and_imaging_info_subtype.xlsx")
patient_ids = df["patient_id"].astype(str).tolist()

# Directories
base_dir = "/mimer/NOBACKUP/groups/biomedicalimaging-kth/miladfa/mama_mia"
dirs = {
    0: os.path.join(base_dir, "cropped_0000"),
    1: os.path.join(base_dir, "cropped_0001"),
    2: os.path.join(base_dir, "cropped_0002")
}

# Check each patient
missing = []
for pid in patient_ids:
    for t in [0, 1, 2]:
        filename = f"{pid}_{str(t).zfill(4)}_cropped.nii.gz"
        if not os.path.exists(os.path.join(dirs[t], filename)):
            missing.append((pid, f"cropped_{str(t).zfill(4)}"))

# Report
if not missing:
    print("✅ All patients have all 3 cropped files.")
else:
    print(f"❌ {len(missing)} missing files:")
    for pid, timepoint in missing:
        print(f"Missing {timepoint} for {pid}")
