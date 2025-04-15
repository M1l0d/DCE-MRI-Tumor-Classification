import os
import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path
from zipfile import ZipFile

# Settings
output_dir = Path("mini_dummy_dataset")
output_dir.mkdir(exist_ok=True)
shape = (64, 64, 64)
timepoints = [0, 1, 2]
patient_ids = [f"DUKE_{i:03d}" for i in range(1, 41)]

# Create folders
for tp in timepoints:
    (output_dir / f"cropped_000{tp}").mkdir(parents=True, exist_ok=True)
(output_dir / "segmentations" / "expert").mkdir(parents=True, exist_ok=True)

# Create clinical Excel file
labels = np.random.choice([0, 1], size=len(patient_ids), p=[0.67, 0.33])
subtypes = ['triple_negative' if y == 1 else 'luminal' for y in labels]
df = pd.DataFrame({'patient_id': patient_ids, 'tumor_subtype': subtypes})
df.to_excel(output_dir / "mini_clinical_info.xlsx", index=False)

# Create NIfTI volumes and segmentations
for pid in patient_ids:
    seg = (np.random.rand(*shape) > 0.95).astype(np.float32)
    nib.save(nib.Nifti1Image(seg, affine=np.eye(4)), output_dir / "segmentations" / "expert" / f"{pid}.nii.gz")

    for tp in timepoints:
        vol = np.random.normal(loc=100 + tp * 10, scale=25, size=shape).astype(np.float32)
        nib.save(nib.Nifti1Image(vol, affine=np.eye(4)), output_dir / f"cropped_000{tp}" / f"{pid}_{tp:04d}_cropped.nii.gz")

# Zip the folder
zip_path = Path("mini_dummy_dataset.zip")
with ZipFile(zip_path, 'w') as zipf:
    for path in output_dir.rglob('*'):
        zipf.write(path, path.relative_to(output_dir.parent))

print("✅ Dummy dataset zipped:", zip_path)
