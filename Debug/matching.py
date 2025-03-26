import os
import pandas as pd

# Define dataset directory
data_dir = "/mimer/NOBACKUP/groups/biomedicalimaging-kth/miladfa/mama_mia/cropped_0001"
filtered_file = "/mimer/NOBACKUP/groups/biomedicalimaging-kth/miladfa/mama_mia/filtered_clinical_and_imaging_info.xlsx"

# Load filtered dataset
filtered_df = pd.read_excel(filtered_file, sheet_name="Sheet1")
valid_patient_ids = set(filtered_df["patient_id"])

# Get list of image files
file_list = [f for f in os.listdir(data_dir) if f.endswith(".nii.gz")]

# Extract patient IDs from filenames
extracted_patient_ids = {"_".join(f.split("_")[:2]) for f in file_list}

# Compare valid patients
matching_patient_count = len(extracted_patient_ids.intersection(valid_patient_ids))

print(f"Total detected patients from images: {len(extracted_patient_ids)}")
print(f"Patients matching the filtered dataset: {matching_patient_count}")
