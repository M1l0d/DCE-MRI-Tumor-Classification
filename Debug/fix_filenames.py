import os

# Define the directory where the incorrect names are
folder_path = "/mimer/NOBACKUP/groups/biomedicalimaging-kth/miladfa/mama_mia/cropped_0001"

# Iterate through all files
for filename in os.listdir(folder_path):
    if filename.endswith("_cropped.nii.gz"):  # Only process .nii.gz files
        base_name = filename.replace("_cropped.nii.gz", "")  # Extract patient ID
        new_filename = f"{base_name}_0001_cropped.nii.gz"  # Append _0001
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_filename)
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} → {new_filename}")

print("✅ Renaming complete!")