import os
import numpy as np
import nibabel as nib
import nrrd

# Directory Paths
DATA_DIR = "/mimer/NOBACKUP/groups/biomedicalimaging-kth/miladfa/mama_mia/images"
MASK_DIR = "/mimer/NOBACKUP/groups/biomedicalimaging-kth/miladfa/mama_mia/segmentations/expert"
OUTPUT_DIR = "/mimer/NOBACKUP/groups/biomedicalimaging-kth/miladfa/mama_mia/cropped_0002"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

"""
Loads a NIfTI file and returns a NumPy array

Returns (data, affine matrix, and header)
"""

def load_nifti(file_path):
    img = nib.load(file_path)
    return img.get_fdata(), img.affine, img.header

"""
Finds the bounding box of the tumor region in a 3D mask

Returns the bounding box as a tuple (x_min, x_max, y_min, y_max, z_min, z_max)
"""
def find_bounding_box(mask):
    coords = np.where(mask > 0)
    x_min, x_max = coords[0].min(), coords[0].max()
    y_min, y_max = coords[1].min(), coords[1].max()
    z_min, z_max = coords[2].min(), coords[2].max()
    return (x_min, x_max, y_min, y_max, z_min, z_max)


"""
Expands the bounding box by a margin while ensuring it stays within the image dimensions

Returns the expanded bounding box as a tuple (x_min, x_max, y_min, y_max, z_min, z_max)
"""
def expand_bounding_box(bbox, margin, shape):
    x_min, x_max, y_min, y_max, z_min, z_max = bbox
    x_min = max(0, x_min - margin)
    x_max = min(shape[0] - 1, x_max + margin)
    y_min = max(0, y_min - margin)
    y_max = min(shape[1] - 1, y_max + margin)
    z_min = max(0, z_min - margin)
    z_max = min(shape[2] - 1, z_max + margin)
    return (x_min, x_max, y_min, y_max, z_min, z_max)

"""
Crops the image using the segmentation mask and saves the cropped image

The margin parameter controls how much to expand the bounding box around the tumor region
"""

def crop_and_save(image, mask, file_name, affine, header, margin=10):
    bbox = find_bounding_box(mask)
    expanded_bbox = expand_bounding_box(bbox, margin, image.shape)
    
    x_min, x_max, y_min, y_max, z_min, z_max = expanded_bbox
    cropped_image = image[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]

    # Save the cropped image
    output_path = os.path.join(OUTPUT_DIR, file_name)
    cropped_nifti = nib.Nifti1Image(cropped_image, affine, header)
    nib.save(cropped_nifti, output_path)
    print(f"Saved cropped image to {output_path}")

def process_one_case(patient_id):
    patient_folder = os.path.join(DATA_DIR, patient_id)
    mask_file = os.path.join(MASK_DIR, f"{patient_id}.nii.gz") # Mask file name matches patient ID

    # Ensure the mask file exists
    if not os.path.exists(mask_file):
        print(f"Skipping {patient_id}: mask file not found")
        return
    
    # Find first post-contrast image (ending with '_0001.nii.gz')
    image_file = None
    for file in sorted(os.listdir(patient_folder)):
        if "_0002.nii.gz" in file:
            image_file = os.path.join(patient_folder, file)
            break

    if not image_file:
        print(f"No post-contrast image not found for {patient_id}")
        return
    
    # Load MRI and mask
    image, affine, header = load_nifti(image_file)
    mask, _, _ = load_nifti(mask_file)

    # Crop and save
    crop_and_save(image, mask, f"{patient_id}_0002_cropped.nii.gz", affine, header)

"""
Processes all patient folders in the dataset
"""
def process_all_cases():
    patient_ids = sorted(os.listdir(DATA_DIR))
    for patient_id in patient_ids:
        patient_path = os.path.join(DATA_DIR, patient_id)
        if os.path.isdir(patient_path):
            process_one_case(patient_id)



"""
Run processing based on user input

If a patient ID is provided, only that patient will be processed
Otherwise, all patients will be processed
"""
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--patient_id", type=str, default=None, help="Process a single patient ID, if not provided all patients will be processed")
    args = parser.parse_args()

    if args.patient_id:
        process_one_case(args.patient_id)
    else:
        process_all_cases()