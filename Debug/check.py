import pandas as pd

filtered_file = "/mimer/NOBACKUP/groups/biomedicalimaging-kth/miladfa/mama_mia/filtered_clinical_and_imaging_info_pcr.xlsx"
df = pd.read_excel(filtered_file, sheet_name="Sheet1")

# Show unique values in pCR column
print(df["pcr"].unique())
