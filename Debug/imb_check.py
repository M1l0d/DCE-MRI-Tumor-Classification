import pandas as pd
df = pd.read_excel("/mimer/NOBACKUP/groups/biomedicalimaging-kth/miladfa/mama_mia/filtered_clinical_and_imaging_info_pcr.xlsx", sheet_name="Sheet1")
df = df.dropna(subset=["pcr"])
print(df["pcr"].value_counts())
