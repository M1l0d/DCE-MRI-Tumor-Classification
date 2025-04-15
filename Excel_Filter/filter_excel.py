import pandas as pd

def filter_excel_by_column(input_file, column_name, output_filtered, output_filtered_out):
    """
    Filters an Excel file based on the presence of values in a specified column.

    Parameters:
        input_file (str): Path to the input Excel file.
        column_name (str): The column to filter by (rows missing this column will be removed).
        output_filtered (str): Path to save the filtered Excel file.
        output_filtered_out (str): Path to save the entries that were filtered out.
    """
    # Load the Excel file
    original_file = pd.read_excel(input_file, sheet_name="dataset_info")  # Adjust sheet name if necessary
    
    # Ensure the column exists
    if column_name not in original_file.columns:
        raise ValueError(f"Column '{column_name}' not found in the dataset.")
    
    # Separate entries based on the presence of values in the specified column
    filtered_file = original_file.dropna(subset=[column_name])
    filtered_entries = original_file[original_file[column_name].isna()]
    
    # Save the outputs
    filtered_file.to_excel(output_filtered, index=False)
    filtered_entries.to_excel(output_filtered_out, index=False)
    
    print(f"Filtering complete!")
    print(f"- Filtered file saved to: {output_filtered} ({len(filtered_file)} entries)")
    print(f"- Filtered-out file saved to: {output_filtered_out} ({len(filtered_entries)} entries)")

# Settings
input_file = "clinical_and_imaging_info.xlsx"  # File path
column_name = "pcr"  # Column to filter by
output_filtered = "filtered_clinical_and_imaging_info_pcr.xlsx"
output_filtered_out = "filtered_out_entries_pcr.xlsx"

filter_excel_by_column(input_file, column_name, output_filtered, output_filtered_out)