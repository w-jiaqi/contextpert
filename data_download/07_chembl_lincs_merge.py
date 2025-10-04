import pandas as pd

file1 = pd.read_csv('')  # Processed OpenTargets file path
file2 = pd.read_csv('')  # Path to LINCS file with 'chembl_id', generated from brd_to_chembl.py

merged_df = pd.merge(file2, file1, left_on="chembl_id", right_on="drugId", how="left")
merged_df.to_csv("data/full_lincs_with_disease.csv", index=False)
