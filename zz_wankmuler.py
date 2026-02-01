import pandas as pd
from scipy.stats import pearsonr
import numpy as np
# --- 1. Load all sheets from an Excel file ---
file_path = "/mnt/d/project/hydraulic/PPT/Wankmuler_psi.xlsx"   # <-- change this
sheets = pd.read_excel(file_path, sheet_name=None)

# --- 2. Create storage for results ---
results = []

# --- 3. Loop through sheets and calculate correlation ---
for sheet_name, df in sheets.items():

    # drop rows with missing values in X or Y
    df_clean = df[["x", "y"]].dropna()
    df_clean = df_clean[df_clean['x']>16]
    #df_clean = df_clean[df_clean['x']<80]
    # compute Pearson correlation
    r, p = pearsonr(df_clean["x"], np.log10(df_clean["y"]))

    # store results
    results.append({
        "Sheet": sheet_name,
        "Pearson_r": r,
        "p_value": p
    })

# --- 4. Save to a DataFrame and print ---
results_df = pd.DataFrame(results)
print(results_df)