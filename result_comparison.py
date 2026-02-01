import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import os
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Compare two raw data files and associated estimates.')
parser.add_argument('file0', type=str, help='Path to the old raw data file (X0)')
parser.add_argument('file1', type=str, help='Path to the new raw data file (X1)')
args = parser.parse_args()

# Derive folders from input files
folder0 = os.path.dirname(args.file0)
folder1 = os.path.dirname(args.file1)

# Loading raw data for comparison
X0 = pd.read_csv(args.file0, header=None, index_col=0)
X1 = pd.read_csv(args.file1, header=None, index_col=0)

# Strip leading single quotes from indices to ensure matching
X0.index = X0.index.astype(str).str.lstrip("'")
X1.index = X1.index.astype(str).str.lstrip("'")

# Align columns for comparison
X1.columns = X0.columns

print(f"Input shapes: X1={X1.shape}, X0={X0.shape}, no missing {X1.isna().sum().sum()}/{X0.isna().sum().sum()}")

# Compare X0 and X1 row by row and print discrepancies
print("--- Row by row comparison (X0 vs X1) ---")
common_idx = X0.index.intersection(X1.index)
common_cols = X0.columns.intersection(X1.columns)

discrepancies_found = 0
for idx in common_idx:
    row0 = X0.loc[idx, common_cols]
    row1 = X1.loc[idx, common_cols]
    
    if not row0.equals(row1):
        mask = (row0 != row1) & ~(row0.isna() & row1.isna())
        if mask.any():
            discrepancies_found += 1
            diff_cols = common_cols[mask]
            print(f"Row {idx} discrepancies in columns: {list(diff_cols)}")
            for col in diff_cols:
                val0 = row0[col]
                val1 = row1[col]
                if pd.isna(val0) != pd.isna(val1):
                    type_str = " (Value vs NaN)"
                else:
                    type_str = " (Value Change)"
                print(f"  Col {col}: X0={val0}, X1={val1}{type_str}")

if discrepancies_found == 0:
    if len(common_idx) == 0:
        print("No overlapping index found to compare.")
    else:
        print("No row-by-row discrepancies found in aligned indices/columns.")
else:
    print(f"\nTotal rows with discrepancies: {discrepancies_found}")

# Read theta and beta estimates
beta0 = pd.read_csv(os.path.join(folder0, "beta_all_methods.csv"), index_col=0)
beta1 = pd.read_csv(os.path.join(folder1, "beta_all_methods.csv"), index_col=0)
theta0 = pd.read_csv(os.path.join(folder0, "theta_all_methods.csv"), index_col=0)
theta1 = pd.read_csv(os.path.join(folder1, "theta_all_methods.csv"), index_col=0)

# Strip leading single quotes from indices to ensure matching (same as with X0 and X1)
for df in [beta0, beta1, theta0, theta1]:
    df.index = df.index.astype(str).str.lstrip("'")

print(f"\nLoaded estimates from {folder0} and {folder1}")
print(f"Theta common indices: {len(theta0.index.intersection(theta1.index))}/{len(theta0)}")

def calculate_correlations(df0, df1, label):
    print(f"\n--- {label} Correlations (Old vs New) ---")
    common_cols = df0.columns.intersection(df1.columns)
    
    results = []
    for col in common_cols:
        common_idx = df0.index.intersection(df1.index)
        v0 = df0.loc[common_idx, col]
        v1 = df1.loc[common_idx, col]
        
        pearson = v0.corr(v1)
        spearman, _ = spearmanr(v0, v1)
        
        print(f"{col:20s}: Pearson = {pearson:.4f}, Spearman = {spearman:.4f}")
        results.append({"Method": col, "Pearson": pearson, "Spearman": spearman})
    
    return pd.DataFrame(results)

beta_corr = calculate_correlations(beta0, beta1, "Beta")
theta_corr = calculate_correlations(theta0, theta1, "Theta")

method = 'em_estimator'

def print_outliers(df0, df1, method, label_prefix, threshold=2.5):
    common_idx = df0.index.intersection(df1.index)
    x = df0.loc[common_idx, method]
    y = df1.loc[common_idx, method]
    
    # Calculate residuals from y=x trend line
    residuals = y - x
    std_res = residuals.std()
    
    # Identify outliers
    outlier_mask = residuals.abs() > threshold * std_res
    outliers = common_idx[outlier_mask]
    
    if len(outliers) > 0:
        print(f"{label_prefix} outliers (> {threshold} sigma):")
        for idx in outliers:
            sigma_dev = residuals.loc[idx] / std_res
            print(f"  Index: {idx}, Deviation: {sigma_dev:.2f} sigma")
    else:
        print(f"{label_prefix} outliers (> {threshold} sigma): None")

print("\n--- Outlier Detection ---")
print_outliers(beta0, beta1, method, 'Beta')
print_outliers(theta0, theta1, method, 'Theta')
