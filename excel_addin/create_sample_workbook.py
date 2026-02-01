"""
Create a sample Excel workbook demonstrating the RankEM functions.

This script creates an Excel file with:
1. Sample score data (with missing values)
2. Instructions for using the RankEM functions
3. Pre-configured cells showing how to call each function
"""

import xlwings as xw
import numpy as np
from pathlib import Path
import sys

# Add parent directory for estimator import
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


def create_sample_data(n_students=20, n_problems=8, missing_rate=0.33):
    """Generate sample score data with missing values."""
    np.random.seed(42)
    
    # Generate student abilities and problem difficulties
    theta_true = np.random.normal(0, 1, n_students)
    beta_true = np.random.normal(0, 0.5, n_problems)
    
    # Generate scores
    mu = 4.0  # Global mean
    X = mu + theta_true[:, np.newaxis] + beta_true[np.newaxis, :]
    X += np.random.normal(0, 0.8, X.shape)  # Add noise
    X = np.clip(X, 0, 6)  # Clamp to valid range
    
    # Add missing values (random pattern)
    missing_mask = np.random.random(X.shape) < missing_rate
    X[missing_mask] = np.nan
    
    return X, theta_true, beta_true


def create_sample_workbook():
    """Create the sample Excel workbook."""
    addin_dir = Path(__file__).parent
    output_path = addin_dir / "RankEM_Sample.xlsx"
    
    print("Creating sample workbook...")
    
    # Generate sample data
    X, theta_true, beta_true = create_sample_data()
    n_students, n_problems = X.shape
    
    # Create new workbook
    app = xw.App(visible=False)
    try:
        wb = app.books.add()
        
        # ===== Data Sheet =====
        data_sheet = wb.sheets[0]
        data_sheet.name = "Scores"
        
        # Title
        data_sheet.range("A1").value = "RankEM Sample Data"
        data_sheet.range("A1").font.bold = True
        data_sheet.range("A1").font.size = 14
        
        # Problem headers
        data_sheet.range("B2").value = "Problem"
        for j in range(n_problems):
            data_sheet.range((2, j + 3)).value = f"P{j+1}"
        
        # Student labels and data
        data_sheet.range("A3").value = "Student"
        for i in range(n_students):
            data_sheet.range((i + 3, 1)).value = f"S{i+1}"
            for j in range(n_problems):
                val = X[i, j]
                if not np.isnan(val):
                    data_sheet.range((i + 3, j + 3)).value = round(val, 1)
                # Leave cell empty for NaN
        
        # Format data range
        data_range = data_sheet.range(f"C3:J{n_students + 2}")
        data_range.number_format = "0.0"
        
        # Add instructions
        row = n_students + 5
        data_sheet.range(f"A{row}").value = "Instructions:"
        data_sheet.range(f"A{row}").font.bold = True
        data_sheet.range(f"A{row+1}").value = "1. Ensure xlwings add-in is enabled (File → Options → Add-Ins)"
        data_sheet.range(f"A{row+2}").value = "2. Go to xlwings tab → Import Functions"
        data_sheet.range(f"A{row+3}").value = "3. Set UDF Modules to: rankEM_functions"
        data_sheet.range(f"A{row+4}").value = f"4. Set PYTHONPATH to: {addin_dir};{addin_dir.parent}"
        data_sheet.range(f"A{row+5}").value = "5. See 'Results' and 'Functions' sheets for examples"
        
        # ===== Results Sheet =====
        results_sheet = wb.sheets.add("Results", after=data_sheet)
        
        results_sheet.range("A1").value = "RankEM Results"
        results_sheet.range("A1").font.bold = True
        results_sheet.range("A1").font.size = 14
        
        # Example formulas (as text, since UDFs aren't active yet)
        results_sheet.range("A3").value = "Student Rankings"
        results_sheet.range("A3").font.bold = True
        results_sheet.range("A4").value = "Formula:"
        results_sheet.range("B4").value = "=RankEM_Ranking(Scores!$C$3:$J$22, \"em\", 1.0)"
        results_sheet.range("A5").value = "(Paste this formula in cell A7)"
        
        results_sheet.range("A10").value = "Statistics"
        results_sheet.range("A10").font.bold = True
        results_sheet.range("A11").value = "Formula:"
        results_sheet.range("B11").value = "=RankEM_Stats(Scores!$C$3:$J$22, \"em\")"
        
        results_sheet.range("A16").value = "Compare All Methods"
        results_sheet.range("A16").font.bold = True
        results_sheet.range("A17").value = "Formula:"
        results_sheet.range("B17").value = "=RankEM_AllMethods(Scores!$C$3:$J$22)"
        
        # ===== Functions Reference Sheet =====
        func_sheet = wb.sheets.add("Functions", after=results_sheet)
        
        func_sheet.range("A1").value = "RankEM Function Reference"
        func_sheet.range("A1").font.bold = True
        func_sheet.range("A1").font.size = 14
        
        functions = [
            ["Function", "Description", "Example"],
            ["RankEM_Theta(data, method, lambda)", "Student ability estimates (θ)", "=RankEM_Theta(A1:H20, \"em\", 1.0)"],
            ["RankEM_Beta(data, method, lambda)", "Problem difficulty estimates (β)", "=RankEM_Beta(A1:H20, \"em\", 1.0)"],
            ["RankEM_Imputed(data, method, lambda)", "Imputed score matrix", "=RankEM_Imputed(A1:H20, \"em\")"],
            ["RankEM_Stats(data, method, lambda)", "Summary statistics", "=RankEM_Stats(A1:H20)"],
            ["RankEM_Ranking(data, method, lambda)", "Ranked student list", "=RankEM_Ranking(A1:H20)"],
            ["RankEM_AllMethods(data, lambda)", "Compare all methods", "=RankEM_AllMethods(A1:H20)"],
            ["RankEM_Version()", "Add-in version", "=RankEM_Version()"],
            ["RankEM_Help()", "Help text", "=RankEM_Help()"],
        ]
        
        for i, row_data in enumerate(functions):
            for j, val in enumerate(row_data):
                func_sheet.range((i + 3, j + 1)).value = val
        
        # Header formatting
        func_sheet.range("A3:C3").font.bold = True
        
        # Methods table
        func_sheet.range("A13").value = "Available Methods:"
        func_sheet.range("A13").font.bold = True
        methods = [
            ["Method", "Description"],
            ["em", "Regularized EM algorithm (recommended for biased missingness)"],
            ["mean_imputation", "Additive mean imputation (simple, fast)"],
            ["day_average", "Day-average rescaling (for block-missing patterns)"],
        ]
        for i, row_data in enumerate(methods):
            for j, val in enumerate(row_data):
                func_sheet.range((i + 14, j + 1)).value = val
        func_sheet.range("A14:B14").font.bold = True
        
        # Activate the Scores sheet
        data_sheet.activate()
        
        # Save
        wb.save(str(output_path))
        print(f"✓ Sample workbook saved to: {output_path}")
        
    finally:
        wb.close()
        app.quit()
    
    return output_path


if __name__ == "__main__":
    try:
        create_sample_workbook()
        print("\nSample workbook created successfully!")
        print("Open it in Excel to test the RankEM functions.")
    except ImportError:
        print("Error: xlwings is not installed.")
        print("Run: pip install xlwings")
    except Exception as e:
        print(f"Error creating workbook: {e}")
        print("\nYou can still use the RankEM functions in any Excel file.")
