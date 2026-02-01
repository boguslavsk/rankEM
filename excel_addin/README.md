# RankEM Excel Add-in

This add-in provides Excel User Defined Functions (UDFs) for running the EM estimation and heuristic algorithms directly within Excel.

## Features

- **RankEM_Theta**: Get student ability estimates (θ)
- **RankEM_Beta**: Get problem difficulty estimates (β)
- **RankEM_Imputed**: Get the imputed score matrix (missing values filled)
- **RankEM_Stats**: Get summary statistics
- **RankEM_Ranking**: Get students ranked by ability
- **RankEM_AllMethods**: Compare results from all three methods

## Requirements

- Python 3.8+ with the rankEM environment
- Microsoft Excel (Windows)
- xlwings package

## Quick Installation

```powershell
# Navigate to the excel_addin directory
cd c:\Users\Michael\src\rankEM\excel_addin

# Run the installation script
python install.py
```

## Manual Setup

### 1. Install xlwings

```powershell
pip install xlwings
```

### 2. Install the xlwings add-in for Excel

```powershell
xlwings addin install
```

### 3. Configure Excel

1. Open Excel
2. Go to the **xlwings** tab in the ribbon
3. Click **Import Functions**
4. Configure:
   - **UDF Modules**: `rankEM_functions`
   - **PYTHONPATH**: `c:\Users\Michael\src\rankEM\excel_addin;c:\Users\Michael\src\rankEM`

## Usage in Excel

### Basic Usage

Select a range containing your score matrix (students in rows, problems in columns). Empty cells are treated as missing data.

| Function | Description |
|----------|-------------|
| `=RankEM_Theta(A1:H20, "em")` | Get θ estimates using EM |
| `=RankEM_Beta(A1:H20, "em")` | Get β estimates using EM |
| `=RankEM_Stats(A1:H20)` | Get summary statistics |
| `=RankEM_Ranking(A1:H20)` | Get ranked student list |

### Methods

| Method | Description |
|--------|-------------|
| `"em"` | Regularized EM algorithm (recommended) |
| `"mean_imputation"` | Additive mean imputation heuristic |
| `"day_average"` | Day-average rescaling heuristic |

### Parameters

- **data**: Excel range containing scores (students × problems)
- **method**: Estimation method (default: `"em"`)
- **lambda_param**: Regularization parameter for EM (default: `1.0`)

### Output

Functions return **array formulas** that automatically expand to fill the required cells:

- `RankEM_Theta`: Returns a column (one θ per student)
- `RankEM_Beta`: Returns a row (one β per problem)
- `RankEM_Imputed`: Returns a matrix (same size as input)
- `RankEM_Stats`: Returns a table of statistics
- `RankEM_Ranking`: Returns a ranked table

## Example Workflow

1. **Enter your data**: Put scores in a range, e.g., `A2:K21` (20 students × 10 problems)
   - Leave cells empty for missing data

2. **Get student rankings**:
   ```
   =RankEM_Ranking(A2:K21, "em", 1.0)
   ```
   This returns a table with Rank, Student Row, and θ Estimate.

3. **Compare methods**:
   ```
   =RankEM_AllMethods(A2:K21)
   ```
   This shows θ estimates from all three methods side-by-side.

4. **Get statistics**:
   ```
   =RankEM_Stats(A2:K21, "em")
   ```
   Shows μ, σ(θ), σ(β), σ(ε), and iteration count.

## Troubleshooting

### "Module not found" error

Ensure the PYTHONPATH in xlwings settings includes:
- `c:\Users\Michael\src\rankEM\excel_addin`
- `c:\Users\Michael\src\rankEM`

### "xlwings" tab not visible in Excel

Run: `xlwings addin install`

### Functions not updating

Press `Ctrl+Alt+F9` to force recalculation, or restart Excel.

### Python errors

Enable debug mode in xlwings settings to see Python console output.

## Technical Details

The add-in uses xlwings to bridge Excel and Python:

1. When you call a UDF like `=RankEM_Theta(...)`, Excel sends the data to Python
2. The `rankEM_functions.py` module processes the data using the `Estimator` class
3. Results are returned to Excel and displayed in cells

This preserves all the functionality of the command-line Python scripts while providing a user-friendly Excel interface.
