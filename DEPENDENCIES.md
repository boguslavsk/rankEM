# Project Dependencies

This project relies on the following external Python libraries and standard modules.

## External Packages
These packages should be installed via `pip`.

| Package | Purpose | Used In |
|:--------|:--------|:--------|
| **numpy** | Numerical operations, array handling, linear algebra (lstsq) | core, experiments, add-in |
| **pandas** | Data loading (CSV/Excel), dataframes, analysis | core, experiments |
| **scipy** | Scientific computing (Spearman rank correlation) | experiments |
| **openpyxl** | Excel file I/O (backend for pandas `read_excel`) | experiments (data_generator) |

### Installation
To install all required external dependencies, run:

```bash
pip install numpy pandas scipy openpyxl
```

## Standard Library Modules
These are built-in to Python and do not require separate installation.

- **argparse**: For parsing command-line arguments in `run_real_data.py` and comparison scripts.
- **csv**: For reading/writing CSV files (used alongside pandas).
- **datetime**: For timestamping reports.
- **os**: For operating system operations (file paths, changing directories).
- **pathlib**: For object-oriented filesystem paths.
- **subprocess**: For installing xlwings and creating module files.
- **sys**: For system paths and executable info.
- **typing**: For type hints (`List`, `Optional`, etc.).

## Internal Module Structure
The project uses the following internal modules:

- **estimator.py**: Contains the `Estimator` class with `em`, `chain_linking`, `mean_imputation`, and `day_average` methods.
- **experiments/**:
  - **data_generator.py**: `DataGenerator` class for simulations.
  - **run_experiments.py**: Main experiment runner.
  - **monitor/** & **analyze/** scripts: Various analysis tools.

