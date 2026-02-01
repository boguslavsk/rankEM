"""
Excel User Defined Functions (UDFs) for RankEM estimation algorithms.

This module exposes the EM and heuristic estimators as Excel functions using xlwings.
Users can select a range of scores and get back estimates for student abilities (theta),
problem difficulties (beta), and imputed values.

Usage in Excel:
    =RankEM_Estimate(A1:H20, "em")           - Returns theta (student abilities)
    =RankEM_Beta(A1:H20, "em")              - Returns beta (problem difficulties)
    =RankEM_Imputed(A1:H20, "em")           - Returns imputed matrix
    =RankEM_Stats(A1:H20, "em")             - Returns statistics summary
"""

import xlwings as xw
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path to import estimator
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from estimator import Estimator


def _range_to_array(data) -> np.ndarray:
    """
    Convert Excel range data to numpy array.
    
    Handles:
    - Empty cells -> NaN
    - Text values -> NaN
    - Numeric values -> float
    """
    if isinstance(data, (int, float)):
        # Single cell
        return np.array([[data]], dtype=float)
    
    # Convert list of lists to numpy array
    arr = np.array(data, dtype=object)
    
    # Create float array, converting non-numeric to NaN
    result = np.empty(arr.shape, dtype=float)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1] if len(arr.shape) > 1 else 1):
            val = arr[i, j] if len(arr.shape) > 1 else arr[i]
            if val is None or val == '' or val == ' ':
                result[i, j] = np.nan
            else:
                try:
                    result[i, j] = float(val)
                except (ValueError, TypeError):
                    result[i, j] = np.nan
    
    return result


def _run_estimator(data, method: str = "em", **kwargs) -> Estimator:
    """
    Run the specified estimator on the data.
    
    Args:
        data: Excel range data
        method: Estimation method - "em", "mean_imputation", or "day_average"
        **kwargs: Additional parameters for the estimator
        
    Returns:
        Estimator instance with results
    """
    X = _range_to_array(data)
    
    method = method.lower().strip()
    
    if method == "em":
        lambda_theta = kwargs.get('lambda_theta', 1.0)
        lambda_beta = kwargs.get('lambda_beta', 1.0)
        max_iter = kwargs.get('max_iter', 100)
        return Estimator.em(X, lambda_theta=lambda_theta, lambda_beta=lambda_beta, max_iter=max_iter)
    
    elif method in ("mean_imputation", "mean", "imputation"):
        return Estimator.mean_imputation(X)
    
    elif method in ("day_average", "day", "average"):
        n_blocks = kwargs.get('n_blocks', 3)
        return Estimator.day_average(X, n_blocks=n_blocks)
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'em', 'mean_imputation', or 'day_average'.")


# =============================================================================
# Excel User Defined Functions (UDFs)
# =============================================================================

@xw.func
@xw.arg('data', np.ndarray, ndim=2)
@xw.ret(expand='table')
def RankEM_Theta(data, method="em", lambda_param=1.0):
    """
    Estimate student abilities (theta) from a score matrix.
    
    Args:
        data: Range of scores (students x problems). Empty cells = missing.
        method: "em", "mean_imputation", or "day_average"
        lambda_param: Regularization parameter for EM (default 1.0)
        
    Returns:
        Column vector of theta estimates, one per student
    """
    try:
        est = _run_estimator(data, method, lambda_theta=lambda_param, lambda_beta=lambda_param)
        # Return as column vector
        return est.theta.reshape(-1, 1)
    except Exception as e:
        return [[f"Error: {str(e)}"]]


@xw.func
@xw.arg('data', np.ndarray, ndim=2)
@xw.ret(expand='table')
def RankEM_Beta(data, method="em", lambda_param=1.0):
    """
    Estimate problem difficulties (beta) from a score matrix.
    
    Args:
        data: Range of scores (students x problems). Empty cells = missing.
        method: "em", "mean_imputation", or "day_average"
        lambda_param: Regularization parameter for EM (default 1.0)
        
    Returns:
        Row vector of beta estimates, one per problem
    """
    try:
        est = _run_estimator(data, method, lambda_theta=lambda_param, lambda_beta=lambda_param)
        # Return as row vector
        return est.beta.reshape(1, -1)
    except Exception as e:
        return [[f"Error: {str(e)}"]]


@xw.func
@xw.arg('data', np.ndarray, ndim=2)
@xw.ret(expand='table')
def RankEM_Imputed(data, method="em", lambda_param=1.0):
    """
    Get the imputed score matrix (missing values filled in).
    
    Args:
        data: Range of scores (students x problems). Empty cells = missing.
        method: "em", "mean_imputation", or "day_average"
        lambda_param: Regularization parameter for EM (default 1.0)
        
    Returns:
        Full matrix with imputed values for missing cells
    """
    try:
        est = _run_estimator(data, method, lambda_theta=lambda_param, lambda_beta=lambda_param)
        return est.X_imputed
    except Exception as e:
        return [[f"Error: {str(e)}"]]


@xw.func
@xw.arg('data', np.ndarray, ndim=2)
@xw.ret(expand='table')
def RankEM_Stats(data, method="em", lambda_param=1.0):
    """
    Get summary statistics from the estimation.
    
    Args:
        data: Range of scores (students x problems). Empty cells = missing.
        method: "em", "mean_imputation", or "day_average"
        lambda_param: Regularization parameter for EM (default 1.0)
        
    Returns:
        Table of statistics: mu, std_theta, std_beta, sigma_epsilon, n_iterations
    """
    try:
        est = _run_estimator(data, method, lambda_theta=lambda_param, lambda_beta=lambda_param)
        
        stats = [
            ["Statistic", "Value"],
            ["Global Mean (μ)", est.mu if est.mu is not None else "N/A"],
            ["Std(θ)", f"{est.std_theta:.4f}" if est.std_theta is not None else "N/A"],
            ["Std(β)", f"{est.std_beta:.4f}" if est.std_beta is not None else "N/A"],
            ["σ(ε) Residual Std Dev", f"{est.sigma_epsilon:.4f}" if est.sigma_epsilon is not None else "N/A"],
            ["Iterations", est.n_iterations if est.n_iterations is not None else "N/A"]
        ]
        return stats
    except Exception as e:
        return [[f"Error: {str(e)}"]]


@xw.func
@xw.arg('data', np.ndarray, ndim=2)
@xw.ret(expand='table')
def RankEM_Ranking(data, method="em", lambda_param=1.0):
    """
    Get students ranked by estimated ability (highest first).
    
    Args:
        data: Range of scores (students x problems). Empty cells = missing.
        method: "em", "mean_imputation", or "day_average"
        lambda_param: Regularization parameter for EM (default 1.0)
        
    Returns:
        Table with Rank, Student Row, and Theta estimate
    """
    try:
        est = _run_estimator(data, method, lambda_theta=lambda_param, lambda_beta=lambda_param)
        
        # Sort by theta (descending)
        sorted_indices = np.argsort(est.theta)[::-1]
        
        results = [["Rank", "Student Row", "θ Estimate"]]
        for rank, idx in enumerate(sorted_indices, 1):
            results.append([rank, idx + 1, f"{est.theta[idx]:+.4f}"])
        
        return results
    except Exception as e:
        return [[f"Error: {str(e)}"]]


@xw.func
@xw.arg('data', np.ndarray, ndim=2)
@xw.ret(expand='table')  
def RankEM_AllMethods(data, lambda_param=1.0):
    """
    Run all three estimation methods and compare results.
    
    Args:
        data: Range of scores (students x problems). Empty cells = missing.
        lambda_param: Regularization parameter for EM (default 1.0)
        
    Returns:
        Comparison table of theta estimates from all methods
    """
    try:
        em_est = _run_estimator(data, "em", lambda_theta=lambda_param, lambda_beta=lambda_param)
        mean_est = _run_estimator(data, "mean_imputation")
        day_est = _run_estimator(data, "day_average")
        
        n_students = len(em_est.theta)
        
        results = [["Student", "θ (EM)", "θ (Mean Imp.)", "θ (Day Avg.)"]]
        for i in range(n_students):
            results.append([
                i + 1,
                f"{em_est.theta[i]:+.4f}",
                f"{mean_est.theta[i]:+.4f}",
                f"{day_est.theta[i]:+.4f}"
            ])
        
        return results
    except Exception as e:
        return [[f"Error: {str(e)}"]]


@xw.func
def RankEM_Version():
    """Return the add-in version."""
    return "RankEM Excel Add-in v1.0.0"


@xw.func
def RankEM_Help():
    """Return help text for available functions."""
    return """RankEM Excel Functions:
    
=RankEM_Theta(data, [method], [lambda])
    Returns student ability estimates (θ)
    
=RankEM_Beta(data, [method], [lambda])  
    Returns problem difficulty estimates (β)
    
=RankEM_Imputed(data, [method], [lambda])
    Returns the imputed score matrix
    
=RankEM_Stats(data, [method], [lambda])
    Returns estimation statistics
    
=RankEM_Ranking(data, [method], [lambda])
    Returns students ranked by ability
    
=RankEM_AllMethods(data, [lambda])
    Compares all three methods

Methods: "em", "mean_imputation", "day_average"
Default lambda: 1.0 (regularization for EM)
"""
