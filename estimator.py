"""
Estimator classes for fitting additive models to score matrices.
"""

import numpy as np
from typing import Optional, List


class Estimator:
    """
    Base class for estimation results.
    
    Attributes:
        theta: Estimated student abilities
        beta: Estimated problem difficulties
        X_imputed: Imputed matrix (missing values filled in)
        mu: Global mean (EM method only)
    """
    
    def __init__(self):
        self.theta: Optional[np.ndarray] = None
        self.beta: Optional[np.ndarray] = None
        self.X_imputed: Optional[np.ndarray] = None
        self.mu: Optional[float] = None
    
    @classmethod
    def em(cls, X: np.ndarray, reg_lambda: float = 1.0,
           max_iter: int = 100, tol: float = 1e-4) -> 'Estimator':
        """
        Regularized EM (Alternating Least Squares) algorithm.
        
        Fits the additive model: X_ij = mu + theta_i + beta_j + epsilon_ij
        
        Args:
            X: Score matrix with NaN for missing values
            reg_lambda: Regularization parameter (shrinks theta and beta toward 0)
            max_iter: Maximum iterations
            tol: Convergence tolerance
            
        Returns:
            Estimator with fitted parameters
        """
        instance = cls()
        X_work = X.copy()
        n_students, n_problems = X_work.shape
        
        # Initialization
        mu = np.nanmean(X_work)
        theta = np.zeros(n_students)
        beta = np.zeros(n_problems)
        
        # Observation masks
        observed_mask = ~np.isnan(X_work)
        n_obs_student = np.sum(observed_mask, axis=1)
        n_obs_problem = np.sum(observed_mask, axis=0)
        
        for iteration in range(max_iter):
            theta_old = theta.copy()
            beta_old = beta.copy()
            
            # Update Theta (student abilities)
            res_theta = X_work - mu - beta[np.newaxis, :]
            theta = np.nansum(res_theta, axis=1) / (n_obs_student + reg_lambda)
            
            # Update Beta (problem difficulties)
            res_beta = X_work - mu - theta[:, np.newaxis]
            beta = np.nansum(res_beta, axis=0) / (n_obs_problem + reg_lambda)
            
            # Check convergence
            diff = np.linalg.norm(theta - theta_old) + np.linalg.norm(beta - beta_old)
            if diff < tol:
                break
        
        # Final imputation
        X_filled = mu + theta[:, np.newaxis] + beta[np.newaxis, :]
        X_filled = np.clip(X_filled, 0, 6)
        
        instance.theta = theta
        instance.beta = beta
        instance.X_imputed = X_filled
        instance.mu = mu
        
        return instance
    
    @classmethod
    def day_linking(cls, X: np.ndarray, groups: List[np.ndarray]) -> 'Estimator':
        """
        Day-linking heuristic estimator.
        
        Uses group × day means and linear algebra to estimate parameters,
        assuming each group worked on different problems on different days.
        
        Args:
            X: Score matrix with NaN for missing values
            groups: List of arrays, each containing student indices for a group
            
        Returns:
            Estimator with fitted parameters
        """
        instance = cls()
        n_students, n_problems = X.shape
        n_groups = len(groups)
        n_days = 3
        block_size = n_problems // n_days
        
        # Calculate group × day means
        group_day_means = np.zeros((n_groups, n_days))
        for g in range(n_groups):
            students = groups[g]
            for d in range(n_days):
                cols = slice(block_size * d, block_size * (d + 1))
                block = X[students, cols]
                if not np.isnan(block).all():
                    group_day_means[g, d] = np.nanmean(block)
                else:
                    group_day_means[g, d] = np.nan
        
        # Build linear system: mean = group_effect + day_effect
        equations = []
        targets = []
        for g in range(n_groups):
            for d in range(n_days):
                if not np.isnan(group_day_means[g, d]):
                    row = np.zeros(n_groups + n_days)
                    row[g] = 1
                    row[n_groups + d] = 1
                    equations.append(row)
                    targets.append(group_day_means[g, d])
        
        # Add constraints: sum of group effects = 0, sum of day effects = 0
        equations.append([1] * n_groups + [0] * n_days)
        targets.append(0)
        equations.append([0] * n_groups + [1] * n_days)
        targets.append(0)
        
        # Solve
        sol, _, _, _ = np.linalg.lstsq(equations, targets, rcond=None)
        day_effects = sol[n_groups:]
        
        # Expand day effects to per-problem beta
        beta_est = np.repeat(day_effects, block_size)
        
        # Estimate theta for each student
        theta_est = np.zeros(n_students)
        for i in range(n_students):
            valid_vals = []
            for d in range(n_days):
                cols = slice(block_size * d, block_size * (d + 1))
                block = X[i, cols]
                if not np.isnan(block).all():
                    valid_vals.extend(block[~np.isnan(block)] - day_effects[d])
            if valid_vals:
                theta_est[i] = np.mean(valid_vals)
        
        # Imputation
        X_filled = theta_est[:, np.newaxis] + beta_est[np.newaxis, :]
        X_filled = np.clip(X_filled, 0, 6)
        
        instance.theta = theta_est
        instance.beta = beta_est
        instance.X_imputed = X_filled
        
        return instance
    
    @classmethod
    def mean_imputation(cls, X: np.ndarray) -> 'Estimator':
        """
        Additive mean imputation heuristic.
        
        Estimates: X_ij ~ row_mean_i + col_mean_j - global_mean
        
        Args:
            X: Score matrix with NaN for missing values
            
        Returns:
            Estimator with fitted parameters
        """
        instance = cls()
        
        row_means = np.nanmean(X, axis=1)
        col_means = np.nanmean(X, axis=0)
        mu = np.nanmean(X)
        
        # Parameters relative to global mean
        theta_est = row_means - mu
        beta_est = col_means - mu
        
        # Imputation
        X_filled = row_means[:, np.newaxis] + col_means[np.newaxis, :] - mu
        X_filled = np.clip(X_filled, 0, 6)
        
        instance.theta = theta_est
        instance.beta = beta_est
        instance.X_imputed = X_filled
        instance.mu = mu
        
        return instance
