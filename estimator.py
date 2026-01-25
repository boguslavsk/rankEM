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
    def em(cls, X: np.ndarray, lambda_theta: float = 1.0, lambda_beta: float = 1.0,
           max_iter: int = 100, tol: float = 1e-4) -> 'Estimator':
        """
        Regularized EM (Alternating Least Squares) algorithm with L2 penalties.
        
        Fits the additive model: X_ij = mu + theta_i + beta_j + epsilon_ij
        
        Minimizes: sum((X_ij - theta_i - beta_j)^2) + lambda_theta * sum(theta_i^2) + lambda_beta * sum(beta_j^2)
        
        Args:
            X: Score matrix with NaN for missing values
            lambda_theta: L2 regularization parameter for theta (student abilities)
            lambda_beta: L2 regularization parameter for beta (problem difficulties)
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
            # Minimizing sum_j (X_ij - mu - theta_i - beta_j)^2 + lambda_theta * theta_i^2
            # Optimal: theta_i = sum_j(X_ij - mu - beta_j) / (n_obs_i + lambda_theta)
            res_theta = X_work - mu - beta[np.newaxis, :]
            theta = np.nansum(res_theta, axis=1) / (n_obs_student + lambda_theta)
            
            # Update Beta (problem difficulties)
            # Minimizing sum_i (X_ij - mu - theta_i - beta_j)^2 + lambda_beta * beta_j^2
            # Optimal: beta_j = sum_i(X_ij - mu - theta_i) / (n_obs_j + lambda_beta)
            res_beta = X_work - mu - theta[:, np.newaxis]
            beta = np.nansum(res_beta, axis=0) / (n_obs_problem + lambda_beta)
            
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
    def chain_linking(cls, X: np.ndarray, groups: List[np.ndarray]) -> 'Estimator':
        """
        Chain-linking heuristic estimator.
        
        Uses group × day means and linear algebra to estimate both group effects
        and day effects, assuming each group worked on different problems on 
        different days. Solves a linear system to separate the effects.
        
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
    
    @classmethod
    def day_average(cls, X: np.ndarray, n_blocks: int = 3) -> 'Estimator':
        """
        Day-average heuristic estimator.
        
        A simple heuristic that:
        1. For each block (day), computes the average observed score
        2. Rescales all days to a common average (the global mean)
        3. Estimates theta_i as the mean of rescaled scores for student i
        
        This is simpler than chain_linking as it doesn't require group structure.
        
        Args:
            X: Score matrix with NaN for missing values
            n_blocks: Number of day blocks (default 3)
            
        Returns:
            Estimator with fitted parameters
        """
        instance = cls()
        n_students, n_problems = X.shape
        block_size = n_problems // n_blocks
        
        # Global mean for rescaling target
        mu = np.nanmean(X)
        
        # Compute day means (average score in each block)
        day_means = np.zeros(n_blocks)
        for d in range(n_blocks):
            start_col = block_size * d
            end_col = n_problems if d == n_blocks - 1 else block_size * (d + 1)
            block = X[:, start_col:end_col]
            day_means[d] = np.nanmean(block)
        
        # Beta: day effect relative to global mean
        # (problems in harder days have lower mean -> lower beta)
        beta_est = np.zeros(n_problems)
        for d in range(n_blocks):
            start_col = block_size * d
            end_col = n_problems if d == n_blocks - 1 else block_size * (d + 1)
            # Day effect is how much this day differs from global mean
            beta_est[start_col:end_col] = day_means[d] - mu
        
        # Rescale X so all days have the same mean
        # X_rescaled[i,j] = X[i,j] - day_mean[d] + global_mean = X[i,j] - beta[j]
        X_rescaled = X.copy()
        for d in range(n_blocks):
            start_col = block_size * d
            end_col = n_problems if d == n_blocks - 1 else block_size * (d + 1)
            X_rescaled[:, start_col:end_col] = X[:, start_col:end_col] - (day_means[d] - mu)
        
        # Theta: mean rescaled score for each student (relative to global mean)
        theta_est = np.nanmean(X_rescaled, axis=1) - mu
        
        # Imputation
        X_filled = mu + theta_est[:, np.newaxis] + beta_est[np.newaxis, :]
        X_filled = np.clip(X_filled, 0, 6)
        
        instance.theta = theta_est
        instance.beta = beta_est
        instance.X_imputed = X_filled
        instance.mu = mu
        
        return instance
