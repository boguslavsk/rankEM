"""
DataGenerator class for loading or simulating score matrices with missing data patterns.
"""

import numpy as np
from typing import Optional, List
from pathlib import Path


class DataGenerator:
    """
    Generates or loads score matrices with optional missing data patterns.
    
    Attributes:
        X_complete: Complete score matrix (before missing values applied)
        X_missing: Matrix with missing values (NaN)
        theta_true: True student abilities (simulation only, None for file input)
        beta_true: True problem difficulties (simulation only, None for file input)
        groups: Student group assignments (for block missingness pattern)
        student_labels: Row labels from file (file input only)
        problem_labels: Column labels from file (file input only)
    """
    
    def __init__(self):
        self.X_complete: Optional[np.ndarray] = None
        self.X_missing: Optional[np.ndarray] = None
        self.theta_true: Optional[np.ndarray] = None
        self.beta_true: Optional[np.ndarray] = None
        self.groups: Optional[List[np.ndarray]] = None
        self.student_labels: Optional[List[str]] = None
        self.problem_labels: Optional[List[str]] = None
        self.n_blocks: int = 3
        self._rng: Optional[np.random.Generator] = None
    
    @classmethod
    def from_file(cls, filename: str) -> 'DataGenerator':
        """
        Load data from xlsx or csv file.
        
        The file should have:
        - First row: problem/column headers
        - First column: student/row labels
        - Remaining cells: score values
        
        Args:
            filename: Path to xlsx or csv file
            
        Returns:
            DataGenerator instance with X_complete populated
        """
        import pandas as pd
        
        instance = cls()
        path = Path(filename)
        
        if path.suffix.lower() == '.xlsx':
            df = pd.read_excel(filename, index_col=0)
        elif path.suffix.lower() == '.csv':
            df = pd.read_csv(filename, index_col=0)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}. Use .xlsx or .csv")
        
        instance.X_complete = df.values.astype(float)
        instance.X_missing = instance.X_complete.copy()
        instance.student_labels = list(df.index)
        instance.problem_labels = list(df.columns)
        instance._rng = np.random.default_rng(seed=42)
        
        return instance
    
    @classmethod
    def from_simulation(cls,
                        n_students: int = 60,
                        n_problems: int = 24,
                        seed: int = 42,
                        sigma: float = 1.5,
                        problem_type: str = 'block',
                        n_blocks: int = 3) -> 'DataGenerator':
        """
        Generate synthetic data with known ground truth.
        
        Args:
            n_students: Number of students (rows)
            n_problems: Number of problems (columns)
            seed: Random seed for reproducibility
            sigma: Standard deviation of noise term
            problem_type: 'uniform' for uniform difficulty, 'block' for block-varying
            
        Returns:
            DataGenerator instance with X_complete, theta_true, beta_true populated
        """
        instance = cls()
        instance.n_blocks = n_blocks
        rng = np.random.default_rng(seed=seed)
        instance._rng = rng
        
        # True student abilities (theta)
        instance.theta_true = rng.normal(loc=0, scale=2, size=n_students)
        
        # True problem difficulties (beta)
        if problem_type == 'uniform':
            instance.beta_true = rng.uniform(-3, 3, size=n_problems)
        elif problem_type == 'block':
            # Block-varying problem difficulty
            block_size = n_problems // n_blocks
            betas = []
            means = [1, 0, -1] # Cycle through Easy, Neutral, Hard
            for i in range(n_blocks):
                m = means[i % len(means)]
                betas.append(rng.uniform(m-3, m+3, size=block_size))
            
            # Handle remainder columns
            remainder = n_problems % n_blocks
            if remainder > 0:
                betas.append(rng.uniform(-3, 3, size=remainder))
                
            instance.beta_true = np.concatenate(betas)
        else:
            raise ValueError("problem_type must be 'uniform' or 'block'")
        
        # Generate score matrix: X = mu + theta + beta + noise
        mu = 3.5
        eps = rng.normal(loc=0, scale=sigma, size=(n_students, n_problems))
        X_underlying = mu + instance.theta_true[:, np.newaxis] + instance.beta_true[np.newaxis, :] + eps
        
        # Discretize and clip to [0, 6] range
        instance.X_complete = np.clip(np.round(X_underlying), 0, 6)
        instance.X_missing = instance.X_complete.copy()
        
        return instance
    
    def apply_missing(self,
                      pattern: str = 'block',
                      correlation: str = 'none',
                      missing_rate: float = 0.33,
                      correlation_strength: float = 1.0,
                      n_blocks: Optional[int] = None) -> 'DataGenerator':
        """
        Apply missing data pattern to the complete matrix.
        
        Args:
            pattern: 'block', 'scattered', or 'both'
            correlation: 'none', 'theta', 'beta', or 'both'
            missing_rate: Proportion of values to make missing (for scattered pattern)
            correlation_strength: How strongly missingness correlates with parameters (0-1)
            n_blocks: Optional number of blocks (overrides instance.n_blocks if provided)
            
        Returns:
            self (for method chaining)
        """
        if self.X_complete is None:
            raise ValueError("No data loaded. Use from_file() or from_simulation() first.")
        
        # Start fresh from complete data
        self.X_missing = self.X_complete.astype(float).copy()
        n_students, n_problems = self.X_missing.shape
        
        if self._rng is None:
            self._rng = np.random.default_rng(seed=42)
        
        if n_blocks is not None:
            self.n_blocks = n_blocks
            
        # Apply block pattern
        if pattern in ('block', 'both'):
            self._apply_block_missing(n_students, n_problems, correlation, correlation_strength)
        
        # Apply scattered pattern
        if pattern in ('scattered', 'both'):
            self._apply_scattered_missing(n_students, n_problems, missing_rate,
                                          correlation, correlation_strength)
        
        return self
    
    def _apply_block_missing(self, n_students: int, n_problems: int,
                              correlation: str = 'none', 
                              correlation_strength: float = 1.0):
        """Apply block-based missingness pattern.
        
        If correlation='theta' or 'both', higher ability students are assigned
        to groups that miss more blocks (later blocks).
        If correlation='beta' or 'both', harder problems (lower beta) are in 
        blocks that more students miss.
        """
        n_groups = self.n_blocks
        block_size = n_problems // n_groups
        
        # Determine student grouping
        if correlation in ('theta', 'both') and self.theta_true is not None:
            # Sort students by theta - higher ability assigned to later groups (more missing)
            # Add noise based on correlation_strength (1.0 = perfect sorting, 0.0 = random)
            noise = self._rng.normal(0, 1, size=n_students) * (1 - correlation_strength)
            perturbed_order = np.argsort(self.theta_true + noise * self.theta_true.std())
            self.groups = np.array_split(perturbed_order, n_groups)
        else:
            # Random permutation (original behavior)
            student_indices = self._rng.permutation(n_students)
            self.groups = np.array_split(student_indices, n_groups)
        
        # Determine problem block ordering
        if correlation in ('beta', 'both') and self.beta_true is not None:
            # Sort problem blocks by mean difficulty - harder blocks (lower beta mean) 
            # will be assigned to later groups (which have more students)
            block_means = []
            for i in range(n_groups):
                start = block_size * i
                end = n_problems if i == n_groups - 1 else block_size * (i + 1)
                block_means.append(np.mean(self.beta_true[start:end]))
            
            # Add noise based on correlation_strength
            noise = self._rng.normal(0, 1, size=n_groups) * (1 - correlation_strength)
            block_means_noisy = np.array(block_means) + noise * np.std(block_means)
            
            # Harder blocks (lower beta) should be missed by more students (later groups)
            # So sort ascending (hardest first) and assign to groups in order
            block_order = np.argsort(block_means_noisy)  # Ascending: hardest first
        else:
            block_order = np.arange(n_groups)
        
        # Each group misses one block of problems
        for group_idx, group in enumerate(self.groups):
            block_idx = block_order[group_idx]
            start_col = block_size * block_idx
            if block_idx == n_groups - 1:
                end_col = n_problems
            else:
                end_col = block_size * (block_idx + 1)
            self.X_missing[group, start_col:end_col] = np.nan

    
    def _apply_scattered_missing(self, n_students: int, n_problems: int,
                                  missing_rate: float, correlation: str,
                                  correlation_strength: float):
        """Apply scattered missingness pattern with optional correlation."""
        # Base probability matrix (uniform)
        prob_missing = np.ones((n_students, n_problems)) * missing_rate
        
        # Adjust probabilities based on correlation
        if correlation in ('theta', 'both') and self.theta_true is not None:
            # Higher ability students have more missingness
            theta_normalized = (self.theta_true - self.theta_true.min()) / (
                self.theta_true.max() - self.theta_true.min() + 1e-10)
            theta_adjustment = theta_normalized[:, np.newaxis] * correlation_strength * missing_rate
            prob_missing = prob_missing + theta_adjustment
        
        if correlation in ('beta', 'both') and self.beta_true is not None:
            # Harder problems (lower beta) have more missingness
            beta_normalized = 1 - (self.beta_true - self.beta_true.min()) / (
                self.beta_true.max() - self.beta_true.min() + 1e-10)
            beta_adjustment = beta_normalized[np.newaxis, :] * correlation_strength * missing_rate
            prob_missing = prob_missing + beta_adjustment
        
        # Clip probabilities to [0, 1]
        prob_missing = np.clip(prob_missing, 0, 1)
        
        # Generate random mask and apply
        random_mask = self._rng.random((n_students, n_problems)) < prob_missing
        
        # Only apply to currently non-missing values
        apply_mask = random_mask & ~np.isnan(self.X_missing)
        self.X_missing[apply_mask] = np.nan
    
    @property
    def n_students(self) -> int:
        """Number of students (rows)."""
        if self.X_complete is None:
            return 0
        return self.X_complete.shape[0]
    
    @property
    def n_problems(self) -> int:
        """Number of problems (columns)."""
        if self.X_complete is None:
            return 0
        return self.X_complete.shape[1]
    
    @property
    def n_missing(self) -> int:
        """Number of missing values."""
        if self.X_missing is None:
            return 0
        return int(np.isnan(self.X_missing).sum())
    
    @property
    def missing_rate_actual(self) -> float:
        """Actual proportion of missing values."""
        if self.X_missing is None:
            return 0.0
        return self.n_missing / self.X_missing.size
