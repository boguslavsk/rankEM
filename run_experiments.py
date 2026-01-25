"""
Comprehensive experiment suite for evaluating estimators under different conditions.
"""

import numpy as np
from scipy.stats import spearmanr
import csv
from datetime import datetime
from data_generator import DataGenerator
from estimator import Estimator


def run_experiments(output_file: str = 'results/experiment_results.csv'):
    """
    Run full factorial experiment suite.
    
    Factors:
    - sigma: [0.5, 1.5, 3.5, 6.0]
    - correlation: ['none', 'theta', 'beta']
    - estimator: ['EM', 'Day', 'Imp']
    - seeds: 50
    
    Output: Long-form CSV with one row per (seed, sigma, correlation, estimator)
    """
    
    # Experimental factors
    sigmas = [0.5, 1.5, 3.5, 6.0]
    correlations = ['none', 'theta', 'beta', 'both']
    estimator_names = ['EM', 'Day', 'Imp']
    n_seeds = 50
    
    # Fixed parameters
    n_students = 60
    n_problems = 24
    n_blocks = 3
    
    # CSV header
    fieldnames = [
        'seed', 'sigma', 'correlation', 'estimator',
        'corr_theta', 'spear_theta', 
        'corr_beta', 'spear_beta',
        'corr_X_missing', 'rmse_theta', 'rmse_beta', 'rmse_X'
    ]
    
    total_runs = len(sigmas) * len(correlations) * len(estimator_names) * n_seeds
    print(f"Starting experiment suite: {total_runs} total runs")
    print(f"Output file: {output_file}")
    print("=" * 70)
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        run_count = 0
        for sigma in sigmas:
            for corr in correlations:
                print(f"\nSigma={sigma}, Correlation={corr}")
                
                for seed in range(n_seeds):
                    # Generate data
                    dg = DataGenerator.from_simulation(
                        n_students=n_students,
                        n_problems=n_problems,
                        seed=seed,
                        sigma=sigma,
                        problem_type='block',
                        n_blocks=n_blocks
                    ).apply_missing(
                        pattern='block',
                        correlation=corr,
                        correlation_strength=1.0,
                        n_blocks=n_blocks
                    )
                    
                    X_miss = dg.X_missing
                    X_true = dg.X_complete
                    theta_true = dg.theta_true
                    beta_true = dg.beta_true
                    groups = dg.groups
                    mask_miss = np.isnan(X_miss)
                    
                    # Center true parameters for fair comparison
                    theta_true_centered = theta_true - np.mean(theta_true)
                    beta_true_centered = beta_true - np.mean(beta_true)
                    
                    for est_name in estimator_names:
                        # Run estimator
                        if est_name == 'EM':
                            est = Estimator.em(X_miss, reg_lambda=0.0)
                        elif est_name == 'Day':
                            est = Estimator.day_linking(X_miss, groups)
                        else:  # 'Imp'
                            est = Estimator.mean_imputation(X_miss)
                        
                        theta_est = est.theta
                        beta_est = est.beta
                        X_imputed = est.X_imputed
                        
                        # Center estimates
                        theta_est_centered = theta_est - np.mean(theta_est)
                        beta_est_centered = beta_est - np.mean(beta_est)
                        
                        # Calculate metrics
                        # Correlations
                        corr_theta = np.corrcoef(theta_est, theta_true)[0, 1]
                        spear_theta = spearmanr(theta_est, theta_true).correlation
                        corr_beta = np.corrcoef(beta_est, beta_true)[0, 1]
                        spear_beta = spearmanr(beta_est, beta_true).correlation
                        corr_X = np.corrcoef(X_imputed[mask_miss], X_true[mask_miss])[0, 1]
                        
                        # RMSE (on centered parameters)
                        rmse_theta = np.sqrt(np.mean((theta_est_centered - theta_true_centered)**2))
                        rmse_beta = np.sqrt(np.mean((beta_est_centered - beta_true_centered)**2))
                        rmse_X = np.sqrt(np.mean((X_imputed - X_true)**2))
                        
                        # Write row
                        row = {
                            'seed': seed,
                            'sigma': sigma,
                            'correlation': corr,
                            'estimator': est_name,
                            'corr_theta': round(corr_theta, 6),
                            'spear_theta': round(spear_theta, 6),
                            'corr_beta': round(corr_beta, 6),
                            'spear_beta': round(spear_beta, 6),
                            'corr_X_missing': round(corr_X, 6),
                            'rmse_theta': round(rmse_theta, 6),
                            'rmse_beta': round(rmse_beta, 6),
                            'rmse_X': round(rmse_X, 6)
                        }
                        writer.writerow(row)
                        run_count += 1
                
                # Progress update
                pct = 100 * run_count / total_runs
                print(f"  Progress: {run_count}/{total_runs} ({pct:.1f}%)")
    
    print("\n" + "=" * 70)
    print(f"Experiment complete. Results saved to: {output_file}")
    print(f"Total runs: {run_count}")


def print_summary(output_file: str = 'results/experiment_results.csv'):
    """Print summary statistics from experiment results."""
    import pandas as pd
    
    df = pd.read_csv(output_file)
    
    print("\n" + "=" * 80)
    print("SUMMARY: Mean metrics by (sigma, correlation, estimator)")
    print("=" * 80)
    
    summary = df.groupby(['sigma', 'correlation', 'estimator']).agg({
        'corr_theta': 'mean',
        'spear_theta': 'mean',
        'corr_beta': 'mean',
        'spear_beta': 'mean',
        'corr_X_missing': 'mean',
        'rmse_theta': 'mean',
        'rmse_beta': 'mean',
        'rmse_X': 'mean'
    }).round(4)
    
    print(summary.to_string())
    
    return summary


if __name__ == "__main__":
    run_experiments()
    print_summary()
