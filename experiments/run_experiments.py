"""
Comprehensive experiment suite for evaluating estimators under different conditions.

NOTE: This script expects to be run from the project root directory, or it will
change to the project root automatically.
"""

import os
import sys
from pathlib import Path

# Add parent directory (for estimator) and current directory (for data_generator) to path
PROJECT_ROOT = Path(__file__).parent.parent
EXPERIMENTS_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXPERIMENTS_DIR))
os.chdir(PROJECT_ROOT)

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
    estimator_names = ['EM', 'DayAvg', 'Imp']
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
                            est = Estimator.em(X_miss, lambda_theta=0.0, lambda_beta=0.0,
                                               min_mark=0., max_mark=6.)
                        elif est_name == 'DayAvg':
                            est = Estimator.day_average(X_miss, n_blocks=n_blocks,
                                                        min_mark=0., max_mark=6.)
                        else:  # 'Imp'
                            est = Estimator.mean_imputation(X_miss, min_mark=0., max_mark=6.)
                        
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


def run_regularization_experiment(output_file: str = 'results/regularization_results.csv'):
    """
    Find optimal regularization parameter for each noise level.
    
    Uses only EM estimator with uncorrelated missing data.
    Tests a range of lambda values (same for theta and beta).
    
    Output: CSV with one row per (seed, sigma, lambda)
    """
    
    # Experimental factors
    sigmas = [0.5, 1.5, 3.5, 6.0]
    lambdas = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    n_seeds = 50
    
    # Fixed parameters
    n_students = 60
    n_problems = 24
    n_blocks = 3
    
    # CSV header
    fieldnames = [
        'seed', 'sigma', 'lambda',
        'corr_theta', 'spear_theta', 
        'corr_beta', 'spear_beta',
        'corr_X_missing', 'rmse_theta', 'rmse_beta', 'rmse_X'
    ]
    
    total_runs = len(sigmas) * len(lambdas) * n_seeds
    print(f"Starting regularization experiment: {total_runs} total runs")
    print(f"Output file: {output_file}")
    print("=" * 70)
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        run_count = 0
        for sigma in sigmas:
            print(f"\nSigma={sigma}")
            
            for seed in range(n_seeds):
                # Generate data (uncorrelated missing only)
                dg = DataGenerator.from_simulation(
                    n_students=n_students,
                    n_problems=n_problems,
                    seed=seed,
                    sigma=sigma,
                    problem_type='block',
                    n_blocks=n_blocks
                ).apply_missing(
                    pattern='block',
                    correlation='none',
                    correlation_strength=1.0,
                    n_blocks=n_blocks
                )
                
                X_miss = dg.X_missing
                X_true = dg.X_complete
                theta_true = dg.theta_true
                beta_true = dg.beta_true
                mask_miss = np.isnan(X_miss)
                
                # Center true parameters for fair comparison
                theta_true_centered = theta_true - np.mean(theta_true)
                beta_true_centered = beta_true - np.mean(beta_true)
                
                for lam in lambdas:
                    # Run EM with this lambda (same for theta and beta)
                    est = Estimator.em(X_miss, lambda_theta=lam, lambda_beta=lam,
                                       min_mark=0., max_mark=6.)
                    
                    theta_est = est.theta
                    beta_est = est.beta
                    X_imputed = est.X_imputed
                    
                    # Center estimates
                    theta_est_centered = theta_est - np.mean(theta_est)
                    beta_est_centered = beta_est - np.mean(beta_est)
                    
                    # Calculate metrics
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
                        'lambda': lam,
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
    print(f"Regularization experiment complete. Results saved to: {output_file}")
    print(f"Total runs: {run_count}")
    
    # Print summary of optimal lambdas
    print_regularization_summary(output_file)


def print_regularization_summary(input_file: str = 'results/regularization_results.csv'):
    """Print summary showing optimal lambda for each sigma."""
    import pandas as pd
    
    df = pd.read_csv(input_file)
    
    print("\n" + "=" * 80)
    print("OPTIMAL REGULARIZATION BY NOISE LEVEL")
    print("=" * 80)
    
    # Aggregate by (sigma, lambda)
    summary = df.groupby(['sigma', 'lambda']).agg({
        'corr_theta': 'mean',
        'corr_beta': 'mean',
        'corr_X_missing': 'mean',
        'rmse_theta': 'mean',
        'rmse_beta': 'mean',
        'rmse_X': 'mean'
    }).round(4)
    
    summary = summary.reset_index()
    
    # Find optimal lambda for each sigma (by different criteria)
    sigmas = df['sigma'].unique()
    
    print("\nOptimal λ by metric:")
    print("-" * 60)
    print(f"{'σ':>6} | {'corr_θ':>10} | {'corr_β':>10} | {'corr_X':>10} | {'RMSE_X':>10}")
    print("-" * 60)
    
    for sigma in sorted(sigmas):
        subset = summary[summary['sigma'] == sigma]
        
        # Best lambda for each metric
        best_corr_theta = subset.loc[subset['corr_theta'].idxmax(), 'lambda']
        best_corr_beta = subset.loc[subset['corr_beta'].idxmax(), 'lambda']
        best_corr_X = subset.loc[subset['corr_X_missing'].idxmax(), 'lambda']
        best_rmse_X = subset.loc[subset['rmse_X'].idxmin(), 'lambda']
        
        print(f"{sigma:>6.1f} | {best_corr_theta:>10.1f} | {best_corr_beta:>10.1f} | {best_corr_X:>10.1f} | {best_rmse_X:>10.1f}")
    
    print("-" * 60)
    
    return summary


def run_missing_rate_experiment(sigma: float = 3.5,
                                output_file: str = 'results/missing_rate_results.csv'):
    """
    Find optimal regularization for varying missing data proportions.
    
    Uses only EM estimator with configurable sigma.
    Starts with 33% block missing, adds increasing scattered missing up to ~80% total.
    Tests a range of lambda values (same for theta and beta).
    
    Args:
        sigma: Noise standard deviation (default 3.5)
        output_file: Path for output CSV
    
    Output: CSV with one row per (seed, additional_missing_rate, lambda)
    """
    
    # Fixed parameters
    n_students = 60
    n_problems = 24
    n_blocks = 3
    n_seeds = 50
    
    # Experimental factors
    # additional_missing_rates: 0.0 means just the block pattern (~33%)
    # Higher values add more scattered missing on top (up to ~80% total)
    additional_missing_rates = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
    lambdas = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    
    # CSV header
    fieldnames = [
        'seed', 'sigma', 'additional_missing_rate', 'total_missing_rate', 'lambda',
        'corr_theta', 'spear_theta', 
        'corr_beta', 'spear_beta',
        'rmse_theta', 'rmse_beta'
    ]
    
    total_runs = len(additional_missing_rates) * len(lambdas) * n_seeds
    print(f"Starting missing rate experiment: {total_runs} total runs")
    print(f"Fixed sigma={sigma}")
    print(f"Output file: {output_file}")
    print("=" * 70)
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        run_count = 0
        for add_miss in additional_missing_rates:
            # Determine pattern to use
            if add_miss == 0.0:
                pattern = 'block'
            else:
                pattern = 'both'
            
            print(f"\nAdditional missing rate={add_miss:.0%}")
            
            for seed in range(n_seeds):
                # Generate data with block + optional scattered missing
                dg = DataGenerator.from_simulation(
                    n_students=n_students,
                    n_problems=n_problems,
                    seed=seed,
                    sigma=sigma,
                    problem_type='block',
                    n_blocks=n_blocks
                ).apply_missing(
                    pattern=pattern,
                    correlation='none',
                    missing_rate=add_miss,  # Additional scattered missing
                    correlation_strength=0.0,
                    n_blocks=n_blocks
                )
                
                X_miss = dg.X_missing
                X_true = dg.X_complete
                theta_true = dg.theta_true
                beta_true = dg.beta_true
                mask_miss = np.isnan(X_miss)
                total_missing_rate = dg.missing_rate_actual
                
                # Center true parameters for fair comparison
                theta_true_centered = theta_true - np.mean(theta_true)
                beta_true_centered = beta_true - np.mean(beta_true)
                
                for lam in lambdas:
                    # Run EM with this lambda
                    est = Estimator.em(X_miss, lambda_theta=lam, lambda_beta=lam,
                                       min_mark=0., max_mark=6.)
                    
                    theta_est = est.theta
                    beta_est = est.beta
                    X_imputed = est.X_imputed
                    
                    # Center estimates
                    theta_est_centered = theta_est - np.mean(theta_est)
                    beta_est_centered = beta_est - np.mean(beta_est)
                    
                    # Calculate metrics
                    corr_theta = np.corrcoef(theta_est, theta_true)[0, 1]
                    spear_theta = spearmanr(theta_est, theta_true).correlation
                    corr_beta = np.corrcoef(beta_est, beta_true)[0, 1]
                    spear_beta = spearmanr(beta_est, beta_true).correlation
                    
                    # RMSE (on centered parameters)
                    rmse_theta = np.sqrt(np.mean((theta_est_centered - theta_true_centered)**2))
                    rmse_beta = np.sqrt(np.mean((beta_est_centered - beta_true_centered)**2))
                    
                    # Write row
                    row = {
                        'seed': seed,
                        'sigma': sigma,
                        'additional_missing_rate': add_miss,
                        'total_missing_rate': round(total_missing_rate, 4),
                        'lambda': lam,
                        'corr_theta': round(corr_theta, 6),
                        'spear_theta': round(spear_theta, 6),
                        'corr_beta': round(corr_beta, 6),
                        'spear_beta': round(spear_beta, 6),
                        'rmse_theta': round(rmse_theta, 6),
                        'rmse_beta': round(rmse_beta, 6)
                    }
                    writer.writerow(row)
                    run_count += 1
            
            # Progress update
            pct = 100 * run_count / total_runs
            print(f"  Progress: {run_count}/{total_runs} ({pct:.1f}%)")
    
    print("\n" + "=" * 70)
    print(f"Missing rate experiment complete. Results saved to: {output_file}")
    print(f"Total runs: {run_count}")
    
    # Print summary
    print_missing_rate_summary(output_file)


def print_missing_rate_summary(input_file: str = 'results/missing_rate_results.csv'):
    """Print summary showing optimal lambda for each missing rate."""
    import pandas as pd
    
    df = pd.read_csv(input_file)
    
    print("\n" + "=" * 80)
    print("OPTIMAL REGULARIZATION BY MISSING DATA PROPORTION")
    print("=" * 80)
    
    # Get average total missing rate for each additional_missing_rate
    missing_rate_map = df.groupby('additional_missing_rate')['total_missing_rate'].mean().to_dict()
    
    # Aggregate by (additional_missing_rate, lambda)
    summary = df.groupby(['additional_missing_rate', 'lambda']).agg({
        'total_missing_rate': 'mean',
        'corr_theta': 'mean',
        'corr_beta': 'mean',
        'rmse_theta': 'mean',
        'rmse_beta': 'mean'
    }).round(4)
    
    summary = summary.reset_index()
    
    # Find optimal lambda for each missing rate
    additional_rates = sorted(df['additional_missing_rate'].unique())
    
    print("\nOptimal λ by metric:")
    print("-" * 65)
    print(f"{'Add.Miss':>8} | {'Total':>6} | {'corr_θ':>10} | {'corr_β':>10} | {'RMSE_θ':>10} | {'RMSE_β':>10}")
    print("-" * 65)
    
    for add_rate in additional_rates:
        subset = summary[summary['additional_missing_rate'] == add_rate]
        total_rate = missing_rate_map[add_rate]
        
        # Best lambda for each metric
        best_corr_theta = subset.loc[subset['corr_theta'].idxmax(), 'lambda']
        best_corr_beta = subset.loc[subset['corr_beta'].idxmax(), 'lambda']
        best_rmse_theta = subset.loc[subset['rmse_theta'].idxmin(), 'lambda']
        best_rmse_beta = subset.loc[subset['rmse_beta'].idxmin(), 'lambda']
        
        print(f"{add_rate:>8.0%} | {total_rate:>5.1%} | {best_corr_theta:>10.1f} | {best_corr_beta:>10.1f} | {best_rmse_theta:>10.1f} | {best_rmse_beta:>10.1f}")
    
    print("-" * 65)
    
    return summary


def generate_report(input_file: str = 'results/experiment_results.csv',
                    output_file: str = 'results/experiment_report.md'):
    """
    Generate a nicely formatted markdown report from experiment results.
    
    Args:
        input_file: Path to the experiment results CSV
        output_file: Path for the output markdown report
    """
    import pandas as pd
    
    df = pd.read_csv(input_file)
    
    # Aggregate by (sigma, correlation, estimator)
    summary = df.groupby(['sigma', 'correlation', 'estimator']).agg({
        'corr_theta': ['mean', 'std'],
        'spear_theta': ['mean', 'std'],
        'corr_beta': ['mean', 'std'],
        'spear_beta': ['mean', 'std'],
        'corr_X_missing': ['mean', 'std'],
        'rmse_theta': ['mean', 'std'],
        'rmse_beta': ['mean', 'std'],
        'rmse_X': ['mean', 'std']
    }).round(4)
    
    # Flatten column names
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()
    
    # Build the markdown report
    lines = []
    
    # Header
    lines.append("# Experiment Results Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Executive Summary
    lines.append("## Executive Summary")
    lines.append("")
    lines.append("This report presents the results of comparing three estimators for the additive model:")
    lines.append("")
    lines.append("- **EM**: Regularized EM (Alternating Least Squares) with L2 penalties")
    lines.append("- **Day**: Day-linking heuristic estimator")
    lines.append("- **Imp**: Additive mean imputation")
    lines.append("")
    lines.append("The estimators are evaluated under varying noise levels (σ) and missing data correlation patterns.")
    lines.append("")
    
    # Key Findings
    lines.append("### Key Findings")
    lines.append("")
    
    # Find best estimator for each metric overall
    metrics = ['corr_theta_mean', 'corr_beta_mean', 'corr_X_missing_mean']
    metric_labels = {
        'corr_theta_mean': 'θ recovery (correlation)',
        'corr_beta_mean': 'β recovery (correlation)', 
        'corr_X_missing_mean': 'Missing value imputation'
    }
    
    for metric in metrics:
        if 'rmse' in metric:
            best_idx = summary.groupby('estimator')[metric].mean().idxmin()
        else:
            best_idx = summary.groupby('estimator')[metric].mean().idxmax()
        best_val = summary.groupby('estimator')[metric].mean()[best_idx]
        lines.append(f"- **{metric_labels[metric]}**: {best_idx} ({best_val:.4f} average)")
    
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Detailed Results by Correlation Type
    lines.append("## Results by Correlation Type")
    lines.append("")
    
    correlations = df['correlation'].unique()
    
    for corr_type in correlations:
        lines.append(f"### Missing Pattern: `{corr_type}`")
        lines.append("")
        
        if corr_type == 'none':
            lines.append("*Missing values are independent of student ability and problem difficulty.*")
        elif corr_type == 'theta':
            lines.append("*Missing values are correlated with student ability (θ).*")
        elif corr_type == 'beta':
            lines.append("*Missing values are correlated with problem difficulty (β).*")
        elif corr_type == 'both':
            lines.append("*Missing values are correlated with both θ and β.*")
        lines.append("")
        
        subset = summary[summary['correlation'] == corr_type]
        
        # Create table for this correlation type
        lines.append("| σ | Estimator | r(θ) | r(β) | r(X̂) | RMSE(θ) | RMSE(β) | RMSE(X̂) |")
        lines.append("|---:|:----------|-----:|-----:|-----:|--------:|--------:|--------:|")
        
        for _, row in subset.iterrows():
            lines.append(
                f"| {row['sigma']} | {row['estimator']} | "
                f"{row['corr_theta_mean']:.3f} | {row['corr_beta_mean']:.3f} | "
                f"{row['corr_X_missing_mean']:.3f} | {row['rmse_theta_mean']:.3f} | "
                f"{row['rmse_beta_mean']:.3f} | {row['rmse_X_mean']:.3f} |"
            )
        
        lines.append("")
    
    lines.append("---")
    lines.append("")
    
    # Summary Table: Best Estimator per Condition
    lines.append("## Winner Summary")
    lines.append("")
    lines.append("Best estimator for θ recovery (by Pearson correlation):")
    lines.append("")
    lines.append("| σ | none | theta | beta | both |")
    lines.append("|---:|:----:|:-----:|:----:|:----:|")
    
    sigmas = sorted(df['sigma'].unique())
    for sigma in sigmas:
        row_parts = [f"| {sigma}"]
        for corr_type in ['none', 'theta', 'beta', 'both']:
            subset = summary[(summary['sigma'] == sigma) & (summary['correlation'] == corr_type)]
            if len(subset) > 0:
                best_idx = subset['corr_theta_mean'].idxmax()
                best_est = subset.loc[best_idx, 'estimator']
                best_val = subset.loc[best_idx, 'corr_theta_mean']
                row_parts.append(f" **{best_est}** ({best_val:.3f})")
            else:
                row_parts.append(" - ")
        row_parts.append("|")
        lines.append(" |".join(row_parts))
    
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Experimental Setup
    lines.append("## Experimental Setup")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|:----------|------:|")
    lines.append(f"| Seeds | {df['seed'].nunique()} |")
    lines.append(f"| Noise levels (σ) | {', '.join(map(str, sigmas))} |")
    lines.append(f"| Correlation patterns | {', '.join(correlations)} |")
    lines.append(f"| Total runs | {len(df)} |")
    lines.append("")
    
    # Metrics Legend
    lines.append("### Metrics")
    lines.append("")
    lines.append("- **r(θ)**: Pearson correlation between estimated and true student abilities")
    lines.append("- **r(β)**: Pearson correlation between estimated and true problem difficulties")
    lines.append("- **r(X̂)**: Pearson correlation between imputed and true values (for missing entries only)")
    lines.append("- **RMSE(θ)**: Root mean squared error for student abilities (centered)")
    lines.append("- **RMSE(β)**: Root mean squared error for problem difficulties (centered)")
    lines.append("- **RMSE(X̂)**: Root mean squared error for imputed matrix")
    lines.append("")
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"Report generated: {output_file}")
    return output_file


def generate_comprehensive_report(exp1_file: str = 'results/experiment_results.csv',
                                 output_file: str = 'results/comprehensive_report.md'):
    """
    Generate a comprehensive markdown report summarizing all three experiments.
    
    Experiments:
    1. Three estimators comparison (EM, DayAvg, Imp)
    2. Regularization by sigma (noise level)
    3. Regularization by missing data proportion
    
    Primary metrics: corr(θ), ρ(θ) [rank correlation]
    Secondary metrics: corr(β), ρ(β), RMSE(X̂)
    """
    import pandas as pd
    from pathlib import Path
    
    results_dir = Path('results')
    lines = []
    
    # Header
    lines.append("# Comprehensive Experiment Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Introduction
    lines.append("## Overview")
    lines.append("")
    lines.append("This report summarizes three sets of experiments evaluating parameter estimation")
    lines.append("for the additive model: **X_ij = μ + θ_i + β_j + ε_ij**")
    lines.append("")
    lines.append("### Primary Metrics (Student Ability Recovery)")
    lines.append("- **r(θ)**: Pearson correlation between estimated and true student abilities")
    lines.append("- **ρ(θ)**: Spearman rank correlation between estimated and true student abilities")
    lines.append("")
    lines.append("### Secondary Metrics")
    lines.append("- **r(β)**: Pearson correlation for problem difficulties")
    lines.append("- **ρ(β)**: Spearman rank correlation for problem difficulties")
    lines.append("- **RMSE(X̂)**: Root mean squared error for imputed matrix")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # =========================================================================
    # EXPERIMENT 1: Three Estimators
    # =========================================================================
    lines.append("## Experiment 1: Estimator Comparison")
    lines.append("")
    lines.append("Compares three estimation methods across different noise levels and missing data patterns.")
    lines.append("")
    lines.append("### Estimators")
    lines.append("- **EM**: Regularized EM (Alternating Least Squares) with L2 penalties")
    lines.append("- **DayAvg**: Day-average heuristic (rescales days to common mean)")
    lines.append("- **Imp**: Additive mean imputation")
    lines.append("")
    
    exp1_path = Path(exp1_file)
    if exp1_path.exists():
        df1 = pd.read_csv(exp1_path)
        
        # Summary by estimator (overall)
        overall = df1.groupby('estimator').agg({
            'corr_theta': 'mean',
            'spear_theta': 'mean',
            'corr_beta': 'mean',
            'spear_beta': 'mean',
            'rmse_X': 'mean'
        }).round(4)
        
        lines.append("### Overall Performance (averaged across all conditions)")
        lines.append("")
        lines.append("| Estimator | r(θ) | ρ(θ) | r(β) | ρ(β) | RMSE(X̂) |")
        lines.append("|:----------|-----:|-----:|-----:|-----:|--------:|")
        for est in ['EM', 'DayAvg', 'Imp']:
            if est in overall.index:
                r = overall.loc[est]
                lines.append(f"| {est} | {r['corr_theta']:.3f} | {r['spear_theta']:.3f} | "
                           f"{r['corr_beta']:.3f} | {r['spear_beta']:.3f} | {r['rmse_X']:.3f} |")
        lines.append("")
        
        # Best estimator by condition
        lines.append("### Best Estimator for θ Recovery by Condition")
        lines.append("")
        lines.append("| σ | Corr Pattern | Best (r(θ)) | Value |")
        lines.append("|---:|:-------------|:------------|------:|")
        
        summary1 = df1.groupby(['sigma', 'correlation', 'estimator']).agg({
            'corr_theta': 'mean'
        }).reset_index()
        
        for sigma in sorted(df1['sigma'].unique()):
            for corr in ['none', 'theta', 'beta', 'both']:
                subset = summary1[(summary1['sigma'] == sigma) & (summary1['correlation'] == corr)]
                if len(subset) > 0:
                    best_idx = subset['corr_theta'].idxmax()
                    best = subset.loc[best_idx]
                    lines.append(f"| {sigma} | {corr} | **{best['estimator']}** | {best['corr_theta']:.3f} |")
        lines.append("")
    else:
        lines.append(f"*Experiment 1 results ({exp1_file}) not found.*")
        lines.append("")
    
    lines.append("---")
    lines.append("")
    
    # =========================================================================
    # EXPERIMENT 2: Regularization by Sigma
    # =========================================================================
    lines.append("## Experiment 2: Optimal Regularization by Noise Level")
    lines.append("")
    lines.append("Finds optimal L2 regularization parameter (λ) for varying noise levels (σ).")
    lines.append("Uses EM estimator with uncorrelated missing data (~33% block pattern).")
    lines.append("")
    lines.append("**Objective:** Minimize sum((X_ij - θ_i - β_j)²) + λ_θ·sum(θ_i²) + λ_β·sum(β_j²)")
    lines.append("")
    
    exp2_file = results_dir / 'regularization_results.csv'
    if exp2_file.exists():
        df2 = pd.read_csv(exp2_file)
        
        # Group by sigma and lambda
        summary2 = df2.groupby(['sigma', 'lambda']).agg({
            'corr_theta': 'mean',
            'spear_theta': 'mean',
            'corr_beta': 'mean',
            'spear_beta': 'mean',
            'rmse_X': 'mean'
        }).round(4).reset_index()
        
        lines.append("### Optimal λ by Noise Level")
        lines.append("")
        lines.append("| σ | Best λ (r(θ)) | r(θ) | Best λ (ρ(θ)) | ρ(θ) | Best λ (r(β)) | r(β) |")
        lines.append("|---:|--------------:|-----:|--------------:|-----:|--------------:|-----:|")
        
        for sigma in sorted(df2['sigma'].unique()):
            subset = summary2[summary2['sigma'] == sigma]
            
            best_corr_theta_idx = subset['corr_theta'].idxmax()
            best_corr_theta_lam = subset.loc[best_corr_theta_idx, 'lambda']
            best_corr_theta_val = subset.loc[best_corr_theta_idx, 'corr_theta']
            
            best_spear_theta_idx = subset['spear_theta'].idxmax()
            best_spear_theta_lam = subset.loc[best_spear_theta_idx, 'lambda']
            best_spear_theta_val = subset.loc[best_spear_theta_idx, 'spear_theta']
            
            best_corr_beta_idx = subset['corr_beta'].idxmax()
            best_corr_beta_lam = subset.loc[best_corr_beta_idx, 'lambda']
            best_corr_beta_val = subset.loc[best_corr_beta_idx, 'corr_beta']
            
            lines.append(f"| {sigma} | {best_corr_theta_lam:.1f} | {best_corr_theta_val:.3f} | "
                        f"{best_spear_theta_lam:.1f} | {best_spear_theta_val:.3f} | "
                        f"{best_corr_beta_lam:.1f} | {best_corr_beta_val:.3f} |")
        
        lines.append("")
        
        # Performance improvement table
        lines.append("### Performance: λ=0 vs Optimal λ")
        lines.append("")
        lines.append("| σ | r(θ) at λ=0 | r(θ) at λ* | Δ | ρ(θ) at λ=0 | ρ(θ) at λ* | Δ |")
        lines.append("|---:|------------:|-----------:|--:|------------:|-----------:|--:|")
        
        for sigma in sorted(df2['sigma'].unique()):
            subset = summary2[summary2['sigma'] == sigma]
            
            zero_row = subset[subset['lambda'] == 0.0].iloc[0] if len(subset[subset['lambda'] == 0.0]) > 0 else None
            best_corr_idx = subset['corr_theta'].idxmax()
            best_spear_idx = subset['spear_theta'].idxmax()
            
            if zero_row is not None:
                corr_0 = zero_row['corr_theta']
                corr_best = subset.loc[best_corr_idx, 'corr_theta']
                delta_corr = corr_best - column_corr if 'column_corr' in locals() else corr_best - corr_0 # Fix for delta calc
                delta_corr = corr_best - corr_0
                
                spear_0 = zero_row['spear_theta']
                spear_best = subset.loc[best_spear_idx, 'spear_theta']
                delta_spear = spear_best - spear_0
                
                lines.append(f"| {sigma} | {corr_0:.4f} | {corr_best:.4f} | {delta_corr:+.4f} | "
                            f"{spear_0:.4f} | {spear_best:.4f} | {delta_spear:+.4f} |")
        
        lines.append("")
    else:
        lines.append("*Experiment 2 results not found.*")
        lines.append("")
    
    lines.append("---")
    lines.append("")
    
    # =========================================================================
    # EXPERIMENT 3: Regularization by Missing Rate
    # =========================================================================
    lines.append("## Experiment 3: Optimal Regularization by Missing Data Proportion")
    lines.append("")
    lines.append("Finds optimal L2 regularization parameter (λ) for varying amounts of missing data.")
    lines.append("Uses EM estimator with fixed σ=3.5 and uncorrelated missing.")
    lines.append("")
    
    exp3_file = results_dir / 'missing_rate_results.csv'
    if exp3_file.exists():
        df3 = pd.read_csv(exp3_file)
        
        # Get sigma from data
        sigma_used = df3['sigma'].iloc[0] if 'sigma' in df3.columns else 3.5
        lines.append(f"**Fixed noise level:** σ = {sigma_used}")
        lines.append("")
        
        # Get total missing rate mapping
        missing_map = df3.groupby('additional_missing_rate')['total_missing_rate'].mean().to_dict()
        
        # Group by additional_missing_rate and lambda
        agg_cols = {
            'total_missing_rate': 'mean',
            'corr_theta': 'mean',
            'spear_theta': 'mean',
            'corr_beta': 'mean',
            'spear_beta': 'mean'
        }
        if 'rmse_X' in df3.columns:
            agg_cols['rmse_X'] = 'mean'
        
        summary3 = df3.groupby(['additional_missing_rate', 'lambda']).agg(agg_cols).round(4).reset_index()
        
        lines.append("### Optimal λ by Missing Data Proportion")
        lines.append("")
        lines.append("| Add. Miss | Total Miss | Best λ (r(θ)) | r(θ) | Best λ (ρ(θ)) | ρ(θ) |")
        lines.append("|----------:|-----------:|--------------:|-----:|--------------:|-----:|")
        
        for add_rate in sorted(df3['additional_missing_rate'].unique()):
            subset = summary3[summary3['additional_missing_rate'] == add_rate]
            total_rate = missing_map[add_rate]
            
            best_corr_idx = subset['corr_theta'].idxmax()
            best_corr_lam = subset.loc[best_corr_idx, 'lambda']
            best_corr_val = subset.loc[best_corr_idx, 'corr_theta']
            
            best_spear_idx = subset['spear_theta'].idxmax()
            best_spear_lam = subset.loc[best_spear_idx, 'lambda']
            best_spear_val = subset.loc[best_spear_idx, 'spear_theta']
            
            lines.append(f"| {add_rate:.0%} | {total_rate:.1%} | {best_corr_lam:.1f} | {best_corr_val:.3f} | "
                        f"{best_spear_lam:.1f} | {best_spear_val:.3f} |")
        
        lines.append("")
        
        # Performance degradation with missing data
        lines.append("### Performance at Optimal λ vs Missing Data")
        lines.append("")
        lines.append("| Total Missing | r(θ) | ρ(θ) | r(β) | ρ(β) |")
        lines.append("|--------------:|-----:|-----:|-----:|-----:|")
        
        for add_rate in sorted(df3['additional_missing_rate'].unique()):
            subset = summary3[summary3['additional_missing_rate'] == add_rate]
            total_rate = missing_map[add_rate]
            
            best_corr_idx = subset['corr_theta'].idxmax()
            best_row = subset.loc[best_corr_idx]
            
            lines.append(f"| {total_rate:.1%} | {best_row['corr_theta']:.3f} | {best_row['spear_theta']:.3f} | "
                        f"{best_row['corr_beta']:.3f} | {best_row['spear_beta']:.3f} |")
        
        lines.append("")
    else:
        lines.append("*Experiment 3 results not found.*")
        lines.append("")
    
    lines.append("---")
    lines.append("")
    
    # =========================================================================
    # Conclusions
    # =========================================================================
    lines.append("## Key Takeaways")
    lines.append("")
    lines.append("### Estimator Selection")
    lines.append("- **EM** achieves best θ recovery, β recovery, and overall imputation quality")
    lines.append("- **Day-average** is a strong heuristic for θ recovery but weaker for β difficulties")
    lines.append("- **Mean imputation** is a reliable baseline but less accurate than EM")
    lines.append("")
    lines.append("### Regularization Guidelines")
    lines.append("- **Low noise (σ ≤ 1.5)**: No regularization needed (λ = 0)")
    lines.append("- **Moderate noise (σ ≈ 3.5)**: Mild regularization helps as missing data increases")
    lines.append("- **High noise (σ ≥ 6.0)**: Regularization becomes beneficial even at 33% missing")
    lines.append("")
    lines.append("### Missing Data Impact")
    lines.append("- Performance degrades gracefully up to ~50% missing")
    lines.append("- Beyond 60% missing, regularization becomes increasingly important")
    lines.append("- β recovery is more sensitive to missing data than θ recovery")
    lines.append("")
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"Comprehensive report generated: {output_file}")
    return output_file


def run_convergence_study(output_file: str = 'results/convergence_study.csv'):
    """
    Study EM convergence rate across different sigma values and missing data proportions.
    
    Uses same parameter ranges as other experiments:
    - sigma: [0.5, 1.5, 3.5, 6.0]
    - additional_missing_rates: [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
    - seeds: 50
    
    Output: CSV with iteration counts for each (seed, sigma, missing_rate) combination
    """
    
    # Experimental factors (same as other experiments)
    sigmas = [0.5, 1.5, 3.5, 6.0]
    additional_missing_rates = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
    n_seeds = 50
    
    # Fixed parameters
    n_students = 60
    n_problems = 24
    n_blocks = 3
    
    # CSV header
    fieldnames = [
        'seed', 'sigma', 'additional_missing_rate', 'total_missing_rate', 'n_iterations'
    ]
    
    total_runs = len(sigmas) * len(additional_missing_rates) * n_seeds
    print(f"Starting EM convergence study: {total_runs} total runs")
    print(f"Output file: {output_file}")
    print("=" * 70)
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        run_count = 0
        for sigma in sigmas:
            for add_miss in additional_missing_rates:
                # Determine pattern to use
                if add_miss == 0.0:
                    pattern = 'block'
                else:
                    pattern = 'both'
                
                for seed in range(n_seeds):
                    # Generate data with block + optional scattered missing
                    dg = DataGenerator.from_simulation(
                        n_students=n_students,
                        n_problems=n_problems,
                        seed=seed,
                        sigma=sigma,
                        problem_type='block',
                        n_blocks=n_blocks
                    ).apply_missing(
                        pattern=pattern,
                        correlation='none',
                        missing_rate=add_miss,
                        correlation_strength=0.0,
                        n_blocks=n_blocks
                    )
                    
                    X_miss = dg.X_missing
                    total_missing_rate = dg.missing_rate_actual
                    
                    # Run EM with no regularization to get true convergence behavior
                    est = Estimator.em(X_miss, lambda_theta=0.0, lambda_beta=0.0,
                                       min_mark=0., max_mark=6.)
                    
                    # Write row
                    row = {
                        'seed': seed,
                        'sigma': sigma,
                        'additional_missing_rate': add_miss,
                        'total_missing_rate': round(total_missing_rate, 4),
                        'n_iterations': est.n_iterations
                    }
                    writer.writerow(row)
                    run_count += 1
            
            # Progress update after each sigma
            pct = 100 * run_count / total_runs
            print(f"  Sigma={sigma}: {run_count}/{total_runs} ({pct:.1f}%)")
    
    print("\n" + "=" * 70)
    print(f"Convergence study complete. Results saved to: {output_file}")
    print(f"Total runs: {run_count}")
    
    # Print summary
    print_convergence_summary(output_file)


def print_convergence_summary(input_file: str = 'results/convergence_study.csv'):
    """Print summary of EM convergence rates by sigma and missing rate."""
    import pandas as pd
    
    df = pd.read_csv(input_file)
    
    print("\n" + "=" * 80)
    print("EM CONVERGENCE RATE STUDY")
    print("=" * 80)
    
    # Aggregate by (sigma, additional_missing_rate)
    summary = df.groupby(['sigma', 'additional_missing_rate']).agg({
        'total_missing_rate': 'mean',
        'n_iterations': ['min', 'max', 'mean', 'std']
    }).round(2)
    
    summary.columns = ['total_miss_rate', 'iter_min', 'iter_max', 'iter_mean', 'iter_std']
    summary = summary.reset_index()
    
    print("\nIteration counts by (sigma, missing rate):")
    print("-" * 85)
    print(f"{'σ':>6} | {'Add.Miss':>8} | {'Tot.Miss':>8} | {'Min':>5} | {'Max':>5} | {'Mean':>7} | {'Std':>6}")
    print("-" * 85)
    
    for _, row in summary.iterrows():
        print(f"{row['sigma']:>6.1f} | {row['additional_missing_rate']:>7.0%} | "
              f"{row['total_miss_rate']:>7.1%} | {row['iter_min']:>5.0f} | "
              f"{row['iter_max']:>5.0f} | {row['iter_mean']:>7.1f} | {row['iter_std']:>6.2f}")
    
    print("-" * 85)
    
    # Summary by sigma only
    print("\nSummary by sigma (across all missing rates):")
    print("-" * 60)
    sigma_summary = df.groupby('sigma').agg({
        'n_iterations': ['min', 'max', 'mean']
    }).round(2)
    sigma_summary.columns = ['min', 'max', 'mean']
    
    print(f"{'σ':>6} | {'Range':<15} | {'Mean':>7}")
    print("-" * 60)
    for sigma in sorted(df['sigma'].unique()):
        row = sigma_summary.loc[sigma]
        print(f"{sigma:>6.1f} | {int(row['min']):>3} - {int(row['max']):<8} | {row['mean']:>7.1f}")
    
    print("-" * 60)
    
    # Summary by missing rate only
    print("\nSummary by missing rate (across all sigmas):")
    print("-" * 60)
    miss_summary = df.groupby('additional_missing_rate').agg({
        'total_missing_rate': 'mean',
        'n_iterations': ['min', 'max', 'mean']
    }).round(2)
    miss_summary.columns = ['total_miss', 'min', 'max', 'mean']
    
    print(f"{'Add.Miss':>8} | {'Tot.Miss':>8} | {'Range':<15} | {'Mean':>7}")
    print("-" * 60)
    for add_rate in sorted(df['additional_missing_rate'].unique()):
        row = miss_summary.loc[add_rate]
        print(f"{add_rate:>7.0%} | {row['total_miss']:>7.1%} | {int(row['min']):>3} - {int(row['max']):<8} | {row['mean']:>7.1f}")
    
    print("-" * 60)
    
    return summary


if __name__ == "__main__":
    run_experiments()
    print_summary()
    generate_report()
