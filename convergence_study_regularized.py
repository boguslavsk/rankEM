"""
Convergence study with regularization parameters.
Tests lambda values: 0.1, 1.0, 5.0
"""
import numpy as np
import csv
import pandas as pd
from data_generator import DataGenerator
from estimator import Estimator

def run_convergence_study_with_regularization(output_file: str = 'results/convergence_study_regularized.csv'):
    """
    Study EM convergence rate with different regularization parameters.
    
    Parameters:
    - sigma: [0.5, 1.5, 3.5, 6.0]
    - additional_missing_rates: [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
    - lambda: [0.1, 1.0, 5.0]
    - seeds: 50
    """
    
    # Experimental factors
    sigmas = [0.5, 1.5, 3.5, 6.0]
    additional_missing_rates = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
    lambdas = [0.1, 1.0, 5.0]
    n_seeds = 50
    
    # Fixed parameters
    n_students = 60
    n_problems = 24
    n_blocks = 3
    
    # CSV header
    fieldnames = [
        'seed', 'sigma', 'additional_missing_rate', 'total_missing_rate', 'lambda', 'n_iterations'
    ]
    
    total_runs = len(sigmas) * len(additional_missing_rates) * len(lambdas) * n_seeds
    print(f"Starting EM convergence study with regularization: {total_runs} total runs")
    print(f"Lambda values: {lambdas}")
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
                    
                    for lam in lambdas:
                        # Run EM with this regularization
                        est = Estimator.em(X_miss, lambda_theta=lam, lambda_beta=lam)
                        
                        # Write row
                        row = {
                            'seed': seed,
                            'sigma': sigma,
                            'additional_missing_rate': add_miss,
                            'total_missing_rate': round(total_missing_rate, 4),
                            'lambda': lam,
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
    
    # Analyze and print summary
    analyze_results(output_file)


def analyze_results(input_file: str = 'results/convergence_study_regularized.csv'):
    """Analyze and print convergence study results."""
    df = pd.read_csv(input_file)
    
    output_file = input_file.replace('.csv', '_summary.txt')
    
    with open(output_file, 'w') as f:
        def pr(s=''):
            print(s)
            f.write(s + '\n')
        
        pr('=' * 80)
        pr('EM CONVERGENCE RATE STUDY WITH REGULARIZATION')
        pr('=' * 80)
        pr()
        pr(f'Total runs: {len(df)}')
        pr(f'Sigmas: {sorted(df["sigma"].unique())}')
        pr(f'Additional missing rates: {sorted(df["additional_missing_rate"].unique())}')
        pr(f'Lambda values: {sorted(df["lambda"].unique())}')
        pr(f'Seeds per condition: {len(df["seed"].unique())}')
        pr()
        
        # Summary by lambda
        pr('=' * 80)
        pr('SUMMARY BY LAMBDA (across all sigmas and missing rates)')
        pr('=' * 80)
        lambda_summary = df.groupby('lambda')['n_iterations'].agg(['min', 'max', 'mean', 'std']).round(2)
        for lam, row in lambda_summary.iterrows():
            non_converged = len(df[(df['lambda']==lam) & (df['n_iterations']>=100)])
            pr(f"  lambda={lam}: iterations {int(row['min']):3} - {int(row['max']):3}, "
               f"mean={row['mean']:.1f}, std={row['std']:.2f}, non-converged={non_converged}")
        pr()
        
        # Summary by missing rate and lambda
        pr('=' * 80)
        pr('SUMMARY BY MISSING RATE AND LAMBDA')
        pr('=' * 80)
        pr()
        pr(f"{'Add.Miss':>8} | {'Tot.Miss':>8} | {'Lambda':>6} | {'Min':>4} | {'Max':>4} | {'Mean':>6} | {'NonConv':>7}")
        pr('-' * 70)
        
        for add_rate in sorted(df['additional_missing_rate'].unique()):
            for lam in sorted(df['lambda'].unique()):
                subset = df[(df['additional_missing_rate']==add_rate) & (df['lambda']==lam)]
                total_miss = subset['total_missing_rate'].mean()
                min_iter = int(subset['n_iterations'].min())
                max_iter = int(subset['n_iterations'].max())
                mean_iter = subset['n_iterations'].mean()
                non_conv = len(subset[subset['n_iterations']>=100])
                pr(f"{add_rate:>7.0%} | {total_miss:>7.1%} | {lam:>6.1f} | {min_iter:>4} | {max_iter:>4} | {mean_iter:>6.1f} | {non_conv:>7}")
            pr('-' * 70)
        pr()
        
        # Comparison table: mean iterations by (lambda, missing_rate)
        pr('=' * 80)
        pr('MEAN ITERATIONS BY (LAMBDA, MISSING_RATE) - Across all sigmas')
        pr('=' * 80)
        pr()
        
        pivot = df.groupby(['additional_missing_rate', 'lambda'])['n_iterations'].mean().unstack()
        pivot = pivot.round(1)
        
        # Header
        header = f"{'Add.Miss':>8} |"
        for lam in sorted(df['lambda'].unique()):
            header += f" λ={lam:>4} |"
        pr(header)
        pr('-' * (10 + 9 * len(df['lambda'].unique())))
        
        for add_rate in sorted(df['additional_missing_rate'].unique()):
            row_str = f"{add_rate:>7.0%} |"
            for lam in sorted(df['lambda'].unique()):
                val = pivot.loc[add_rate, lam]
                row_str += f" {val:>6.1f} |"
            pr(row_str)
        pr()
        
        # Key findings
        pr('=' * 80)
        pr('KEY FINDINGS')
        pr('=' * 80)
        pr()
        
        # Non-convergence by lambda
        pr("Non-convergence cases by lambda:")
        for lam in sorted(df['lambda'].unique()):
            non_conv = df[(df['lambda']==lam) & (df['n_iterations']>=100)]
            if len(non_conv) > 0:
                pr(f"  lambda={lam}: {len(non_conv)} cases")
                for (add_miss,), count in non_conv.groupby(['additional_missing_rate']).size().items():
                    pr(f"    at {add_miss:.0%} additional missing: {count} cases")
            else:
                pr(f"  lambda={lam}: 0 cases (all converged)")
        pr()
        
        # Typical convergence ranges
        converged = df[df['n_iterations'] < 100]
        pr("Typical convergence (excluding non-converged):")
        for lam in sorted(df['lambda'].unique()):
            subset = converged[converged['lambda']==lam]
            if len(subset) > 0:
                pr(f"  lambda={lam}: {int(subset['n_iterations'].min())}-{int(subset['n_iterations'].max())} iterations, mean={subset['n_iterations'].mean():.1f}")
    
    print(f"\nSummary saved to: {output_file}")


if __name__ == "__main__":
    run_convergence_study_with_regularization()
