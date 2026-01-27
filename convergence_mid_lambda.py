"""
Convergence study with intermediate regularization parameters.
Tests lambda values: 0.25, 0.5, 0.75
"""
import numpy as np
import csv
import pandas as pd
from data_generator import DataGenerator
from estimator import Estimator

def run_study():
    output_file = 'results/convergence_study_mid_lambda.csv'
    
    # Experimental factors
    sigmas = [0.5, 1.5, 3.5, 6.0]
    additional_missing_rates = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
    lambdas = [0.25, 0.5, 0.75]
    n_seeds = 50
    
    # Fixed parameters
    n_students = 60
    n_problems = 24
    n_blocks = 3
    
    fieldnames = [
        'seed', 'sigma', 'additional_missing_rate', 'total_missing_rate', 'lambda', 'n_iterations'
    ]
    
    total_runs = len(sigmas) * len(additional_missing_rates) * len(lambdas) * n_seeds
    print(f"Starting convergence study with lambdas {lambdas}: {total_runs} runs")
    print("=" * 70)
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        run_count = 0
        for sigma in sigmas:
            for add_miss in additional_missing_rates:
                pattern = 'block' if add_miss == 0.0 else 'both'
                
                for seed in range(n_seeds):
                    dg = DataGenerator.from_simulation(
                        n_students=n_students, n_problems=n_problems, seed=seed,
                        sigma=sigma, problem_type='block', n_blocks=n_blocks
                    ).apply_missing(
                        pattern=pattern, correlation='none',
                        missing_rate=add_miss, correlation_strength=0.0, n_blocks=n_blocks
                    )
                    
                    X_miss = dg.X_missing
                    total_missing_rate = dg.missing_rate_actual
                    
                    for lam in lambdas:
                        est = Estimator.em(X_miss, lambda_theta=lam, lambda_beta=lam)
                        
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
            
            pct = 100 * run_count / total_runs
            print(f"  Sigma={sigma}: {run_count}/{total_runs} ({pct:.1f}%)")
    
    print(f"\nResults saved to: {output_file}")
    
    # Analyze
    analyze_results(output_file)


def analyze_results(input_file):
    df = pd.read_csv(input_file)
    
    # Also load previous results for comparison
    df_prev = pd.read_csv('results/convergence_study_regularized.csv')
    df0 = pd.read_csv('results/convergence_study.csv')
    
    output_file = input_file.replace('.csv', '_summary.txt')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        def pr(s=''):
            print(s)
            f.write(s + '\n')
        
        pr('=' * 80)
        pr('EM CONVERGENCE STUDY: INTERMEDIATE LAMBDA VALUES')
        pr('=' * 80)
        pr()
        pr(f'Lambda values tested: 0.25, 0.5, 0.75')
        pr()
        
        # Overall by lambda
        pr('OVERALL SUMMARY BY LAMBDA')
        pr('-' * 60)
        for lam in [0.25, 0.5, 0.75]:
            subset = df[df['lambda'] == lam]
            non_conv = len(subset[subset['n_iterations'] >= 100])
            mean_iter = subset['n_iterations'].mean()
            max_iter = subset['n_iterations'].max()
            pr(f'  lam={lam}: range {int(subset["n_iterations"].min())}-{int(max_iter)}, '
               f'mean={mean_iter:.1f}, non-converged={non_conv}')
        pr()
        
        # Comparison table with all lambdas
        pr('COMPARISON: ALL LAMBDA VALUES')
        pr('=' * 80)
        pr()
        pr('Non-convergence count by (lambda, missing rate):')
        pr('-' * 80)
        
        all_lambdas = [0, 0.1, 0.25, 0.5, 0.75, 1.0, 5.0]
        header = f"{'Miss%':>6} |"
        for lam in all_lambdas:
            header += f" lam={lam:<4} |"
        pr(header)
        pr('-' * 80)
        
        for add_rate in sorted(df['additional_missing_rate'].unique()):
            row = f"{df[df['additional_missing_rate']==add_rate]['total_missing_rate'].mean():5.0%} |"
            
            for lam in all_lambdas:
                if lam == 0:
                    subset = df0[df0['additional_missing_rate']==add_rate]
                elif lam in [0.1, 1.0, 5.0]:
                    subset = df_prev[(df_prev['additional_missing_rate']==add_rate) & (df_prev['lambda']==lam)]
                else:
                    subset = df[(df['additional_missing_rate']==add_rate) & (df['lambda']==lam)]
                
                non_conv = len(subset[subset['n_iterations']>=100])
                row += f" {non_conv:>6} |"
            pr(row)
        pr()
        
        # Mean iterations
        pr('Mean iterations by (lambda, missing rate):')
        pr('-' * 80)
        pr(header)
        pr('-' * 80)
        
        for add_rate in sorted(df['additional_missing_rate'].unique()):
            row = f"{df[df['additional_missing_rate']==add_rate]['total_missing_rate'].mean():5.0%} |"
            
            for lam in all_lambdas:
                if lam == 0:
                    subset = df0[df0['additional_missing_rate']==add_rate]
                elif lam in [0.1, 1.0, 5.0]:
                    subset = df_prev[(df_prev['additional_missing_rate']==add_rate) & (df_prev['lambda']==lam)]
                else:
                    subset = df[(df['additional_missing_rate']==add_rate) & (df['lambda']==lam)]
                
                mean_iter = subset['n_iterations'].mean()
                row += f" {mean_iter:>6.1f} |"
            pr(row)
        pr()
        
        # Key finding
        pr('=' * 80)
        pr('KEY FINDING: Where does convergence become reliable?')
        pr('=' * 80)
        pr()
        
        for lam in [0.1, 0.25, 0.5, 0.75, 1.0]:
            if lam in [0.1, 1.0]:
                subset = df_prev[df_prev['lambda']==lam]
            else:
                subset = df[df['lambda']==lam]
            
            non_conv = len(subset[subset['n_iterations']>=100])
            pct = 100 * non_conv / len(subset)
            pr(f'  lam={lam}: {non_conv} failures ({pct:.1f}%)')
    
    print(f"\nSummary saved to: {output_file}")


if __name__ == "__main__":
    run_study()
