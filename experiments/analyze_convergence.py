"""Analyze and print convergence study results.

NOTE: Run from project root directory.
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)

import pandas as pd

df = pd.read_csv('results/convergence_study.csv')

# Open output file
with open('results/convergence_summary.txt', 'w') as f:
    def pr(s=''):
        print(s)
        f.write(s + '\n')
    
    pr('=' * 80)
    pr('EM CONVERGENCE RATE STUDY')
    pr('=' * 80)
    pr()
    pr(f'Total runs: {len(df)}')
    pr(f'Sigmas: {sorted(df["sigma"].unique())}')
    pr(f'Additional missing rates: {sorted(df["additional_missing_rate"].unique())}')
    pr(f'Seeds per condition: {len(df["seed"].unique())}')
    pr()
    
    # Summary by sigma
    pr('=' * 80)
    pr('SUMMARY BY SIGMA (across all missing rates)')
    pr('=' * 80)
    sigma_summary = df.groupby('sigma')['n_iterations'].agg(['min', 'max', 'mean', 'std']).round(2)
    for sigma, row in sigma_summary.iterrows():
        pr(f"  sigma={sigma:.1f}: iterations {int(row['min']):3} - {int(row['max']):3}, mean={row['mean']:.1f}, std={row['std']:.2f}")
    pr()
    
    # Summary by missing rate
    pr('=' * 80)
    pr('SUMMARY BY MISSING RATE (across all sigmas)')
    pr('=' * 80)
    miss_summary = df.groupby('additional_missing_rate').agg({
        'total_missing_rate': 'mean',
        'n_iterations': ['min', 'max', 'mean', 'std']
    }).round(2)
    miss_summary.columns = ['total_miss', 'min', 'max', 'mean', 'std']
    for add_rate, row in miss_summary.iterrows():
        pr(f"  add_miss={add_rate:4.0%}, total={row['total_miss']:5.1%}: "
           f"iterations {int(row['min']):3} - {int(row['max']):3}, mean={row['mean']:.1f}, std={row['std']:.2f}")
    pr()
    
    # Full grid
    pr('=' * 80)
    pr('FULL GRID: Iterations by (sigma, missing rate)')
    pr('=' * 80)
    pr()
    grid = df.groupby(['sigma', 'additional_missing_rate'])['n_iterations'].agg(['min', 'max', 'mean', 'std']).round(1)
    grid = grid.reset_index()
    
    pr(f"{'sigma':>6} | {'add_miss':>8} | {'min':>5} | {'max':>5} | {'mean':>7} | {'std':>6}")
    pr('-' * 50)
    
    current_sigma = None
    for _, row in grid.iterrows():
        if current_sigma != row['sigma']:
            if current_sigma is not None:
                pr('-' * 50)
            current_sigma = row['sigma']
        pr(f"{row['sigma']:>6.1f} | {row['additional_missing_rate']:>7.0%} | "
           f"{int(row['min']):>5} | {int(row['max']):>5} | {row['mean']:>7.1f} | {row['std']:>6.1f}")
    
    pr()
    pr('=' * 80)
    pr('KEY FINDINGS')
    pr('=' * 80)
    pr()
    
    # Find cases where max_iter was hit (100 iterations)
    did_not_converge = df[df['n_iterations'] >= 100]
    pr(f"Cases that did NOT converge (hit max_iter=100): {len(did_not_converge)}")
    if len(did_not_converge) > 0:
        pr("  These occurred at:")
        for (sigma, add_miss), count in did_not_converge.groupby(['sigma', 'additional_missing_rate']).size().items():
            total_miss = df[(df['sigma']==sigma) & (df['additional_missing_rate']==add_miss)]['total_missing_rate'].mean()
            pr(f"    sigma={sigma}, add_miss={add_miss:.0%} (total ~{total_miss:.1%}): {count} cases")
    pr()
    
    # Typical convergence
    converged = df[df['n_iterations'] < 100]
    pr(f"Typical convergence (excluding non-converged cases):")
    pr(f"  Overall: {int(converged['n_iterations'].min())} - {int(converged['n_iterations'].max())} iterations, mean={converged['n_iterations'].mean():.1f}")
    pr()
    
    # By miss rate
    pr("Typical convergence by missing rate:")
    for add_rate in sorted(df['additional_missing_rate'].unique()):
        subset = converged[converged['additional_missing_rate'] == add_rate]
        if len(subset) > 0:
            total_miss = df[df['additional_missing_rate']==add_rate]['total_missing_rate'].mean()
            pr(f"  {add_rate:4.0%} additional ({total_miss:5.1%} total): "
               f"{int(subset['n_iterations'].min()):3} - {int(subset['n_iterations'].max()):3} iterations, mean={subset['n_iterations'].mean():.1f}")

print("\nSummary saved to: results/convergence_summary.txt")
