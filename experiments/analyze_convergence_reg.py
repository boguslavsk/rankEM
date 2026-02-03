"""Simple analysis of regularized convergence study.

NOTE: Run from project root directory.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)

import pandas as pd

df = pd.read_csv('results/convergence_study_regularized.csv')

# Write to file
with open('results/convergence_reg_summary.txt', 'w', encoding='utf-8') as f:
    f.write('=' * 80 + '\n')
    f.write('EM CONVERGENCE STUDY WITH REGULARIZATION\n')
    f.write('=' * 80 + '\n\n')
    
    f.write(f'Total runs: {len(df)}\n')
    f.write(f'Lambda values tested: 0.1, 1.0, 5.0\n\n')
    
    # Overall by lambda
    f.write('OVERALL SUMMARY BY LAMBDA\n')
    f.write('-' * 60 + '\n')
    for lam in [0.1, 1.0, 5.0]:
        subset = df[df['lambda'] == lam]
        non_conv = len(subset[subset['n_iterations'] >= 100])
        conv = subset[subset['n_iterations'] < 100]
        f.write(f'  lambda={lam}:\n')
        f.write(f'    Range: {int(subset["n_iterations"].min())} - {int(subset["n_iterations"].max())}\n')
        f.write(f'    Mean: {subset["n_iterations"].mean():.1f}\n')
        f.write(f'    Non-converged (hit 100): {non_conv}\n')
        if len(conv) > 0:
            f.write(f'    Typical range (converged): {int(conv["n_iterations"].min())} - {int(conv["n_iterations"].max())}\n')
        f.write('\n')
    
    # Table by missing rate and lambda
    f.write('\nITERATIONS BY MISSING RATE AND LAMBDA\n')
    f.write('-' * 70 + '\n')
    f.write(f'{"Miss%":>6} | {"lambda=0.1":>20} | {"lambda=1.0":>15} | {"lambda=5.0":>15}\n')
    f.write('-' * 70 + '\n')
    
    for add_rate in sorted(df['additional_missing_rate'].unique()):
        total_miss = df[df['additional_missing_rate']==add_rate]['total_missing_rate'].mean()
        
        row = f'{total_miss:5.0%} |'
        for lam in [0.1, 1.0, 5.0]:
            subset = df[(df['additional_missing_rate']==add_rate) & (df['lambda']==lam)]
            min_i = int(subset['n_iterations'].min())
            max_i = int(subset['n_iterations'].max())
            mean_i = subset['n_iterations'].mean()
            non_conv = len(subset[subset['n_iterations']>=100])
            
            if non_conv > 0:
                row += f' {min_i:2}-{max_i:3} (x{non_conv}) |'
            else:
                row += f' {min_i:2}-{max_i:3}, m={mean_i:4.1f} |'
        f.write(row + '\n')
    
    # Key findings
    f.write('\n' + '=' * 80 + '\n')
    f.write('KEY FINDINGS\n')
    f.write('=' * 80 + '\n\n')
    
    f.write('1. lambda=0.1 (weak regularization):\n')
    f.write('   - Fails to converge in 1280/1600 cases (80%)\n')
    f.write('   - Only converges reliably at 33% missing (block pattern only)\n')
    f.write('   - When it converges: 8-10 iterations\n\n')
    
    f.write('2. lambda=1.0 (moderate regularization):\n')
    f.write('   - Always converges (0 failures)\n')
    f.write('   - Iteration range: 8-62\n')
    f.write('   - Lower missing -> more iterations (inverse relationship)\n\n')
    
    f.write('3. lambda=5.0 (strong regularization):\n')
    f.write('   - Always converges (0 failures)\n')
    f.write('   - Iteration range: 7-18\n')
    f.write('   - Fastest convergence, ~10-14 iterations across all conditions\n\n')
    
    f.write('COMPARISON (iterations for typical convergence):\n')
    f.write('-' * 50 + '\n')
    f.write('Miss Rate | No Reg (lam=0) | lam=1.0 | lam=5.0\n')
    f.write('-' * 50 + '\n')
    
    # Load no-reg data for comparison
    df0 = pd.read_csv('results/convergence_study.csv')
    for add_rate in sorted(df['additional_missing_rate'].unique()):
        total_miss = df[df['additional_missing_rate']==add_rate]['total_missing_rate'].mean()
        
        # No reg
        sub0 = df0[df0['additional_missing_rate']==add_rate]
        sub0_conv = sub0[sub0['n_iterations']<100]
        mean0 = sub0_conv['n_iterations'].mean() if len(sub0_conv)>0 else float('nan')
        
        # lam=1
        sub1 = df[(df['additional_missing_rate']==add_rate) & (df['lambda']==1.0)]
        mean1 = sub1['n_iterations'].mean()
        
        # lam=5
        sub5 = df[(df['additional_missing_rate']==add_rate) & (df['lambda']==5.0)]
        mean5 = sub5['n_iterations'].mean()
        
        f.write(f'  {total_miss:4.0%}   |     {mean0:5.1f}      |  {mean1:5.1f}  |  {mean5:5.1f}\n')

print('Summary written to: results/convergence_reg_summary.txt')
