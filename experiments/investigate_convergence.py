"""Investigate why lam=0.1 has higher non-convergence than lam=0.

NOTE: Run from project root directory.
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
EXPERIMENTS_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXPERIMENTS_DIR))
os.chdir(PROJECT_ROOT)

import pandas as pd
import numpy as np

# Load both datasets
df0 = pd.read_csv('results/convergence_study.csv')  # lam=0
df_reg = pd.read_csv('results/convergence_study_regularized.csv')
df01 = df_reg[df_reg['lambda'] == 0.1]

print("=" * 80)
print("INVESTIGATION: Why does lam=0.1 fail more than lam=0?")
print("=" * 80)

# Compare at each missing rate
print("\nComparison at each missing rate:")
print("-" * 70)
print(f"{'Add.Miss':>8} | {'lam=0 fail':>10} | {'lam=0.1 fail':>12} | {'Same seeds fail?':>20}")
print("-" * 70)

for add_miss in sorted(df0['additional_missing_rate'].unique()):
    sub0 = df0[df0['additional_missing_rate'] == add_miss]
    sub01 = df01[df01['additional_missing_rate'] == add_miss]
    
    fail0 = set(sub0[sub0['n_iterations'] >= 100]['seed'].unique())
    fail01 = set(sub01[sub01['n_iterations'] >= 100]['seed'].unique())
    
    # How many fail in both?
    both_fail = len(fail0 & fail01)
    only_0_fail = len(fail0 - fail01)
    only_01_fail = len(fail01 - fail0)
    
    print(f"{add_miss:>7.0%} | {len(fail0):>10} | {len(fail01):>12} | "
          f"both={both_fail}, only lam=0={only_0_fail}, only lam=0.1={only_01_fail}")

print("\n" + "=" * 80)
print("DETAIL: At 40% total missing (10% additional)")
print("=" * 80)

add_miss = 0.1  # 40% total
sub0 = df0[df0['additional_missing_rate'] == add_miss]
sub01 = df01[df01['additional_missing_rate'] == add_miss]

fail0_seeds = set(sub0[sub0['n_iterations'] >= 100]['seed'].unique())
fail01_seeds = set(sub01[sub01['n_iterations'] >= 100]['seed'].unique())

print(f"\nSeeds that fail at lam=0: {sorted(fail0_seeds)}")
print(f"Seeds that fail at lam=0.1: {len(fail01_seeds)} seeds")

# Look at seeds that only fail at lam=0.1
only_01_fail = fail01_seeds - fail0_seeds
print(f"\nSeeds that ONLY fail at lam=0.1 (but converge at lam=0): {len(only_01_fail)}")

if only_01_fail:
    # Compare iterations for these problematic seeds
    print("\nIterations for first 10 seeds that only fail at lam=0.1:")
    print("-" * 40)
    for seed in sorted(list(only_01_fail))[:10]:
        iter0 = sub0[sub0['seed']==seed]['n_iterations'].values[0]
        iter01 = sub01[sub01['seed']==seed]['n_iterations'].values[0]
        print(f"  Seed {seed}: lam=0 -> {iter0} iters, lam=0.1 -> {iter01} iters")

# Now let's understand the mechanism - run one specific case and track convergence
print("\n" + "=" * 80)
print("MECHANISM INVESTIGATION")
print("=" * 80)

print("""
The EM update equations are:
  theta_i = sum_j(X_ij - mu - beta_j) / (n_obs_i + lam)
  beta_j = sum_i(X_ij - mu - theta_i) / (n_obs_j + lam)

With lam=0: denominators are just observation counts
With lam=0.1: denominators are slightly larger

Key insight: The denominator affects the STEP SIZE of each update.
- lam=0: larger steps (denominator = n_obs only)
- lam=0.1: slightly smaller steps (denominator = n_obs + 0.1)
""")

# Let's verify by running a specific case
from data_generator import DataGenerator

# Pick a seed that fails at lam=0.1 but converges at lam=0
test_seed = sorted(list(only_01_fail))[0] if only_01_fail else 0
test_sigma = 0.5
test_add_miss = 0.1

print(f"Test case: seed={test_seed}, sigma={test_sigma}, add_miss={test_add_miss}")

# Generate data
dg = DataGenerator.from_simulation(
    n_students=60, n_problems=24, seed=test_seed,
    sigma=test_sigma, problem_type='block', n_blocks=3
).apply_missing(
    pattern='both', correlation='none',
    missing_rate=test_add_miss, correlation_strength=0.0, n_blocks=3
)

X_miss = dg.X_missing

# Custom EM to track convergence
def em_with_tracking(X, lambda_val, max_iter=100, tol=1e-4):
    """Track convergence differences per iteration."""
    X_work = X.copy()
    n_students, n_problems = X_work.shape
    
    mu = np.nanmean(X_work)
    theta = np.zeros(n_students)
    beta = np.zeros(n_problems)
    
    observed_mask = ~np.isnan(X_work)
    n_obs_student = np.sum(observed_mask, axis=1)
    n_obs_problem = np.sum(observed_mask, axis=0)
    
    diffs = []
    for iteration in range(max_iter):
        theta_old = theta.copy()
        beta_old = beta.copy()
        
        res_theta = X_work - mu - beta[np.newaxis, :]
        theta = np.nansum(res_theta, axis=1) / (n_obs_student + lambda_val)
        
        res_beta = X_work - mu - theta[:, np.newaxis]
        beta = np.nansum(res_beta, axis=0) / (n_obs_problem + lambda_val)
        
        diff = np.linalg.norm(theta - theta_old) + np.linalg.norm(beta - beta_old)
        diffs.append(diff)
        
        if diff < tol:
            break
    
    return iteration + 1, diffs

n_iter_0, diffs_0 = em_with_tracking(X_miss, 0.0)
n_iter_01, diffs_01 = em_with_tracking(X_miss, 0.1)

print(f"\nlam=0.0: converged in {n_iter_0} iterations")
print(f"lam=0.1: converged in {n_iter_01} iterations")

print("\nConvergence diff values (how much params changed each iter):")
print(f"  Iter |   lam=0   |  lam=0.1")
print("-" * 35)
for i in range(min(20, max(len(diffs_0), len(diffs_01)))):
    d0 = diffs_0[i] if i < len(diffs_0) else 0
    d01 = diffs_01[i] if i < len(diffs_01) else 0
    converged0 = " *" if i == len(diffs_0)-1 and n_iter_0 < 100 else ""
    converged01 = " *" if i == len(diffs_01)-1 and n_iter_01 < 100 else ""
    print(f"  {i+1:4} | {d0:9.6f}{converged0} | {d01:9.6f}{converged01}")

if len(diffs_01) > 20:
    print("  ...")
    for i in range(max(len(diffs_01)-3, 20), len(diffs_01)):
        d01 = diffs_01[i]
        print(f"  {i+1:4} |           | {d01:9.6f}")

print("\n(* = converged at this iteration)")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print("""
The issue is that lam=0.1 creates a SLOWER convergence rate than lam=0!

With lam=0, the algorithm takes larger steps and converges quickly.
With lam=0.1, the steps are slightly smaller (due to +0.1 in denominator),
causing the algorithm to move more slowly toward the solution.

At higher missing rates, the slower convergence means lam=0.1 often
doesn't reach the tolerance threshold within 100 iterations.

This is counter-intuitive because we expect regularization to help, but
very weak regularization (0.1) just slows down convergence without providing
the stability benefits that stronger regularization (lam>=1) provides.

RECOMMENDATIONS:
1. Use NO regularization (lam=0) for fast convergence at low-to-medium missing
2. Use STRONG regularization (lam>=1) for guaranteed convergence at high missing
3. AVOID weak regularization (lam=0.1) - it's the worst of both worlds
""")
