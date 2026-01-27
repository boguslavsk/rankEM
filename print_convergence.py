"""Quick script to print convergence study results."""
import pandas as pd

df = pd.read_csv('results/convergence_study.csv')

print('=' * 80)
print('EM CONVERGENCE RATE STUDY')
print('=' * 80)

# Summary by sigma
print('\nSummary by sigma (across all missing rates):')
print('-' * 50)
sigma_summary = df.groupby('sigma')['n_iterations'].agg(['min', 'max', 'mean']).round(2)
for sigma, row in sigma_summary.iterrows():
    print(f"sigma={sigma:.1f}: iterations {int(row['min']):3} - {int(row['max']):3}, mean={row['mean']:.1f}")

# Summary by missing rate
print('\nSummary by missing rate (across all sigmas):')
print('-' * 60)
miss_summary = df.groupby('additional_missing_rate').agg({
    'total_missing_rate': 'mean',
    'n_iterations': ['min', 'max', 'mean']
}).round(2)
miss_summary.columns = ['total_miss', 'min', 'max', 'mean']
for add_rate, row in miss_summary.iterrows():
    print(f"add_miss={add_rate:5.0%}, total={row['total_miss']:5.1%}: iterations {int(row['min']):3} - {int(row['max']):3}, mean={row['mean']:.1f}")

# Full grid
print('\nFull Grid: Iterations by (sigma, missing rate)')
print('-' * 70)
grid = df.groupby(['sigma', 'additional_missing_rate'])['n_iterations'].agg(['min', 'max', 'mean']).round(1)
grid = grid.reset_index()
print(f"{'sigma':>6} | {'add_miss':>8} | {'min':>5} | {'max':>5} | {'mean':>7}")
print('-' * 70)
for _, row in grid.iterrows():
    print(f"{row['sigma']:>6.1f} | {row['additional_missing_rate']:>7.0%} | {int(row['min']):>5} | {int(row['max']):>5} | {row['mean']:>7.1f}")
