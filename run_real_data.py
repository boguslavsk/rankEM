"""
Run the estimator and heuristics on a real dataset and produce a full report.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from estimator import Estimator


def load_csv_data(filepath: str) -> tuple[list[str], np.ndarray]:
    """
    Load score matrix from CSV file.
    
    The first column contains row labels (ignored in processing but used for output).
    Empty cells are treated as missing data (NaN).
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        Tuple of (row_labels, score_matrix) where:
        - row_labels: List of strings from the first column
        - score_matrix: numpy array with NaN for missing values
    """
    # Read CSV, treating empty strings as NaN
    df = pd.read_csv(filepath, header=None, na_values=['', ' '])
    
    # Extract first column as row labels
    row_labels = df.iloc[:, 0].astype(str).tolist()
    
    # Extract remaining columns as data matrix
    X = df.iloc[:, 1:].values.astype(float)
    
    return row_labels, X


def compute_missing_stats(X: np.ndarray) -> dict:
    """Compute statistics about missing data patterns."""
    n_students, n_problems = X.shape
    total_cells = n_students * n_problems
    missing_mask = np.isnan(X)
    n_missing = np.sum(missing_mask)
    
    # Missing by row (student)
    missing_per_student = np.sum(missing_mask, axis=1)
    
    # Missing by column (problem)
    missing_per_problem = np.sum(missing_mask, axis=0)
    
    return {
        'n_students': n_students,
        'n_problems': n_problems,
        'total_cells': total_cells,
        'n_missing': n_missing,
        'missing_rate': n_missing / total_cells,
        'missing_per_student_mean': np.mean(missing_per_student),
        'missing_per_student_std': np.std(missing_per_student),
        'missing_per_problem_mean': np.mean(missing_per_problem),
        'missing_per_problem_std': np.std(missing_per_problem),
        'students_with_complete_data': np.sum(missing_per_student == 0),
        'problems_with_complete_data': np.sum(missing_per_problem == 0),
    }


def generate_report(X: np.ndarray, 
                   results: dict, 
                   missing_stats: dict,
                   output_file: str):
    """Generate a comprehensive markdown report."""
    
    lines = []
    
    # Header
    lines.append("# Real Data Analysis Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Data Overview
    lines.append("## Data Overview")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|:-------|------:|")
    lines.append(f"| Number of Students | {missing_stats['n_students']} |")
    lines.append(f"| Number of Problems | {missing_stats['n_problems']} |")
    lines.append(f"| Total Cells | {missing_stats['total_cells']} |")
    lines.append(f"| Missing Cells | {missing_stats['n_missing']} |")
    lines.append(f"| **Missing Rate** | **{missing_stats['missing_rate']:.2%}** |")
    lines.append("")
    
    # Missing Data Pattern
    lines.append("### Missing Data Distribution")
    lines.append("")
    lines.append("| Dimension | Mean Missing | Std Missing | Fully Observed |")
    lines.append("|:----------|-------------:|------------:|---------------:|")
    lines.append(f"| Per Student | {missing_stats['missing_per_student_mean']:.2f} | "
                f"{missing_stats['missing_per_student_std']:.2f} | "
                f"{missing_stats['students_with_complete_data']} |")
    lines.append(f"| Per Problem | {missing_stats['missing_per_problem_mean']:.2f} | "
                f"{missing_stats['missing_per_problem_std']:.2f} | "
                f"{missing_stats['problems_with_complete_data']} |")
    lines.append("")
    
    # Score Distribution (observed only)
    observed_scores = X[~np.isnan(X)]
    lines.append("### Score Distribution (Observed Values)")
    lines.append("")
    lines.append("| Statistic | Value |")
    lines.append("|:----------|------:|")
    lines.append(f"| Min | {np.min(observed_scores):.1f} |")
    lines.append(f"| Max | {np.max(observed_scores):.1f} |")
    lines.append(f"| Mean | {np.mean(observed_scores):.3f} |")
    lines.append(f"| Median | {np.median(observed_scores):.1f} |")
    lines.append(f"| Std Dev | {np.std(observed_scores):.3f} |")
    lines.append("")
    
    lines.append("---")
    lines.append("")
    
    # Estimation Results
    lines.append("## Estimation Results")
    lines.append("")
    
    for method_name, est in results.items():
        lines.append(f"### {method_name}")
        lines.append("")
        
        # Convergence info (for EM only)
        if est.n_iterations is not None:
            lines.append(f"**Convergence:** {est.n_iterations} iterations")
            lines.append("")
        
        # Parameter estimates summary
        lines.append("#### Parameter Estimates Summary")
        lines.append("")
        lines.append("| Parameter | Mean | Std Dev | Min | Max |")
        lines.append("|:----------|-----:|--------:|----:|----:|")
        
        # Theta (student abilities)
        lines.append(f"| θ (Student Ability) | {np.mean(est.theta):.3f} | "
                    f"{np.std(est.theta):.3f} | {np.min(est.theta):.3f} | {np.max(est.theta):.3f} |")
        
        # Beta (problem difficulties)
        lines.append(f"| β (Problem Difficulty) | {np.mean(est.beta):.3f} | "
                    f"{np.std(est.beta):.3f} | {np.min(est.beta):.3f} | {np.max(est.beta):.3f} |")
        lines.append("")
        
        # Variability estimates
        lines.append("#### Variability Estimates")
        lines.append("")
        lines.append("| Measure | Estimate |")
        lines.append("|:--------|--------:|")
        if est.std_theta is not None:
            lines.append(f"| Std(θ) - Student Ability Spread | {est.std_theta:.4f} |")
        if est.std_beta is not None:
            lines.append(f"| Std(β) - Problem Difficulty Spread | {est.std_beta:.4f} |")
        if est.sigma_epsilon is not None:
            lines.append(f"| **σ(ε) - Residual Std Dev** | **{est.sigma_epsilon:.4f}** |")
        if est.mu is not None:
            lines.append(f"| μ (Global Mean) | {est.mu:.4f} |")
        lines.append("")
        
    lines.append("---")
    lines.append("")
    
    # Comparison Table
    lines.append("## Method Comparison Summary")
    lines.append("")
    lines.append("| Method | σ(ε) | Std(θ) | Std(β) | Iterations |")
    lines.append("|:-------|-----:|-------:|-------:|-----------:|")
    
    for method_name, est in results.items():
        n_iter = est.n_iterations if est.n_iterations is not None else "-"
        lines.append(f"| {method_name} | {est.sigma_epsilon:.4f} | "
                    f"{est.std_theta:.4f} | {est.std_beta:.4f} | {n_iter} |")
    lines.append("")
    
    lines.append("---")
    lines.append("")
    
    # Detailed Parameter Values
    lines.append("## Detailed Parameter Estimates")
    lines.append("")
    
    # Use the EM estimator for detailed output (as the primary method)
    em_est = results.get('EM Estimator', list(results.values())[0])
    
    lines.append("### Student Abilities (θ) - EM Estimator")
    lines.append("")
    lines.append("Students sorted by estimated ability (highest first):")
    lines.append("")
    
    # Sort students by ability
    sorted_students = np.argsort(em_est.theta)[::-1]
    lines.append("| Rank | Student ID | θ Estimate |")
    lines.append("|-----:|-----------:|-----------:|")
    for rank, student_id in enumerate(sorted_students[:15], 1):
        lines.append(f"| {rank} | {student_id + 1} | {em_est.theta[student_id]:+.3f} |")
    lines.append("| ... | ... | ... |")
    for rank, student_id in enumerate(sorted_students[-5:], len(sorted_students) - 4):
        lines.append(f"| {rank} | {student_id + 1} | {em_est.theta[student_id]:+.3f} |")
    lines.append("")
    
    lines.append("### Problem Difficulties (β) - EM Estimator")
    lines.append("")
    lines.append("| Problem ID | β Estimate |")
    lines.append("|-----------:|-----------:|")
    for prob_id, beta_val in enumerate(em_est.beta):
        lines.append(f"| {prob_id + 1} | {beta_val:+.3f} |")
    lines.append("")
    
    lines.append("---")
    lines.append("")
    
    # Model Interpretation
    lines.append("## Model Interpretation")
    lines.append("")
    lines.append("The additive model assumes: **X_ij = μ + θ_i + β_j + ε_ij**")
    lines.append("")
    lines.append("Where:")
    lines.append("- **X_ij**: Score for student i on problem j")
    lines.append("- **μ**: Global mean score")
    lines.append("- **θ_i**: Student i's ability (deviation from mean)")
    lines.append("- **β_j**: Problem j's effect (deviation from mean, positive = easier)")
    lines.append("- **ε_ij**: Random error term")
    lines.append("")
    
    em_est = results.get('EM Estimator', list(results.values())[0])
    if em_est.mu is not None:
        lines.append(f"With μ = {em_est.mu:.2f}, an average student on an average problem is expected to score ~{em_est.mu:.1f}.")
        lines.append("")
        
        # Example interpretations
        best_student = np.argmax(em_est.theta)
        worst_student = np.argmin(em_est.theta)
        easiest_prob = np.argmax(em_est.beta)
        hardest_prob = np.argmin(em_est.beta)
        
        lines.append("### Key Findings")
        lines.append("")
        lines.append(f"- **Highest ability student**: Student {best_student + 1} "
                    f"(θ = {em_est.theta[best_student]:+.2f})")
        lines.append(f"- **Lowest ability student**: Student {worst_student + 1} "
                    f"(θ = {em_est.theta[worst_student]:+.2f})")
        lines.append(f"- **Easiest problem**: Problem {easiest_prob + 1} "
                    f"(β = {em_est.beta[easiest_prob]:+.2f})")
        lines.append(f"- **Hardest problem**: Problem {hardest_prob + 1} "
                    f"(β = {em_est.beta[hardest_prob]:+.2f})")
        lines.append("")
        
        ability_range = em_est.theta.max() - em_est.theta.min()
        difficulty_range = em_est.beta.max() - em_est.beta.min()
        lines.append(f"- **Ability spread**: {ability_range:.2f} points")
        lines.append(f"- **Difficulty spread**: {difficulty_range:.2f} points")
        lines.append(f"- **Noise level (σε)**: {em_est.sigma_epsilon:.2f} points")
        lines.append("")
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"Report generated: {output_file}")
    return output_file



def save_all_estimates_to_csv(results: dict, data_dir: Path, row_labels: list[str] = None):
    """
    Save all theta and beta estimates from all methods to consolidated CSV files.
    
    Args:
        results: Dictionary of method_name -> Estimator
        data_dir: Directory to save files
        row_labels: Optional list of row labels for theta estimates
    """
    # Build theta DataFrame with all methods as columns
    if row_labels is not None:
        theta_data = {'row_label': row_labels}
    else:
        theta_data = {'student_id': range(1, len(list(results.values())[0].theta) + 1)}
    for method_name, est in results.items():
        # Create column name from method name
        col_name = method_name.lower().replace(' ', '_')
        theta_data[col_name] = est.theta
    
    theta_df = pd.DataFrame(theta_data)
    theta_file = data_dir / 'theta_all_methods.csv'
    theta_df.to_csv(theta_file, index=False)
    print(f"Saved all theta estimates to: {theta_file}")
    
    # Build beta DataFrame with all methods as columns
    beta_data = {'problem_id': range(1, len(list(results.values())[0].beta) + 1)}
    for method_name, est in results.items():
        col_name = method_name.lower().replace(' ', '_')
        beta_data[col_name] = est.beta
    
    beta_df = pd.DataFrame(beta_data)
    beta_file = data_dir / 'beta_all_methods.csv'
    beta_df.to_csv(beta_file, index=False)
    print(f"Saved all beta estimates to: {beta_file}")


def main():
    """Main entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run estimators on real data batch')
    parser.add_argument('batch', type=str, help='Batch name (subfolder under data/)')
    args = parser.parse_args()
    
    batch_name = args.batch
    
    # Setup paths
    data_dir = Path('data') / batch_name
    if not data_dir.exists():
        print(f"Error: Batch directory not found: {data_dir}")
        return
    
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    # Find CSV file (exclude estimate files we generate)
    csv_files = [f for f in data_dir.glob('*.csv') 
                 if not any(x in f.name for x in ['estimates', 'theta_', 'beta_', 'parameter_', 'all_methods'])]
    if not csv_files:
        print(f"No CSV files found in {data_dir}/ directory")
        return
    
    csv_file = csv_files[0]
    print(f"Processing batch: {batch_name}")
    print(f"Loading data from: {csv_file}")
    
    # Load data
    row_labels, X = load_csv_data(str(csv_file))
    print(f"Data shape: {X.shape} (students × problems)")
    
    # Compute missing data statistics
    missing_stats = compute_missing_stats(X)
    print(f"Missing rate: {missing_stats['missing_rate']:.2%}")
    print()
    
    # Run estimators
    results = {}
    
    # 1. EM Estimator
    print("Running EM Estimator...")
    em_est = Estimator.em(X, lambda_theta=1.0, lambda_beta=1.0)
    results['EM Estimator'] = em_est
    print(f"  Converged in {em_est.n_iterations} iterations")
    print(f"  σ(ε) = {em_est.sigma_epsilon:.4f}")
    print()
    
    # 2. Mean Imputation Heuristic
    print("Running Mean Imputation...")
    imp_est = Estimator.mean_imputation(X)
    results['Mean Imputation'] = imp_est
    print(f"  σ(ε) = {imp_est.sigma_epsilon:.4f}")
    print()
    
    # 3. Day Average Heuristic
    print("Running Day Average...")
    day_est = Estimator.day_average(X, n_blocks=3)
    results['Day Average'] = day_est
    print(f"  σ(ε) = {day_est.sigma_epsilon:.4f}")
    print()
    
    # Generate report (batch-specific filename)
    report_file = results_dir / f'{batch_name}_analysis.md'
    generate_report(X, results, missing_stats, str(report_file))
    
    # Save consolidated files with all methods as columns
    print()
    save_all_estimates_to_csv(results, data_dir, row_labels)
    
    # Print summary to console
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Data: {missing_stats['n_students']} students × {missing_stats['n_problems']} problems")
    print(f"Missing rate: {missing_stats['missing_rate']:.2%}")
    print()
    print("Estimates:")
    print("-" * 50)
    print(f"{'Method':<20} {'σ(ε)':<10} {'Std(θ)':<10} {'Std(β)':<10} {'Iters':<8}")
    print("-" * 50)
    for name, est in results.items():
        n_iter = str(est.n_iterations) if est.n_iterations is not None else "-"
        print(f"{name:<20} {est.sigma_epsilon:<10.4f} {est.std_theta:<10.4f} {est.std_beta:<10.4f} {n_iter:<8}")
    print("-" * 50)
    print()
    print(f"Full report saved to: {report_file}")


if __name__ == '__main__':
    main()
