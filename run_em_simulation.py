
import numpy as np

def regularized_em_ranking(X_in, reg_lambda=1.0, max_iter=100, tol=1e-4):
    """
    Optimized Regularized EM (ALS).
    """
    X = X_in.copy()
    n_students, n_problems = X.shape
    
    # Initialization
    mu = np.nanmean(X)
    theta = np.zeros(n_students)
    beta = np.zeros(n_problems)
    
    # Masks
    observed_mask = ~np.isnan(X)
    n_obs_student = np.sum(observed_mask, axis=1)
    n_obs_problem = np.sum(observed_mask, axis=0)
    
    for iteration in range(max_iter):
        theta_old = theta.copy()
        beta_old = beta.copy()
        
        # 1. Update Theta
        res_theta = X - mu - beta[np.newaxis, :]
        theta = np.nansum(res_theta, axis=1) / (n_obs_student + reg_lambda)
        
        # 2. Update Beta
        res_beta = X - mu - theta[:, np.newaxis]
        beta = np.nansum(res_beta, axis=0) / (n_obs_problem + reg_lambda)
        
        diff = np.linalg.norm(theta - theta_old) + np.linalg.norm(beta - beta_old)
        if diff < tol:
            break
            
    X_filled = mu + theta[:, np.newaxis] + beta[np.newaxis, :]
    X_filled = np.clip(X_filled, 0, 6)
    return theta, beta, X_filled, mu

def run_study():
    # --- 1. Simulate Data ---
    rng = np.random.default_rng(seed=42)
    Ns = 60
    Np = 24
    
    # True parameters
    a = rng.normal(loc=0, scale=2, size=Ns)       # Student Ability
    b = rng.uniform(-3, 3, size=Np)               # Problem Difficulty
    
    # Create Full Matrix X
    eps = rng.normal(loc=0, scale=1.5, size=(Ns, Np))
    X_true_underlying = 3.5 + a[:, np.newaxis] + b[np.newaxis, :] + eps
    X_true = np.clip(np.round(X_true_underlying), 0, 6) # Discrete Truth
    
    # Create Missing Data
    X_missing = X_true.astype(float)
    student_indices = rng.permutation(Ns)
    groups = np.split(student_indices, 3)
    
    for i in range(3):
        start_col = 8 * i
        end_col = 8 * (i + 1)
        X_missing[groups[i], start_col:end_col] = np.nan
        
    print(f"Values missing: {np.isnan(X_missing).sum()} / {X_true.size}")

    # --- 2. Compare Lambdas ---
    lambdas = [0.0, 0.0001, 0.001, 0.01, 0.1, 0.5, 1.0]
    
    print("\n" + "="*80)
    print(f"{'Lambda':<8} | {'RMSE(Theta)*':<12} | {'RMSE(Beta)*':<12} | {'RMSE(Matrix)':<12}")
    print("-" * 80)
    
    results = []
    
    # Center true parameters for fair comparison (remove intercept shift)
    a_centered = a - np.mean(a)
    b_centered = b - np.mean(b)
    
    for lam in lambdas:
        theta_est, beta_est, X_fill, mu_est = regularized_em_ranking(X_missing, reg_lambda=lam, max_iter=500)
        
        # Center estimates
        th_centered = theta_est - np.mean(theta_est)
        be_centered = beta_est - np.mean(beta_est)
        
        # Calculate RMSEs
        rmse_theta = np.sqrt(np.mean((th_centered - a_centered)**2))
        rmse_beta  = np.sqrt(np.mean((be_centered - b_centered)**2))
        rmse_matrix = np.sqrt(np.mean((X_fill - X_true)**2))
        
        print(f"{lam:<8.1f} | {rmse_theta:<12.4f} | {rmse_beta:<12.4f} | {rmse_matrix:<12.4f}")
        
        results.append({
            'lambda': lam,
            'rmse_theta': rmse_theta,
            'rmse_beta': rmse_beta,
            'rmse_matrix': rmse_matrix
        })
        
    # --- 3. Summary ---
    best_theta = min(results, key=lambda x: x['rmse_theta'])
    best_beta = min(results, key=lambda x: x['rmse_beta'])
    best_matrix = min(results, key=lambda x: x['rmse_matrix'])
    
    print("="*80)
    print(f"Optimal Lambda for Student Ability (Theta):   {best_theta['lambda']} (RMSE {best_theta['rmse_theta']:.4f})")
    print(f"Optimal Lambda for Problem Difficulty (Beta): {best_beta['lambda']} (RMSE {best_beta['rmse_beta']:.4f})")
    print(f"Optimal Lambda for Missing Data (Matrix X):   {best_matrix['lambda']} (RMSE {best_matrix['rmse_matrix']:.4f})")
    print("="*80)
    
    if best_theta['lambda'] == best_beta['lambda'] == best_matrix['lambda']:
        print(">> CONCLUSION: All metrics agree on the same optimal lambda.")
    else:
        print(">> CONCLUSION: Optimal estimates diverge!")

if __name__ == "__main__":
    run_study()
