
import numpy as np
from scipy.stats import spearmanr

# --- Data Generation ---
def generate_data(seed, sigma, problem_type='block'):
    rng = np.random.default_rng(seed=seed)
    Ns = 60
    Np = 24
    
    a = rng.normal(loc=0, scale=2, size=Ns)
    
    if problem_type == 'uniform':
        b = rng.uniform(-3, 3, size=Np)
    elif problem_type == 'block':
        # Block-varying problem difficulty
        b = np.concatenate([
            rng.uniform(-2, 4, size=8),  # Day 0: Mean +1 (Easy)
            rng.uniform(-3, 3, size=8),  # Day 1: Mean 0 (Neutral)
            rng.uniform(-4, 2, size=8)   # Day 2: Mean -1 (Hard)
        ])
    else:
        raise ValueError("problem_type must be 'uniform' or 'block'")
    
    eps = rng.normal(loc=0, scale=sigma, size=(Ns, Np))
    X_true_underlying = 3.5 + a[:, np.newaxis] + b[np.newaxis, :] + eps
    X_true = np.clip(np.round(X_true_underlying), 0, 6)
    
    X_missing = X_true.astype(float)
    student_indices = rng.permutation(Ns)
    groups = np.split(student_indices, 3)
    
    for i in range(3):
        start_col = 8 * i
        end_col = 8 * (i + 1)
        X_missing[groups[i], start_col:end_col] = np.nan
        
    return X_missing, X_true, a, b, groups

# --- Methods (Same as before) ---
def method_em(X_in):
    X = X_in.copy()
    n_s, n_p = X.shape
    mu = np.nanmean(X)
    theta = np.zeros(n_s)
    beta = np.zeros(n_p)
    obs_mask = ~np.isnan(X)
    n_s_counts = np.sum(obs_mask, axis=1)
    n_p_counts = np.sum(obs_mask, axis=0)
    for _ in range(100):
        theta_old = theta.copy()
        res_theta = X - mu - beta[np.newaxis, :]
        theta = np.nansum(res_theta, axis=1) / n_s_counts
        res_beta = X - mu - theta[:, np.newaxis]
        beta = np.nansum(res_beta, axis=0) / n_p_counts
        if np.linalg.norm(theta - theta_old) < 1e-4: break
    X_fill = mu + theta[:, np.newaxis] + beta[np.newaxis, :]
    X_fill = np.clip(X_fill, 0, 6)
    return theta, beta, X_fill

def method_day_linking(X, groups):
    group_day_means = np.zeros((3, 3))
    for g in range(3):
        students = groups[g]
        for d in range(3):
            cols = slice(8*d, 8*(d+1))
            block = X[students, cols]
            if not np.isnan(block).all():
                group_day_means[g, d] = np.nanmean(block)
            else:
                group_day_means[g, d] = np.nan
    equations = []
    targets = []
    for g in range(3):
        for d in range(3):
            if not np.isnan(group_day_means[g, d]):
                row = np.zeros(6) 
                row[g] = 1; row[3+d] = 1
                equations.append(row); targets.append(group_day_means[g, d])
    equations.append([1,1,1, 0,0,0]); targets.append(0)
    equations.append([0,0,0, 1,1,1]); targets.append(0)
    sol, _, _, _ = np.linalg.lstsq(equations, targets, rcond=None)
    day_effects_3 = sol[3:] 
    beta_est = np.repeat(day_effects_3, 8) 
    theta_est = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        valid_vals = []
        for d in range(3):
            cols = slice(8*d, 8*(d+1))
            block = X[i, cols]
            if not np.isnan(block).all():
                valid_vals.extend(block[~np.isnan(block)] - day_effects_3[d])
        theta_est[i] = np.mean(valid_vals)
    X_fill = theta_est[:, np.newaxis] + beta_est[np.newaxis, :]
    return theta_est, beta_est, X_fill

def method_mean_imputation(X):
    # Additive Model Heuristic: X_ij ~ RowMean_i + ColMean_j - GlobalMean
    row_means = np.nanmean(X, axis=1)
    col_means = np.nanmean(X, axis=0)
    mu = np.nanmean(X)
    
    # We essentially solve for parameters directly
    theta_est = row_means - mu
    beta_est = col_means - mu
    
    # Imputation
    # Broadcasting: (N) + (M) - Scalar -> (N, M)
    X_fill_proxy = row_means[:, np.newaxis] + col_means[np.newaxis, :] - mu
    X_fill = np.clip(X_fill_proxy, 0, 6)
    
    return theta_est, beta_est, X_fill

# --- Comparison Runner ---
def run_study(problem_type='block'):
    sigmas = [0.5, 1.5, 3.0, 5.0]
    seeds = 30
    
    methods = [('EM', method_em), ('Day', method_day_linking), ('Imp', method_mean_imputation)]
    
    print(f"=====================================================================================")
    print(f"Simulation Mode: {problem_type.upper()}")
    print(f"=====================================================================================")
    
    for sigma in sigmas:
        print(f"\n>>> Running Sigma = {sigma} (30 seeds) ...")
        
        metrics = {name: {'corr_t': [], 'spear_t': [], 'corr_b': [], 'spear_b': [], 'corr_x': []} for name, _ in methods}
        
        for s in range(seeds):
            X_miss, X_true, a_true, b_true, groups = generate_data(s, sigma, problem_type=problem_type)
            mask_miss = np.isnan(X_miss)
            
            for name, func in methods:
                if name == 'Day': th, be, X_f = func(X_miss, groups)
                else:             th, be, X_f = func(X_miss)
                
                metrics[name]['corr_t'].append(np.corrcoef(th, a_true)[0,1])
                metrics[name]['spear_t'].append(spearmanr(th, a_true).correlation)
                metrics[name]['corr_b'].append(np.corrcoef(be, b_true)[0,1])
                # X Miss
                metrics[name]['corr_x'].append(np.corrcoef(X_f[mask_miss], X_true[mask_miss])[0,1])

        print(f"{'Method':<8} | {'Theta Corr':<10} {'Theta Spr':<10} | {'Beta Corr':<10} | {'X Miss Corr':<10}")
        print("-" * 65)
        for name, _ in methods:
            m = metrics[name]
            print(f"{name:<8} | "
                  f"{np.mean(m['corr_t']):.4f}     {np.mean(m['spear_t']):.4f}     | "
                  f"{np.mean(m['corr_b']):.4f}     | "
                  f"{np.mean(m['corr_x']):.4f}")

if __name__ == "__main__":
    run_study(problem_type='block')
