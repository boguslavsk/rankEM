
import numpy as np
from scipy.stats import spearmanr
from data_generator import DataGenerator
from estimator import Estimator

# --- Comparison Runner ---
def run_study(problem_type='block'):
    sigmas = [0.5, 1.5, 3.0, 5.0]
    seeds = 30
    
    methods = ['EM', 'Day', 'Imp']
    
    print(f"=====================================================================================")
    print(f"Simulation Mode: {problem_type.upper()}")
    print(f"=====================================================================================")
    
    for sigma in sigmas:
        print(f"\n>>> Running Sigma = {sigma} (30 seeds) ...")
        
        metrics = {name: {'corr_t': [], 'spear_t': [], 'corr_b': [], 'spear_b': [], 'corr_x': []} for name in methods}
        
        for s in range(seeds):
            # Use DataGenerator class
            dg = DataGenerator.from_simulation(
                n_students=60,
                n_problems=24,
                seed=s,
                sigma=sigma,
                problem_type=problem_type
            ).apply_missing(pattern='block')
            
            X_miss = dg.X_missing
            X_true = dg.X_complete
            a_true = dg.theta_true
            b_true = dg.beta_true
            groups = dg.groups
            mask_miss = np.isnan(X_miss)
            
            # Run each estimator
            for name in methods:
                if name == 'EM':
                    est = Estimator.em(X_miss, reg_lambda=0.0)
                elif name == 'Day':
                    est = Estimator.day_linking(X_miss, groups)
                else:  # 'Imp'
                    est = Estimator.mean_imputation(X_miss)
                
                th, be, X_f = est.theta, est.beta, est.X_imputed
                
                metrics[name]['corr_t'].append(np.corrcoef(th, a_true)[0,1])
                metrics[name]['spear_t'].append(spearmanr(th, a_true).correlation)
                metrics[name]['corr_b'].append(np.corrcoef(be, b_true)[0,1])
                # X Miss
                metrics[name]['corr_x'].append(np.corrcoef(X_f[mask_miss], X_true[mask_miss])[0,1])

        print(f"{'Method':<8} | {'Theta Corr':<10} {'Theta Spr':<10} | {'Beta Corr':<10} | {'X Miss Corr':<10}")
        print("-" * 65)
        for name in methods:
            m = metrics[name]
            print(f"{name:<8} | "
                  f"{np.mean(m['corr_t']):.4f}     {np.mean(m['spear_t']):.4f}     | "
                  f"{np.mean(m['corr_b']):.4f}     | "
                  f"{np.mean(m['corr_x']):.4f}")

if __name__ == "__main__":
    run_study(problem_type='block')
