# rankEM: Unified Student Ranking via Regularized Expectation-Maximization

## 1. Introduction and Motivation

This repo is designed to solve a very simple problem: ranking students based on their performance on a set of problems, where not every student has a chance to attempt every problem.

In educational assessment, we frequently encounter "sparse" datasets where not every student attempts every problem. For example, this repo was motivated by the practical problem of selecting top performers out of 120 students who attended three days of exams with 8 problems given a day. Each student was expected to attend on two days of their choice out of three, but some were able to attend only one day for logistical reasons. That means that at least 33% of the possible problem scores were missing. For each problem student had a chance to attempt, they were scored on a scale from 0 to 6. The scores could be also binarised.

The primary challenge is that a simple average of observed scores is often an unfair metric for ranking. It implicitly assumes that all problems are equally difficult and that missing data occurs randomly. This approach fails in scenarios where students self-select problems or follow different tracks. For example, a student who scores 4.5 on exclusively "hard" problems should likely be ranked higher than a student who scores 5.0 on exclusively "easy" problems. In our case, we could not assume neither that different days will have problems of the same average difficulty nor that the student choice of the day to attend would be uncorrelated with student ability.

Our goal is to implement a unified ranking system that simultaneously estimates **Student Ability** ($\theta$) and **Problem Difficulty** ($\beta$), decoupling these latent variables to produce a fair comparison.

The algorithm implemented here also estimates the full student/problem interaction matrix and works for any patterns of missing data, including non-block missing data patterns and dependencies between missing patterns and other parameters.

## 2. The Mathematical Model

We model the score $X_{ij}$ of Student $i$ on Problem $j$ using a simple linear additive model. This is conceptually similar to a Two-Way ANOVA without interaction terms or a simplified Rasch model:

$$
X_{ij} = \mu + \theta_i + \beta_j + \epsilon_{ij}
$$

Where:
* $\mu$: The global average score across all observations.
* $\theta_i$: **Student Ability** (deviation from the global mean). Positive values indicate higher-than-average performance.
* $\beta_j$: **Problem "Easiness"** (deviation from the global mean). Positive values indicate an easier problem; negative values indicate a harder one.
* $\epsilon_{ij}$: A stochastic error term representing noise (luck, careless mistakes, etc.), assumed to follow a Gaussian distribution $\epsilon_{ij} \sim \mathcal{N}(0, \sigma^2)$. We also assume that $\epsilon_{ij}$ are independent from each other, from $\theta$, $\beta$, and from the missing data distribution.

## 3. Heuristic Approaches

Before employing complex iterative algorithms, it is useful to consider simpler heuristics and understand their limitations.

### Method A: Simple/adjusted Row Averages
The most common approach is to rank students by the mean of their observed scores.
* **Pros:** Simple to calculate and explain.
* **Cons:** Highly susceptible to bias. It treats a score of 6 on a trivial problem as identical to a score of 6 on the hardest problem.

An obvious improvement to this metric can be obtained by renormalising scores available for each problem or group of problems delivered simultaneously. For example, individual problem scores can be weighted inversely to the total number of points awarded to all students for that problem. When the observations are missing in day blocks, block-wise normalisation can be used.

This approach can provide unbiased student ability estimates when missing data patterns are uncorrelated with problem difficulties and student abilities. However, it will break down in the presense of such correlations. For example, if strong students cluster together to attend on the same day, they will drive up the average score achieved on that day and the renormalisation will drag down the results of other students attending on the same day. 


### Method B: The ANOVA Heuristic (Additive Adjustment)
For a missing observation $X_{ij}$, natural estimators are row average $\text{RowMean}_i$, column average $\text{ColMean}_j$. We can combine them together. Given our model, a good way to combine them is to add them together and then remove the global mean that was double counted:

$$
\hat{X}_{ij} = \text{RowMean}_i + \text{ColMean}_j - \text{GlobalMean}
$$

* **Pros:** This method captures the additive nature of the model and restores the variance lost by simpler averaging methods. It is computationally instant.
* **Cons:** This method fails under **Biased Missingness**.
    * If a problem was part of a set that was not attempted by the best students, the heuristic will estimate the problem to be harder than it is. This will in turn lower the estimated ability of students who did attempt the problem, even if they performed well on it.
    * Conversely, if a student attended days with hard problems, this heuristic will underestimate their hypothetical performance on other problems.

### Method C: The Chain-Linking Heuristic
This method explicitly models the group structure of the data. It assumes students are divided into groups (cohorts) and problems are checked in blocks (days). 
* It calculates the mean score for each group on each day block.
* It sets up a linear system where `Mean_gd = GroupEffect_g + DayEffect_d`.
* Solving this system (with constraints) provides estimates for group abilities and day difficulties.
* **Pros:** Can disentangle student ability from problem difficulty better than simple averages when the block structure is known. It actually works on par with the EM algorithm and gives the same results for our particular case when missing patterns are simple and uniform across students.
* **Cons:** Requires explicit knowledge of the group/block structure. May fail if there's no overlap between groups and days (disconnected graph). Will become very noisy for small groups and complex missing data patterns.

## 4. The Proposed Solution: Regularized EM Algorithm

To solve these issues with the heuristics, we employ an **Expectation-Maximization (EM)** algorithm. This method handles the circular dependency: to know a student's true ability, we must know the difficulty of the problems they skipped; to know a problem's difficulty, we must know the ability of the students who solved it.

### The Algorithm Steps

1.  **Initialization:**
    Fill all missing entries in the matrix with the global mean $\mu$.

2.  **M-Step (Maximization/Estimation):**
    Calculate the parameters $\theta_i$ and $\beta_j$ that minimize the prediction error on the *observed* data. We apply **Regularization** (Ridge/L2 penalty) to handle the error term $\epsilon$ and prevent overfitting on students with few data points.

$$
    \theta_i = \frac{\sum_{j \in \text{Observed}} (X_{ij} - \mu - \beta_j)}{N_i + \lambda_\theta}
$$

Where $N_i$ is the count of problems student $i$ solved, and $\lambda_\theta$ is the regularization term for student ability.

3.  **M-Step (continued):**
    Similarly, update the problem difficulty parameters $\beta_j$:

    $$
        \beta_j = \frac{\sum_{i \in \text{Observed}} (X_{ij} - \mu - \theta_i)}{M_j + \lambda_\beta}
    $$

    Where $M_j$ is the count of students who attempted problem $j$, and $\lambda_\beta$ is the regularization term for problem difficulty.

4.  **E-Step (Expectation/Imputation):**
    Update the values for the missing entries based on the newly estimated parameters:

$$
    X_{ij}^{\text{missing}} = \mu + \theta_i + \beta_j
$$

5.  **Convergence:**
    Repeat steps 2 and 3 until the parameters stabilize (change $< 10^{-4}$) or the maximum number of iterations is reached .

### Why EM Outperforms Heuristics
The EM algorithm utilizes the entire network of connections in the data. Even if Student A and Student B never solved the same problem, they are linked through Student C, who solved problems common to both. This allows information to "flow" through the matrix, correcting the rankings even in the "Bias Trap" scenario where heuristics fail.

## 5. Implementation Strategy

* **Data Structure:** A sparse matrix of size $120 \times 24$.
* **Hyperparameters:**
    * **Regularization ($\lambda_\theta, \lambda_\beta$):** For moderate share of missing data (up to 50%) and realistic signal-to-noise ratio in problem resutls, the algorithm converges fine without regularisation. For higher share of missing data, we recommend increasing the regularization parameters.



## 7. Using on Real Data

### Usage

Run the estimator on your data using the command-line interface:

```bash
python run_real_data.py <batch_name>
```

Where `<batch_name>` is the name of a subfolder under the `data/` directory containing your CSV file.

### Data Format

The input CSV file should have:
- **No header row** — data starts on line 1
- **First column**: Row labels (e.g., student IDs) — used for labeling outputs but not in calculations
- **Remaining columns**: Numeric scores (one column per problem)
- **Empty cells**: Treated as missing data (NaN)

Example:
```csv
Alice,5,3,
Bob,,4,6
Charlie,4,,5
```

### Output Files

The script generates several output files:

| File | Description |
|------|-------------|
| `results/<batch>_analysis.md` | Comprehensive markdown report with all results |
| `data/<batch>/theta_all_methods.csv` | Student ability estimates from all methods |
| `data/<batch>/beta_all_methods.csv` | Problem difficulty estimates from all methods |

## 8. Convergence Study Results

We conducted extensive simulations to characterize the EM algorithm's convergence behavior under various conditions. The key findings are summarized below.

### Study Parameters

- **Matrix size**: 120 students × 24 problems (baseline 33% block-missing)
- **Additional missing rates tested**: 0%, 10%, 20%, 30%, 40%, 50%, 60%, 70%
- **Noise levels (σ of ε)**: 0.5, 1.5, 3.5, 6.0
- **Regularization values (λ)**: 0, 0.1, 1.0, 5.0
- **Runs per condition**: 50 random seeds

### Key Findings

| λ | Convergence | Typical Iterations | Notes |
|---|-------------|-------------------|-------|
| **0** (unbiased) | 98.25% success | 8–17 | Fails only at ≥73% missing; **produces unbiased estimates** |
| **0.1** | 20% success | 8–10 (when converged) | Fails at >33% missing |
| **1.0** | 100% success | 8–62 | Reliable across all conditions |
| **5.0** | 100% success | 7–18 | Fastest, strongest shrinkage, biased estimates |

#### Effect of Missing Data Rate

| Total Missing | λ = 0 (unbiased) | λ = 1.0 | λ = 5.0 |
|---------------|------------------|---------|---------|
| 33% | 9.5 | 9.0 | 7.8 |
| 40% | 9.7 | 44.9 | 13.8 |
| 47% | 9.9 | 43.6 | 13.1 |
| 54% | 10.2 | 41.1 | 12.8 |
| 60% | 10.6 | 39.7 | 12.1 |
| 67% | 11.3 | 37.1 | 11.3 |
| 73% | 12.2* | 32.6 | 10.3 |
| 80% | 14.4* | 26.4 | 8.9 |

*Some non-convergence cases at these levels (28 out of 1600 runs total)

### Recommendations

1. **Use λ = 0 (unregularized) when possible** — produces unbiased estimates and converges reliably up to ~67% missing data
2. **Use λ ≥ 1.0** when missing rate exceeds 70% or guaranteed convergence is required
3. **Noise level (σ) has minimal impact** on convergence — the algorithm is robust across σ = 0.5 to 6.0
4. **Missing data up to 67%** is handled well even without regularization
5. For very sparse data (>70% missing), **λ = 1.0** offers a good balance between bias and convergence

## 9. References

* **Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977).** *Maximum likelihood from incomplete data via the EM algorithm.* (The foundational paper).
* **Koren, Y., Bell, R., & Volinsky, C. (2009).** *Matrix Factorization Techniques for Recommender Systems.* (Describes similar techniques used for filling sparse matrices in the Netflix Prize).
* **Rasch, G. (1960).** *Probabilistic Models for Some Intelligence and Attainment Tests.* (The theoretical basis for additive models in testing).
* **Pelánek, R. (2017).** *Bayesian Knowledge Tracing, Logistic Models, and Beyond: An Overview of Learner Modeling Techniques.*
