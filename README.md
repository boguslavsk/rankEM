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
We observed that simple imputation (using global means) dampens variance. A better heuristic estimates missing values using row and column marginals:

$$
\hat{X}_{ij} = \text{RowMean}_i + \text{ColMean}_j - \text{GlobalMean}
$$

* **Pros:** This method captures the additive nature of the model and restores the variance lost by simpler averaging methods. It is computationally instant.
* **Cons:** This method fails under **Biased Missingness** (The "Bias Trap").
    * *Scenario:* If "Honors" students only attempt "Hard" problems, their `RowMean` will be artificially low because they never attempted the easy problems to boost their average.
    * *Result:* The heuristic cannot disentangle whether the student is weak or the problems were simply hard, leading to incorrect rankings.

## 4. The Proposed Solution: Regularized EM Algorithm

To solve the "Bias Trap," we employ a **Regularized Expectation-Maximization (EM)** algorithm. This method handles the circular dependency: to know a student's true ability, we must know the difficulty of the problems they skipped; to know a problem's difficulty, we must know the ability of the students who solved it.

### The Algorithm Steps

1.  **Initialization:**
    Fill all missing entries in the matrix with the global mean $\mu$.

2.  **M-Step (Maximization/Estimation):**
    Calculate the parameters $\theta_i$ and $\beta_j$ that minimize the prediction error on the *observed* data. We apply **Regularization** (Ridge/L2 penalty) to handle the error term $\epsilon$ and prevent overfitting on students with few data points.

$$
    \theta_i = \frac{\sum_{j \in \text{Observed}} (X_{ij} - \mu - \beta_j)}{N_i + \lambda}
$$

Where $N_i$ is the count of problems student $i$ solved, and $\lambda$ is the regularization term.

4.  **E-Step (Expectation/Imputation):**
    Update the values for the missing entries based on the newly estimated parameters:

$$
    X_{ij}^{\text{missing}} = \mu + \theta_i + \beta_j
$$

5.  **Convergence:**
    Repeat steps 2 and 3 until the parameters stabilize (change $< 10^{-4}$).

### Why EM Outperforms Heuristics
The EM algorithm utilizes the entire network of connections in the data. Even if Student A and Student B never solved the same problem, they are linked through Student C, who solved problems common to both. This allows information to "flow" through the matrix, correcting the rankings even in the "Bias Trap" scenario where heuristics fail.

## 5. Implementation Strategy

* **Data Structure:** A sparse matrix of size $120 \times 24$.
* **Hyperparameters:**
    * **Regularization ($\lambda$):** We recommend a value between $1.0$ and $5.0$. This "shrinks" estimates toward the mean for students with very little data, preventing wild guesses based on noise.
* **Validation:**
    Post-calculation, we should plot **Estimated Ability ($\theta$)** against **Raw Score (0-6)**. If the relationship is non-linear (e.g., a steeper slope between 5 and 6), the model has successfully learned that achieving a perfect score is exponentially harder than achieving a mediocre one, without requiring explicit non-linear programming.

## 6. References

* **Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977).** *Maximum likelihood from incomplete data via the EM algorithm.* (The foundational paper).
* **Koren, Y., Bell, R., & Volinsky, C. (2009).** *Matrix Factorization Techniques for Recommender Systems.* (Describes similar techniques used for filling sparse matrices in the Netflix Prize).
* **Rasch, G. (1960).** *Probabilistic Models for Some Intelligence and Attainment Tests.* (The theoretical basis for additive models in testing).
* **Pelánek, R. (2017).** *Bayesian Knowledge Tracing, Logistic Models, and Beyond: An Overview of Learner Modeling Techniques.*
