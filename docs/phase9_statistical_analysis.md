# Phase 9 Statistical Analysis

## Scope and inferential unit

The focal contrast is the original accuracy-only policy versus middle-budget BC-GPTSwarm (`bc_budget_075`). The optimizer seed is the independent unit for learned-policy comparisons, giving only three paired observations. We report paired raw effects, sample standard deviations, exploratory 95% Student-t intervals with two degrees of freedom, Cohen's dz, and exact two-sided sign-flip p-values. With three pairs, the smallest attainable two-sided exact p-value is 0.25.

The ablation matrix is exploratory and was not preregistered as a confirmatory family, so no ablation-wide null-hypothesis tests are performed. This avoids turning many underpowered comparisons into significance claims.

## Seed-level focal comparison

| Outcome | Mean paired effect (BC - baseline) | SD | 95% CI | Cohen's dz | Exact p |
|---|---:|---:|---:|---:|---:|
| Accuracy | -1.39 pp | 4.81 pp | [-13.34, +10.56] pp | -0.289 | 1.000 |
| Tokens/question | -14.44 | 89.45 | [-236.65, +207.76] | -0.161 | 0.750 |
| Evaluation latency/run | -1.41 s | 10.61 s | [-27.78, +24.95] s | -0.133 | 0.750 |
| Normalized budget violation vs zero | +4.05% | 13.11% | [-28.52, +36.62]% | 0.309 | 0.750 |

The token effect is -5.75% relative to the baseline mean. Every seed-level interval includes zero, and no exact test can reject a zero effect. Cohen's dz values are included only as descriptive standardized effects; with n=3 they are unstable.

## Secondary paired-answer analysis

Across 72 seed-question pairs, BC-GPTSwarm has two unique wins and the baseline has three. Exact McNemar p = 1.000. The conditional matched odds ratio is 0.667 with exact 95% CI [0.056, 5.820]. The paired risk difference is -1.39 percentage points; a 20,000-resample question-cluster bootstrap gives [-5.56, +2.78] percentage points.

This item-level calculation is secondary. It conditions on the three fitted policies and resamples the 24 held-out questions; it does not create 72 independent optimizer replications.

## Interpretation

- The study provides no statistical evidence that middle-budget BC-GPTSwarm changes accuracy, token use, or latency relative to the original policy.
- Failure to reject zero budget violation is not evidence of budget equivalence or compliance. No equivalence margin was predeclared, and two of three seed means exceed the target.
- Broad intervals and the exact-test resolution limit make the null result inconclusive rather than evidence that the methods are identical.
- The Phase 8 descriptive conclusions remain appropriate: the mechanism is operational, but accuracy improvement and tight-budget convergence are not established.
