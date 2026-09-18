# Phase 8 analysis

## Scope and uncertainty

The analysis uses all 18 Phase 6 runs and all 12 additional Phase 7 runs. Each
condition has three graph seeds and 24 held-out questions. Means, sample standard
deviations, and exploratory 95% Student-t intervals are generated from the
committed run CSVs. With only three seeds, the intervals are very wide and no
difference is treated as statistically established.

## Main comparison

| Condition | Accuracy (mean +/- SD) | Tokens/question | Normalized violation | Per-query compliance |
|---|---:|---:|---:|---:|
| Original accuracy-only | 0.250 +/- 0.110 | 251.1 +/- 90.4 | +0.104 | 0.46 |
| Feasible accuracy-only | 0.222 +/- 0.105 | 239.6 +/- 19.6 | +0.053 | 0.53 |
| Fixed penalty | 0.208 +/- 0.083 | 230.4 +/- 21.3 | +0.013 | 0.54 |
| BC, 0.50 target | 0.208 +/- 0.110 | 232.3 +/- 31.9 | +0.288 | 0.38 |
| BC, 0.75 target | 0.236 +/- 0.105 | 236.7 +/- 29.8 | +0.040 | 0.50 |
| BC, 1.00 target | 0.208 +/- 0.083 | 238.9 +/- 18.5 | -0.212 | 0.75 |

The original baseline has the highest main-study mean accuracy, 0.250. The
middle-budget proposed method reaches 0.236, a difference of -0.014, while using
5.8% fewer tokens. This is a small descriptive tradeoff, not an improvement.
Its mean cost still exceeds the target by 4.0%. The high-budget condition is the
only BC setting with clear aggregate slack; the low budget misses by 28.8%.

The low target is 180.4 tokens rather than exactly half the 303.3-token Phase 4
full-graph reference because the predeclared feasibility rule raises it to 105%
of the newly measured 171.8-token direct-path floor.

## Accuracy-cost and Pareto analysis

Across all main and ablation condition means, the descriptive non-dominated set
is:

- dual rate 0.50: accuracy 0.222 at 229.6 tokens/question;
- raw-utility BC: accuracy 0.250 at 232.4 tokens/question; and
- 2T2A BC: accuracy 0.278 at 419.5 tokens/question.

These are operating points, not statistically separated winners. Their accuracy
standard deviations are 0.134, 0.150, and 0.048 respectively. The raw-utility
point matches the original baseline mean while using fewer tokens, but it is an
ablation with a different feasible stochastic evaluation rule and has the
largest seed variance. The 2T2A point buys 0.042 accuracy over 1T1A BC while
using 77% more tokens and violating the middle budget by 84%.

## Accuracy-latency tradeoff

Main-condition summed held-out inference latency ranges only from 27.4 to 30.1
seconds, and its ordering does not consistently match token use. The run-level
latency-accuracy correlation is 0.287. The 2T2A ablation averages 44.9 seconds.
These local CPU timings are descriptive: system load produced one 2,109-second
optimization wall-time outlier for fixed-penalty seed 33 even though its summed
evaluation inference latency was 27.6 seconds. The outlier is retained; provider
latency and tokens are more interpretable than wall-clock optimization time.

## Graph size and agent count

Across all 30 runs, active-agent count correlates 0.424 with accuracy and active
edge count correlates 0.224. These are non-causal associations strongly affected
by the 2T2A condition. At the middle budget, 2T2A averages 0.278 accuracy and
419.5 tokens versus 0.236 and 236.7 for 1T1A. More agents therefore increase
both mean accuracy and resource demand in this reduced setting, without meeting
the resource constraint.

## Budget response and optimization stability

Observed BC evaluation cost increases only modestly with budget: 232.3, 236.7,
and 238.9 tokens for the low, middle, and high targets. Final mean dual values
move in the expected direction: 1.129, 0.427, and 0.005. However, final training
batch cost ratios remain 1.506, 1.321, and 0.950. Twelve iterations are therefore
enough for the high budget but not for convergence at tighter targets.

Changing the dual rate from 0.05 to 0.20 to 0.50 changes final mean dual values
from 0.108 to 0.427 to 1.066, but mean cost remains 233.6, 236.7, and 229.6
tokens. Accuracy means are 0.222, 0.236, and 0.222. This does not show a stable
accuracy advantage for the default rate. Raw utility yields 0.250 accuracy and
232.4 tokens versus 0.236 and 236.7 for centered utility, but its accuracy SD is
0.150; centering does not help in this small run.

## Paired wins and failure cases

Pairing the original baseline and middle-budget BC by seed and question produces
72 comparisons:

- both correct: 15;
- both wrong: 52;
- baseline only correct: 3; and
- BC only correct: 2.

The five disagreements are saved in
`results/processed/disagreement_cases.csv`. BC-only wins occur in high-school
computer science and global facts for seed 11. Baseline-only wins occur in
high-school microeconomics (seed 11), high-school computer science (seed 22),
and electrical engineering (seed 33). The electrical-engineering loss also uses
306 proposed tokens versus 148 baseline tokens, illustrating that extra calls
can hurt both cost and correctness.

There are zero provider errors, unparseable final answers, or disconnected graph
failures in Phases 6-7. The dominant failure mode is ordinary task error: both
methods are wrong in 52 of 72 paired cases. Feasibility conditioning fixes
structural failures but not weak-model reasoning.

## Variance and conclusion

Main-condition accuracy SD ranges from 0.083 to 0.110. Raw utility has SD 0.150
and the high-dual ablation 0.134. Exploratory 95% intervals for original accuracy
(-0.024, 0.524) and middle-budget BC (-0.024, 0.497) overlap almost completely.
The evidence supports only a null conclusion: BC-GPTSwarm is implemented and its
dual responds sensibly to budget pressure, but under the reduced 1T1A local-model
allowance it does not reliably satisfy tight budgets or improve accuracy.
