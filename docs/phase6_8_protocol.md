# Phases 6-8 controlled study protocol

This protocol was frozen before running the Phase 6 and Phase 7 real-model
conditions. It uses the same committed 24-example MMLU development and
validation snapshots, prompts, parser, local `qwen2.5:0.5b` model, temperature
zero, four-token cap, model seed, graph seeds, and 12-by-4 optimization allowance
as the Phase 4 behavioral reproduction.

## Main comparison

Each condition runs with graph seeds 11, 22, and 33:

1. `original_accuracy`: publication-style raw-accuracy REINFORCE and thresholded
   evaluation. This is the reproduced GPTSwarm baseline.
2. `feasible_accuracy`: centered-accuracy optimization under the feasible
   conditional policy. This controls for validity conditioning.
3. `fixed_penalty`: the same feasible policy with fixed coefficient 0.25 on
   normalized token cost. This controls for a scalar cost penalty.
4. `bc_budget_050`, `bc_budget_075`, and `bc_budget_100`: BC-GPTSwarm with
   target ratios 0.50, 0.75, and 1.00 of the Phase 4 full-graph mean of
   303.2917 tokens per validation example.

Every operational budget is the larger of its reference ratio and 105% of the
newly measured truthful direct-path mean on the optimization set. This rule is
predeclared so the lowest target cannot become a disconnected or structurally
infeasible budget. BC-GPTSwarm uses primal rate 0.1, dual rate 0.2, initial dual
value zero, and centered utility. Condition order rotates by seed to reduce
warm-up confounding. Responses are not cached.

The primary comparison is original accuracy-only GPTSwarm versus BC-GPTSwarm.
The feasible-accuracy and fixed-penalty conditions identify whether any observed
difference is caused by valid-graph conditioning, generic cost pressure, or the
dynamic dual mechanism.

## Ablations

Additional real-model conditions use the middle budget and the same three seeds:

- raw rather than centered utility;
- dual learning rate 0.05;
- dual learning rate 0.50;
- a larger 2-truthful/2-adversarial graph.

Together with the main-study rows, these isolate budget level, objective,
utility centering, dual rate, graph scale, and agent count. No condition will be
removed because of accuracy, cost, failure, or convergence behavior.

## Metrics and analysis

Every run records accuracy, input/output/total tokens, local API cost, summed
provider latency, active agents, active edges, candidate graph size,
optimization time, expected-budget violation, per-query compliance, failures,
edge probabilities, and the complete optimization trajectory. Aggregation will
report means, sample standard deviations, and exploratory 95% Student-t
intervals over three seeds. With only three seeds and 24 held-out questions,
intervals will be treated as descriptive rather than confirmatory.

Phase 8 will report accuracy-cost and accuracy-latency operating points, a
descriptive Pareto set, graph-size relationships, budget response, seed
variance, optimization trajectories, and paired question-level wins and failure
cases. No statistically weak difference will be described as a major gain.
