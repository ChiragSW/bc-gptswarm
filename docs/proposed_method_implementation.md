# Phase 5: BC-GPTSwarm implementation

## Scope

Phase 5 adds the selected expected-token-budget extension without changing the
model prompts, node routines, candidate edge order, cycle-rejection rule, or
majority-vote decision used by the baseline. The implementation lives in
`src/gptswarm_ext/constrained.py`; the baseline accuracy-only optimizer remains
available unchanged.

## Objective and updates

For a sampled feasible graph `G`, binary task utility `U`, measured total-token
cost `C`, and budget `B`, BC-GPTSwarm maximizes expected utility subject to
`E[C] <= B`. Its per-sample policy score is

    (U - batch_mean(U)) - lambda * (C / B - 1).

Adam performs gradient ascent through REINFORCE by minimizing the negative
score-weighted graph log-probability. After each batch, the scalar dual variable
is updated by

    lambda <- max(0, lambda + dual_lr * (batch_mean(C) / B - 1)).

The token count is treated as observed black-box cost; no gradient is propagated
through provider execution. Centering applies only to utility. The normalized
cost violation is retained because its constant term has zero expected score
gradient but is needed by the dual update.

## Feasible graph distribution

A sampled graph is feasible only when at least one agent reaches the decision
node. Training uses rejection sampling conditioned on this event rather than
counting disconnected zero-call graphs as cheap solutions. Because edges into
the decision sink cannot create a cycle, the feasibility probability has the
closed form

    P(F) = 1 - product_(e into decision) (1 - p_e).

The training log-probability is `log p(G) - log P(F)`. Including the normalizer
correction makes the score function correspond to the conditional policy rather
than the unconditioned Bernoulli policy. At evaluation time, thresholding is
used as in the baseline. If thresholding removes every decision edge, the
highest-probability decision edge is selected and the intervention is logged in
the graph mask.

## Configuration and telemetry

`configs/proposed_smoke.yaml` exposes:

- total-token budget;
- primal and dual learning rates;
- initial non-negative dual value;
- batch size and iteration count;
- centered-versus-raw utility switch;
- feasibility rule and probability threshold.

The proposed runner first measures the mean truthful direct-path cost on the
optimization split. A budget below this floor is rejected before optimization.
Raw JSONL records contain the prompt specification, dataset and configuration
hashes, source commits, graph masks, node responses, token counts, normalized
violations, gradients, edge probabilities, and dual trajectory. The summary CSV
adds mean-token budget error and per-query compliance.

## Complexity

Let `E` be the number of candidate edges, `M` the batch size, `I` the number of
iterations, `q=P(F)` the feasible-graph probability, and `T_exec` the graph's
model-execution cost. The current simple cycle check makes one graph draw
`O(E^2)` in the worst case. Conditional rejection sampling therefore costs
`O(E^2/q)` in expectation. End-to-end optimization costs
`O(I M (E^2/q + T_exec))`; model execution dominates in practice. Parameter
memory is `O(E)`, excluding persisted execution traces.

## Phase 5 validation boundary

Offline tests verify the feasibility condition, acyclicity, deterministic
threshold repair, projected dual update, finite gradients, and cost-sensitive
policy behavior. The deterministic mock smoke tier is a software sanity check,
not evidence that the method improves accuracy or satisfies budgets with a real
model. Those controlled comparisons belong to Phase 6.
