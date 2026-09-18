# Research Proposal: BC-GPTSwarm

## Title candidate

**BC-GPTSwarm: Budget-Constrained Edge Optimization for Language-Agent Graphs**

## One-sentence contribution

We extend GPTSwarm with a primal--dual token-budget constraint so that task utility is optimized while expected graph-execution cost must remain within a specified budget.

## Research question

Can a primal--dual constrained version of GPTSwarm learn communication graphs that satisfy explicit expected token budgets while retaining more MMLU accuracy than simple budget-agnostic or fixed-penalty alternatives under matched conditions?

This question separates two outcomes:

1. **constraint control:** how closely the learned policy meets a requested budget; and
2. **task trade-off:** how much accuracy is retained at that budget.

The proposal does not assume that the method will improve raw accuracy or dominate every scalarized baseline.

## Hypotheses

- **H1 (primary):** across feasible target budgets, the primal--dual method will have a lower absolute expected-budget violation and a higher fraction of budget-compliant evaluation runs than a fixed-penalty method whose coefficient is not tuned separately for every budget.
- **H2:** tightening the target budget will reduce mean token usage and usually reduce active graph structure, with a measurable accuracy--resource trade-off.
- **H3 (exploratory):** at a matched realized token count, constrained optimization may retain more accuracy than random graph sampling because it can preserve useful communication while suppressing costly or adversarial paths.

H3 is intentionally exploratory. A null or negative accuracy result will be reported directly.

## Original GPTSwarm baseline

The baseline is the Phase 0 adversarial MMLU edge-optimization setting:

- truthful and adversarial input--output agents;
- a majority-vote decision node;
- fixed internal agent edges and stochastic candidate inter-agent edges;
- Bernoulli edge logits with cycle-rejecting DAG sampling;
- task utility equal to answer correctness; and
- REINFORCE/Adam optimization of edge probabilities.

The mandatory minimum uses one truthful and one adversarial agent (1T1A). The preferred main setting uses three truthful and three adversarial agents (3T3A) if the smoke-test budget permits. Direct-answer, full-connectivity, random-connectivity, and accuracy-only learned-connectivity baselines use the same data, prompts, model, decoding settings, and evaluation procedure.

## Proposed modification

BC-GPTSwarm adds three components without redesigning the graph:

1. a resource meter that records input, output, and total tokens for every model call;
2. a budget-normalized constraint included in the stochastic edge-policy update; and
3. a non-negative dual variable updated from the observed batch-level budget violation.

The graph sampler, node routines, topological executor, prompts, and decision rule remain unchanged. Monetary cost and latency are logged as secondary outcomes but are not placed in the first constrained objective, avoiding a multi-objective redesign.

## Mathematical objective

For input $x$, sampled feasible graph $G \sim D_{\theta}$, task utility $U(G,x)$, measured total-token cost $C(G,x)$, and target token budget $B>0$, solve

$$
\max_{\theta}\; \mathbb{E}_{x,G}[U(G,x)]
\quad \text{subject to} \quad
\mathbb{E}_{x,G}[C(G,x)] \le B.
$$

Use the dimensionless constraint

$$
g(\theta)=\mathbb{E}_{x,G}\left[\frac{C(G,x)}{B}-1\right]\le 0
$$

and Lagrangian

$$
\mathcal{L}(\theta,\lambda)=
\mathbb{E}_{x,G}\left[
U(G,x)-\lambda\left(\frac{C(G,x)}{B}-1\right)
\right], \qquad \lambda\ge 0.
$$

For a sampled mini-batch, the edge-policy gradient is estimated with REINFORCE:

$$
\widehat{\nabla_{\theta}\mathcal{L}}
=\frac{1}{M}\sum_{i=1}^{M}
\left(A_i-\lambda\left(\frac{C_i}{B}-1\right)\right)
\nabla_{\theta}\log p_{\theta}(G_i),
$$

where $A_i$ is a centered task-utility advantage. The dual update is

$$
\lambda \leftarrow
\left[\lambda+\eta_{\lambda}
\left(\frac{1}{M}\sum_{i=1}^{M}\frac{C_i}{B}-1\right)
\right]_{+}.
$$

Centering, normalization, optimizer choice, and update frequency will be fixed in configuration and logged. The implementation will verify gradient signs with a deterministic dummy model before API experiments.

### Feasibility rule

A zero-call or disconnected graph is not a scientifically meaningful way to satisfy a budget. Before optimization, a smoke test will estimate the minimum cost of a valid direct-answer execution. Budgets below that empirical floor are declared infeasible and excluded. Every learned graph must retain at least one valid input-to-decision path. The same feasibility rule will be applied to the accuracy-only and fixed-penalty controls so it cannot favor the proposal.

Target budgets will be predeclared after smoke calibration as ratios of a named reference execution, rather than chosen after seeing test accuracy. The initial candidate ratios are 0.50, 0.75, and 1.00 of the accuracy-only learned-graph token use, subject to the feasibility floor. If this circular reference is impractical during training, the full-graph or direct-answer calibration cost will be used and documented before the main runs.

## Independent variables

### Primary

- optimization method: accuracy-only, fixed token penalty, or primal--dual constraint;
- target budget ratio $B/B_{\mathrm{ref}}$;
- graph scale: 1T1A and, if affordable, 3T3A;
- random seed.

### Ablation variables

- cost signal: actual total tokens versus active-edge count proxy;
- fixed scalarization coefficient $\alpha$;
- dual learning rate $\eta_{\lambda}$;
- dual update frequency and optional clipping;
- utility centering/advantage baseline on versus off.

Model, prompts, temperature, maximum output tokens, dataset split, question order, graph candidates, iteration count, and evaluation parser will otherwise be controlled.

## Dependent variables

- MMLU answer correctness and aggregate accuracy;
- input, output, and total tokens per question;
- normalized and absolute budget violation;
- budget-compliance indicator and compliance rate;
- estimated or actual API cost;
- end-to-end latency;
- number of active nodes and edges;
- learned edge probabilities and graph connectivity;
- optimizer wall-clock time and dual-variable trajectory.

## Evaluation metrics

### Primary metrics

1. mean MMLU accuracy;
2. mean total tokens per question;
3. mean normalized violation $\mathbb{E}[C/B-1]$;
4. absolute budget error $|\mathbb{E}[C]-B|/B$; and
5. fraction of evaluation batches/runs satisfying the budget.

### Secondary metrics

- input/output-token decomposition;
- dollar cost under a frozen price table, labeled as an estimate where appropriate;
- median and mean latency with dispersion;
- active nodes/edges;
- accuracy at matched realized cost;
- optimizer stability across seeds; and
- an accuracy--token plot with nondominated operating points (without claiming a continuous Pareto frontier from sparse samples).

Results will report mean and standard deviation across seeds where affordable. Confidence intervals and paired tests will be used only if the sample size and dependence structure justify them.

## Datasets and tasks

The primary task is the same MMLU multiple-choice setting selected in Phase 0. Optimization and model-selection examples must remain separate from final evaluation examples. The exact subjects, split construction, sample counts, and dataset revision will be fixed in Phase 3 before baseline experiments.

The original study's 153-question validation setting is the upper reference. A smaller predeclared subset is acceptable if API cost requires it, provided all methods receive identical examples and the limitation is reported. No second benchmark is required for the minimum viable study; adding one is conditional on completing the primary evaluation with adequate seeds.

## Baselines

1. **Direct answer:** one model answers without swarm communication.
2. **Full graph:** all feasible candidate communication edges active.
3. **Random graph:** edge inclusion probability fixed at 0.5.
4. **GPTSwarm accuracy-only:** the reproduced original edge objective.
5. **Fixed-penalty GPTSwarm:** maximize $\mathbb{E}[U-\alpha C/B_{\mathrm{ref}}]$ for predeclared $\alpha$ values.
6. **BC-GPTSwarm:** the proposed primal--dual constraint for each feasible $B$.

The fixed-penalty baseline is essential: it tests whether the dual update adds value beyond simply subtracting token cost from reward.

## Ablation experiments

1. **Constraint mechanism:** learned dual variable versus fixed penalty.
2. **Cost fidelity:** measured tokens versus active-edge-count proxy.
3. **Budget severity:** each predeclared feasible budget ratio.
4. **Dual dynamics:** at least two $\eta_{\lambda}$ values and, if instability appears, clipped versus unclipped $\lambda$.
5. **Graph scale:** 1T1A versus 3T3A if affordable.
6. **Variance reduction:** centered task advantage versus raw binary utility.

Each ablation changes only one named factor where possible. Failed or unstable configurations remain in the experiment log.

## Expected contribution

The expected contribution is a focused empirical and methodological study of explicit resource constraints in GPTSwarm's stochastic communication-graph optimizer:

- a minimal, reproducible token-instrumented GPTSwarm edge baseline;
- a primal--dual constrained objective that targets a user-specified expected token budget;
- a controlled comparison with accuracy-only and fixed-penalty optimization; and
- evidence about budget feasibility, accuracy retention, graph sparsity, and optimization stability across budget levels.

This is a modest extension. Efficiency-aware multi-agent optimization, routing, pruning, and model substitution already exist. The intended contribution is not a claim to invent cost-aware multi-agent systems, but to test explicit budget satisfaction within the GPTSwarm formulation under controlled conditions.

## Risks and mitigations

| Risk | Consequence | Mitigation |
|---|---|---|
| Budget is below the cost of a valid execution | Degenerate/disconnected graphs or runaway $\lambda$ | Measure a feasibility floor; require a valid decision path; declare infeasible budgets |
| Token cost depends weakly on sampled edges | Little optimizer signal | Log active calls/edges; compare actual-token and edge-proxy signals; inspect cost variance before full runs |
| REINFORCE and dual updates are unstable | Oscillating cost and poor convergence | Normalize constraints, use an advantage baseline, sweep a small predeclared set of dual rates, and retain trajectories |
| Fixed-penalty tuning is unfair | Misleading advantage for the proposal | Give it a predeclared coefficient grid and equal optimization-call allowance; compare at matched realized cost |
| API/model nondeterminism | High seed variance | Pin model identifier/settings, cache when scientifically valid, log raw responses, use multiple seeds where affordable |
| Model retirement or price changes | Weak reproducibility | Record exact identifiers and dates; separate token metrics from time-varying dollar estimates |
| Reduced MMLU subset | Low statistical power | Predeclare samples, use paired questions, report uncertainty, avoid significance claims when underpowered |
| Related work is closer than currently known | Novelty overstatement | Keep claims narrow, update the review before submission, and release the study as an empirical extension |
| Constrained method only matches fixed penalty | No superiority result | Treat budget calibration and stability as findings; report the null result rather than retuning post hoc |

## Phase 1 go/no-go criteria for implementation

Proceed to implementation only if the Phase 3 smoke test confirms that:

1. provider usage metadata exposes reliable input and output token counts, or an explicitly documented tokenizer estimate is available;
2. at least two non-degenerate feasible budget levels exist between a valid minimum execution and the full/reference graph;
3. candidate graphs produce enough cost variation for an optimizer to observe; and
4. the accuracy-only baseline executes end to end.

If condition 2 or 3 fails, the predefined fallback is a fixed cost-aware scalarization study with actual-token versus edge-count objectives. That fallback would be labeled as a reduced contribution, not silently substituted.
