# Phase 1: Research Direction Analysis

## Decision summary

**Selected direction:** budget-constrained GPTSwarm (BC-GPTSwarm).

> We extend GPTSwarm with a primal--dual token-budget constraint so that task utility is optimized while expected graph-execution cost must remain within a specified budget.

This is deliberately narrower than a generic "efficient multi-agent system." It preserves GPTSwarm's graph representation, node routines, DAG sampler, and REINFORCE optimizer. The required additions are token accounting, a budget-normalized constraint, and a learned non-negative Lagrange multiplier. The output is an accuracy--token trade-off with an explicit budget-feasibility measurement rather than an uncalibrated weighted score.

The literature scan was performed on 2026-09-18. Ratings below concern a small university project built on the Phase 0 MMLU edge-optimization reproduction, not the intrinsic value of each broader research area.

## Rating rubric

- **Novelty:** distinctiveness after accounting for closely related published and preprint work available at the scan date.
- **Difficulty / effort:** engineering and research-design burden relative to the Phase 0 baseline.
- **Compute:** expected API/token and local-compute burden.
- **Measurability:** likelihood of obtaining an interpretable result even if accuracy does not improve.
- **Contribution:** defensibility as a focused research-paper contribution.
- **Failure risk:** chance that the mechanism is unstable, confounded, or yields an inconclusive comparison.
- **Type:** an *upgrade* targets an additional desirable property; a *sidegrade* primarily studies an alternative design without a directional guarantee.

Ratings use Low (L), Medium (M), and High (H). For measurability and contribution, High is favorable; for difficulty, effort, compute, and risk, High is unfavorable.

## Comparative assessment

| # | Candidate | Novelty | Difficulty / effort | Compute | Measurability | Required GPTSwarm change | Experiment complexity | Contribution | Failure risk | Type |
|---:|---|:---:|:---:|:---:|:---:|---|:---:|:---:|:---:|---|
| 1 | Cost-aware graph optimization | L--M | L / L | L | H | Add token/cost term to reward | L | M | L--M | Upgrade |
| 2 | Latency-aware GPTSwarm | M | M / M | M | M | Instrument runtime and add latency term | H | M | H | Upgrade |
| 3 | Dynamic agent pruning | L | H / H | M | H | Query-conditioned node/edge gates | H | M | H | Upgrade |
| 4 | Adaptive graph routing | L--M | H / H | H | H | Router plus multiple graph candidates | H | H | H | Upgrade |
| 5 | Confidence-aware routing | L | M / M | M | M | Confidence estimator and activation rule | H | M | H | Upgrade |
| 6 | Early-exit GPTSwarm | L--M | M / M | L--M | H | Intermediate decisions and stop rule | M--H | M | M--H | Upgrade |
| 7 | Robustness-aware GPTSwarm | M--H | M / M | M--H | H | Corruption model and robust objective | H | H | M--H | Upgrade |
| 8 | Heterogeneous model swarm | L--M | M / M--H | M | H | Per-node model assignment/routing | H | M--H | H | Sidegrade / upgrade |
| **9** | **Budget-constrained GPTSwarm** | **M** | **M / L--M** | **L--M** | **H** | **Token meter, constraint, dual update** | **M** | **M--H** | **M** | **Upgrade** |
| 10 | Memory-augmented GPTSwarm | M | H / H | H | M | Memory store, retrieval, write policy | H | M | H | Sidegrade / upgrade |
| 11 | Consensus alternatives | L | L / L | L--M | H | Swap decision node only | M | L--M | L | Sidegrade |
| 12 | Agent specialization | L--M | M / M | M--H | M | Role prompts or specialist models | H | M | H | Sidegrade / upgrade |
| 13 | Graph sparsification | L | L--M / L--M | L | H | Edge penalty or pruning stage | M | M | L--M | Upgrade |
| 14 | Query-complexity-aware graph selection | L--M | H / H | M--H | M--H | Difficulty estimator and graph policy | H | M--H | H | Upgrade |
| 15 | Multi-objective graph optimization | L--M | H / H | H | H | Multiple costs and Pareto/search machinery | H | H | H | Upgrade |

## Candidate-by-candidate analysis

### 1. Cost-aware graph optimization

- **Idea:** optimize accuracy minus a fixed weighted token or monetary-cost penalty.
- **Strengths:** the smallest code change; token use and cost are directly measurable; almost any run produces an interpretable trade-off.
- **Weaknesses:** choosing the penalty weight is arbitrary and scale-dependent. A single scalarized reward does not guarantee that a requested budget is met.
- **Novelty context:** AgentPrune directly prunes spatial and temporal communication for economy; Optima jointly targets effectiveness and token efficiency; AgentSlimming compresses agent workflows through pruning and cheaper-model substitution. A generic cost penalty would therefore be weakly differentiated.
- **Verdict:** retain as a required control and ablation, not the main contribution.

### 2. Latency-aware GPTSwarm

- **Idea:** include measured end-to-end inference latency in graph optimization.
- **Strengths:** operationally important and less identical to token minimization, especially when branches can execute concurrently.
- **Weaknesses:** hosted API latency depends on queueing, network conditions, retries, caching, provider throttling, and scheduling. Sequential GPTSwarm execution also confounds topology with implementation details.
- **Changes:** timestamp every request, define critical-path versus wall-clock cost, control concurrency, and repeat enough times to estimate noisy latency.
- **Verdict:** useful later as a reported dependent variable; too noisy for the first optimized constraint.

### 3. Dynamic agent pruning

- **Idea:** suppress agents or messages on a per-query basis.
- **Strengths:** can save substantial resources and expose interpretable activation patterns.
- **Weaknesses:** requires a conditional gating policy, not just GPTSwarm's task-level edge probabilities. Training that policy without leaking evaluation labels is non-trivial.
- **Novelty context:** AgentPrune, AgentDropout, and AgentSlimming already cover closely related pruning and elimination mechanisms.
- **Verdict:** too crowded and architecturally larger than the target project.

### 4. Adaptive graph routing

- **Idea:** select or generate a different communication graph for each input.
- **Strengths:** potentially captures genuine heterogeneity in question difficulty and type.
- **Weaknesses:** needs graph-conditioned features, a routing controller, a family of candidate graphs, and enough data to learn routing rather than memorization.
- **Novelty context:** G-Designer learns task-adaptive communication topologies, while MasRouter routes queries across collaboration modes, roles, and LLMs.
- **Verdict:** strong research area, but too large and already well populated.

### 5. Confidence-aware routing

- **Idea:** solicit additional agents only when an initial answer is insufficiently confident.
- **Strengths:** intuitive, query-adaptive, and potentially cheap.
- **Weaknesses:** verbal or logit confidence from current LLM APIs is often miscalibrated; correctness labels can accidentally influence threshold selection. Confidence estimation becomes a second research problem.
- **Novelty context:** Debate Only When Necessary uses confidence to trigger multi-agent debate, and newer work also studies confidence/consensus-driven collaboration.
- **Verdict:** potentially valuable, but less controlled than an observed-token constraint.

### 6. Early-exit GPTSwarm

- **Idea:** stop execution when an intermediate answer reaches sufficient agreement or confidence.
- **Strengths:** easy to explain, saves calls, and gives a clear cost--accuracy curve.
- **Weaknesses:** GPTSwarm's DAG currently defines dependencies before execution. Meaningful stopping requires ordered checkpoints and can favor particular topologies. Consensus may be confidently wrong, especially with correlated agents.
- **Novelty context:** confidence-triggered debate already realizes a close version of this idea.
- **Verdict:** feasible backup direction, but novelty is limited and the stopping criterion adds calibration risk.

### 7. Robustness-aware GPTSwarm

- **Idea:** optimize expected performance under corrupted, adversarial, or failed agent responses.
- **Strengths:** naturally extends the selected adversarial MMLU setting and can yield measurable degradation curves rather than requiring a positive accuracy gain.
- **Weaknesses:** the corruption distribution is a strong design choice. Robustness to synthetic noise may not generalize to real failures, and evaluation requires a matrix of corruption types and severities.
- **Changes:** introduce reproducible corruption operators and optimize worst-case, risk-sensitive, or mixture utility.
- **Verdict:** defensible and moderately novel, but experimental scope is larger than the budget extension.

### 8. Heterogeneous model swarm

- **Idea:** place cheaper and stronger models at different nodes.
- **Strengths:** makes cost/quality trade-offs practically meaningful and can outperform a homogeneous system at equal spend.
- **Weaknesses:** changes both topology and model capability. Provider-specific prices, prompt compatibility, rate limits, and model retirement weaken reproducibility.
- **Novelty context:** MasRouter explicitly includes LLM routing, while AgentSlimming includes lower-cost model substitution.
- **Verdict:** reserve as a future extension after a same-model budget study isolates graph effects.

### 9. Budget-constrained GPTSwarm — selected

- **Idea:** maximize expected task utility subject to an expected token-cost limit, using a learned dual variable rather than a hand-selected fixed penalty.
- **Strengths:** small change to the native stochastic-edge optimizer; actual tokens are observable; budget violation is an explicit success criterion; budget sweeps produce an interpretable frontier even if no method dominates in accuracy.
- **Weaknesses:** discrete graphs and noisy LLM outputs make the primal and dual updates high variance. Infeasible budgets can collapse the graph or drive the multiplier upward indefinitely.
- **Changes:** add resource telemetry, a normalized constraint, a non-negative dual update, a feasibility floor, and budget-specific logging.
- **Experiment complexity:** moderate. The main comparison is accuracy-only GPTSwarm, fixed-penalty GPTSwarm, and the proposed constrained optimizer at predeclared budgets under identical calls, prompts, and data.
- **Novelty context:** efficiency-aware agent optimization is crowded, so the claim must be narrow: explicit expected-budget control inside GPTSwarm's stochastic graph optimization. No priority or state-of-the-art claim will be made.
- **Verdict:** best balance of scope, measurability, and defensibility.

### 10. Memory-augmented GPTSwarm

- **Idea:** let nodes share or retrieve intermediate memories across steps or examples.
- **Strengths:** potentially helpful for long tasks and repeated domains.
- **Weaknesses:** memory schema, retrieval, write policy, retention, contamination, and privacy all become independent variables. MMLU questions offer little natural need for cross-example memory.
- **Verdict:** mismatched to the minimum reproduction and too confounded for a small extension.

### 11. Consensus alternatives

- **Idea:** compare majority vote, confidence weighting, a judge LLM, and related output aggregators.
- **Strengths:** very low implementation risk; differences are easy to measure.
- **Weaknesses:** the study changes the decision node rather than graph optimization. Judge-based methods add cost and model bias, while confidence weighting inherits calibration problems.
- **Verdict:** useful ablation or follow-up, but not a strong standalone contribution.

### 12. Agent specialization

- **Idea:** assign distinct reasoning roles, prompts, subjects, or models to nodes.
- **Strengths:** can improve diversity and reveal which roles survive graph optimization.
- **Weaknesses:** role quality is prompt-sensitive; improvements may be due to prompt engineering rather than topology. Subject-specific training and testing complicate generalization claims.
- **Verdict:** a plausible sidegrade, but harder to attribute causally and broader than the selected constraint.

### 13. Graph sparsification

- **Idea:** minimize or prune communication edges while preserving accuracy.
- **Strengths:** aligns closely with GPTSwarm's edge logits and is inexpensive to evaluate.
- **Weaknesses:** edge count is only a weak proxy for resource use: one retained edge can carry a long response, and removing an edge may not eliminate a node call.
- **Novelty context:** AgentPrune and G-Designer already foreground sparse/economical communication, while AgentSlimming prunes graph nodes.
- **Verdict:** include edge count as a diagnostic and as a proxy-cost ablation; do not use it as the main contribution.

### 14. Query-complexity-aware graph selection

- **Idea:** predict question difficulty and select a smaller or larger graph.
- **Strengths:** directly captures the intuition that easy questions need less collaboration.
- **Weaknesses:** requires a leakage-resistant difficulty target, a trained or calibrated predictor, multiple graph policies, and evaluation of routing regret. Difficulty may be model-specific and unstable.
- **Novelty context:** overlaps adaptive routing, confidence-triggered debate, and MasRouter's query-dependent collaboration selection.
- **Verdict:** potentially strong but exceeds the desired engineering and data budget.

### 15. Multi-objective graph optimization

- **Idea:** jointly optimize accuracy, token cost, monetary cost, and latency, possibly producing a Pareto set.
- **Strengths:** represents deployment trade-offs more faithfully than a single objective.
- **Weaknesses:** token count and monetary cost are correlated while latency is noisy; objective scaling and Pareto-search choices multiply runs and analysis. The result may be a broad framework without a clean causal finding.
- **Verdict:** too broad. The selected one-constraint method is a controlled first step; latency and dollar cost remain reported outcomes.

## Selection rationale

BC-GPTSwarm wins because it satisfies five constraints simultaneously:

1. **Small:** it changes the optimizer and telemetry, not the agent architecture.
2. **Defensible:** it asks whether an explicit budget can be met, rather than claiming that efficiency-aware MAS optimization is new.
3. **Measurable:** accuracy, tokens, and violation are numeric and available for every execution.
4. **Fair:** the baseline and proposed method can use the same nodes, prompts, models, examples, and graph sampler.
5. **Informative under a null result:** failure to beat fixed scalarization still reveals budget feasibility, dual stability, and the graph-size/cost relationship.

## Closest verified primary work

- Zhuge et al., [GPTSwarm: Language Agents as Optimizable Graphs](https://proceedings.mlr.press/v235/zhuge24a.html), ICML 2024.
- Zhang et al., [Cut the Crap: An Economical Communication Pipeline for LLM-based Multi-Agent Systems](https://arxiv.org/abs/2410.02506), 2024 (AgentPrune).
- Chen et al., [Optima: Optimizing Effectiveness and Efficiency for LLM-Based Multi-Agent System](https://arxiv.org/abs/2410.08115), 2024.
- Zhang et al., [G-Designer: Architecting Multi-agent Communication Topologies via Graph Neural Networks](https://arxiv.org/abs/2410.11782), 2024.
- Yue et al., [MasRouter: Learning to Route LLMs for Multi-Agent Systems](https://arxiv.org/abs/2502.11133), 2025.
- Wang et al., [AgentDropout: Dynamic Agent Elimination for Token-Efficient and High-Performance LLM-Based Multi-Agent Collaboration](https://arxiv.org/abs/2503.18891), 2025.
- Eo et al., [Debate Only When Necessary: Adaptive Multiagent Collaboration for Efficient LLM Reasoning](https://arxiv.org/abs/2504.05047), 2025.
- Chen et al., [AgentSlimming: Towards Efficient and Cost-Aware Multi-Agent Systems](https://aclanthology.org/2026.acl-long.1387/), ACL 2026.

These sources establish adjacent work; they do not prove that the selected formulation is unique. The paper will state a narrow methodological contribution and will revise its positioning if a closer prior method is found.
