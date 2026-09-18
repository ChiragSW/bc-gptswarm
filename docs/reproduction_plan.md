# Minimal GPTSwarm Reproduction Plan

## Goal

Reproduce the central, scientifically relevant behavior of GPTSwarm before adding any extension: a probabilistic DAG of language agents whose inter-agent edges are optimized by REINFORCE and whose learned topology improves over unoptimized graph controls on a fixed task.

This plan deliberately does not attempt to reproduce every paper table.

## Chosen scope: adversarial MMLU edge optimization

The minimum baseline is the paper's MMLU adversarial experiment:

- truthful input-output agents;
- deliberately adversarial input-output agents;
- a majority-vote final decision node;
- potential directed edges between agents and into the decision node;
- independent edge logits, sequential DAG-constrained Bernoulli sampling;
- REINFORCE updates with Adam;
- direct-answer, full-graph, random-graph, and optimized-graph evaluations.

This is the best minimum because it exercises the paper's main graph representation, message passing, graph sampling, and edge optimizer. Mini Crosswords is much more expensive, HumanEval emphasizes node optimization instead of graph topology, and GAIA introduces search, files, multimodality, and extra API dependencies without optimizing the graph.

## Reproduction claim and non-claim

We will test whether the implementation reproduces the *qualitative behavior* reported by GPTSwarm under a documented, currently available model: learned edge probabilities should suppress harmful communication and match or improve on comparable random/full graphs.

We will not claim an exact numerical replication of 2024 GPT-4 Turbo results unless the original immutable model snapshot is demonstrably available. Replacement-model results will be labeled a controlled reproduction, not an exact replication.

## Source baseline

- Preserve an unmodified reference to publication-era revision `1e1e5899d51bde690672417cf92aa4de7b210422`.
- Record any compatibility changes as local patches in a separate baseline adapter layer.
- Keep original/baseline code visibly separate from the future proposed method.
- Record the current official revision as contextual evidence only: `c23a827f561c934ce21dd950408f7606aa4a8821`.

## Experimental matrix

### Mandatory smoke tier

Use five fixed MMLU examples and one small optimization cycle to validate:

1. direct-answer execution;
2. composite graph execution;
3. acyclic graph sampling;
4. majority-vote aggregation;
5. nonzero gradients on edge logits;
6. optimizer updates;
7. token, latency, graph, and error logging.

The smoke tier validates software only and must not be reported as research evidence.

### Mandatory baseline tier

Run the following with a 1T1A swarm:

| Condition | Purpose |
|---|---|
| Direct answer | Single-agent accuracy reference |
| Full graph | Dense orchestration control |
| Random graph (`p=0.5`) | Unoptimized stochastic topology control |
| Optimized graph | Reproduced GPTSwarm edge optimizer |

Evaluate all conditions on the same fixed validation subset. Use the same provider, model, prompts, temperature, answer parser, retry policy, and concurrency. Cache only when the cache key includes every scientifically relevant request parameter.

### Preferred confirmation tier

If the measured budget permits, repeat the optimized/random comparison with the paper's 3T3A configuration. The official paper used 200 iterations with four graph samples per iteration and evaluated 153 validation questions. We will scale toward that configuration only after measuring cost in the smoke tier.

## Dataset protocol

1. Use the official repository's MMLU downloader and adapter initially.
2. Record the upstream URL, archive checksum, extraction date, subject list, and exact row ordering.
3. Keep optimization and evaluation splits disjoint (`dev` for optimization, `val` for evaluation, matching the official driver).
4. Predeclare any smaller subset by deterministic index before running any method.
5. Store the selected example IDs or content hashes so every condition uses the identical subset.

## Graph and optimizer fidelity

The baseline must preserve:

- directed acyclic composite graphs;
- fixed internal agent edges;
- potential inter-agent edges and eligible final-node edges;
- one learnable logit per potential edge;
- sigmoid Bernoulli sampling;
- cycle rejection during sequential realization;
- log-probability accumulation for sampled decisions;
- task accuracy as the sole optimization utility;
- Adam with learning rate 0.1 for the MMLU reproduction;
- majority vote for the adversarial experiment.

Potential-edge ordering must be deterministic and logged because sequential cycle rejection makes the graph distribution order-dependent.

## Reproducibility controls

- Seed Python, NumPy, and PyTorch explicitly.
- Use at least three predeclared seeds for reported stochastic comparisons if budget permits.
- Pin Python and dependencies in a lock file.
- Pin the model/version or save the provider's returned model identifier.
- Save prompts and request parameters with each run.
- Log every sampled edge mask, log probability, utility, learned logit/probability, raw answer, parsed answer, target, token count, latency, retry, and exception.
- Use a run manifest containing Git commit, dirty-worktree state, config checksum, dataset checksum, environment details, and timestamp.
- Do not silently continue after a failed node in research runs; classify and report failed examples.

## Metrics

Primary:

- MMLU accuracy;
- optimized minus random accuracy;
- optimized minus full-graph accuracy;
- recovery relative to direct-answer accuracy.

Secondary diagnostics:

- prompt, completion, and total tokens;
- estimated/actual API cost;
- end-to-end and per-node latency;
- active nodes and edges;
- learned probabilities by edge type;
- failure/retry rate;
- optimization wall-clock time.

Resource metrics are diagnostics during reproduction. They must not alter the baseline reward; doing so would be the later proposed method, not GPTSwarm.

## Statistical plan

- Treat questions as paired observations across conditions whenever deterministic caching/model settings allow.
- Report accuracy per seed and aggregate mean and standard deviation across seeds.
- Report bootstrap confidence intervals over questions, stratified by seed when appropriate.
- For paired binary outcomes, consider McNemar's test only when sample size and discordant counts make it meaningful.
- Do not infer significance from overlapping or non-overlapping error bars alone.

The exact test will be finalized before main experiments, not selected after viewing results.

## Compute and cost gate

Before any full run:

1. run the five-example smoke tier;
2. measure calls, tokens, latency, and current provider cost;
3. extrapolate the mandatory and preferred tiers;
4. set a written maximum budget and stop condition;
5. obtain user approval before incurring material API cost.

The paper reports about USD 5.32 for 3T3A optimization and USD 1.82 for its inference using historical prices/models, but those values are not valid current quotes. The reproduction budget will use observed current usage.

## Anticipated deviations

| Issue | Planned treatment |
|---|---|
| Original hosted models may be retired | Use a pinned available replacement; label as behavioral reproduction |
| Provider sampling may remain nondeterministic | Repeat seeds/runs and retain raw responses |
| MMLU driver lacks complete seeding | Add a baseline compatibility wrapper without changing the algorithm |
| Dependency set is old and loosely locked | Create a tested Python 3.10 lock/constraints file |
| Node exceptions may be swallowed | Add research-run failure accounting; do not convert failures into favorable scores |
| Static historical price table | Compute costs from logged usage and a dated pricing configuration |
| Sequential edge order can vary | Sort and log potential edges deterministically |

Every actual deviation will be appended to a deviation log before results are interpreted.

## Implementation stages for later phases

### Stage A: frozen baseline adapter

- Vendor or reference only the required official graph, node, swarm, edge-distribution, MMLU-agent, decision, and evaluator paths.
- Add provider abstraction, deterministic configuration, and structured logging around them.
- Avoid rewriting the graph engine unless compatibility requires a narrowly documented patch.

### Stage B: offline tests

- Use mock LLM responses to test graph execution, DAG preservation, voting, edge sampling, gradients, and logs without API cost.
- Verify that high-probability harmful edges can be suppressed in a synthetic deterministic task.

### Stage C: API smoke test

- Run the five-example smoke tier.
- Freeze prompts/config after the smoke test unless a documented correctness defect is found.

### Stage D: baseline reproduction

- Run the predeclared experimental matrix.
- Write all raw results to machine-readable files and the experiment ledger.
- Compare trends with the paper without tuning toward its reported number.

## Acceptance criteria before proposing an extension

The baseline is ready only when:

1. all offline tests pass;
2. a sampled graph is always acyclic;
3. the same saved graph and cached responses replay identically;
4. edge logits receive finite gradients and change in the expected direction on a synthetic task;
5. direct, full, random, and optimized conditions complete under one shared evaluation protocol;
6. raw outputs and all required usage/graph metadata are present;
7. costs are within the approved cap;
8. observed deviations from the paper are documented.

## Expected artifacts in Phases 3-4

- baseline source/adapters and unit tests;
- pinned environment files;
- versioned YAML experiment configurations;
- dataset manifest and checksums;
- `experiments/experiment_log.csv`;
- raw JSONL traces;
- `results/baseline_results.csv`;
- generated tables/figures;
- paper updates describing the exact implementation and observed baseline results.

## Phase 1 recommendation

Evaluate cost-aware edge optimization first, with budget-constrained or early-exit execution as the closest competing direction. A likely one-sentence contribution is:

> We extend GPTSwarm's edge objective with an explicit resource penalty so that graph quality is optimized jointly with communication cost.

This is a recommendation for gap analysis, not a selected contribution. Phase 1 must compare it systematically against the other candidate directions before committing to a method.

