# GPTSwarm: Original Paper Analysis

## Scope and source traceability

This document analyzes *GPTSwarm: Language Agents as Optimizable Graphs* (Zhuge et al., ICML 2024; arXiv:2402.16823v3). The attached 25-page v3 PDF was treated as the authoritative paper. The official repository at <https://github.com/metauto-ai/GPTSwarm> was inspected separately as implementation evidence.

- Paper version: arXiv:2402.16823v3, dated 22 August 2024.
- Publication: Proceedings of ICML 2024, PMLR 235:62743-62767.
- Publication-era repository revision inspected by date: `1e1e5899d51bde690672417cf92aa4de7b210422` (23 August 2024).
- Current repository revision inspected: `c23a827f561c934ce21dd950408f7606aa4a8821` (5 February 2026).
- The ten commits between those revisions modify search support, packaging, and documentation, but not the core graph, edge optimizer, node optimizer, or experiment drivers.

Claims below distinguish the paper's stated method and results from observations made while inspecting the repository.

## 1. Problem statement

LLM agent systems often encode reasoning strategies and multi-agent workflows in separate, manually engineered codebases. GPTSwarm asks whether these systems can be represented in a common computational abstraction and then improved automatically. Its answer is to model operations as nodes, agents as directed graphs of operations, and a multi-agent swarm as a composite graph whose inter-agent communication edges can be optimized.

The paper targets two kinds of manual design:

1. node behavior, usually controlled by prompts; and
2. orchestration, controlled by which operations or agents communicate.

## 2. GPTSwarm architecture

### 2.1 Graph hierarchy

A single agent is a directed acyclic graph

`G = (N, E, F, o)`,

where `N` is the node set, `E` is the directed edge set, `F = {f_n}` assigns a computational routine to every node, and `o` is an output node. A routine may query an LLM, call a tool or API, process a file, or execute another operation.

Given `K` agent graphs, GPTSwarm creates a composite graph from the union of their nodes, internal edges, and routines. Additional inter-agent edges define communication. The composite graph is also restricted to a DAG and has a designated output/decision node.

### 2.2 Nodes

A node receives the original task input `x` and outputs from its predecessor nodes, applies its routine, then passes its output to successors. In the repository, `Node` maintains predecessor/successor lists, inputs, and outputs and executes asynchronously. Concrete operations include direct answering, adversarial answering, chain-of-thought steps, reflection, branching, web search, file analysis, and final-decision aggregation.

### 2.3 Edges and message passing

An edge is both an execution dependency and an information channel. A successor is scheduled only after its useful predecessors execute. Node outputs become successor inputs; nodes may process predecessor outputs separately or combine them. Inter-agent edges therefore change both the information available to downstream agents and the effective workflow.

### 2.4 Graph execution

The paper's Algorithm 1 executes nodes in topological order. The repository computes in-degrees for nodes that can reach an output node, seeds each graph input node with a copy of the input, executes ready nodes asynchronously, and collects output-node results. Each node has retry/timeout handling, although non-timeout exceptions are logged and can lead to missing outputs.

### 2.5 Swarm construction

The repository's `Swarm` instantiates named agents through registries, inserts their graphs into a `CompositeGraph`, adds a final decision operation, and enumerates potential edges. With edge optimization enabled, the potential set includes directed node pairs between different agents plus eligible edges into the decision node. Internal agent edges may remain fixed. Without edge optimization, agent output nodes connect directly to the decision node.

## 3. Algorithm and pipeline

The end-to-end edge-optimization path is:

1. Construct one or more task-specific agent graphs.
2. Compose them and enumerate legal potential inter-agent connections.
3. Assign a Bernoulli logit to each potential edge.
4. Sample a DAG by considering potential edges sequentially, accepting a sampled edge only if it does not introduce a cycle.
5. Execute the sampled graph on one or more task examples.
6. Convert task performance to a scalar utility.
7. update edge logits with a policy-gradient estimator and Adam;
8. evaluate a graph sampled from the learned distribution or threshold the learned probabilities.

Node optimization is a separate loop:

1. execute a fixed graph on task examples;
2. store node-level input/output histories and final task feedback;
3. propose prompt or demonstration changes for individual nodes; and
4. retain the better candidate according to a node/task evaluator.

The paper presents edge and node optimization as general interfaces. The concrete optimizers are task-specific in important details.

## 4. What exactly is optimized

### 4.1 Edge parameters

For `d` potential edges, GPTSwarm learns `d` real-valued logits. After a sigmoid, each gives a Bernoulli inclusion probability. The internal topology of each agent is fixed in the reported edge experiments; the learned variables control additional edges between agents and, in the implementation, eligible links to the final decision node.

The target is expected task utility:

`max_theta E[G ~ D_theta][u_tau(G)]`,

where `D_theta` is the induced distribution over feasible DAGs and `u_tau` is task utility.

### 4.2 Node parameters

The conceptual node parameters are prompts `p_n` attached to routines `f_n^(p_n)`. The update operator receives a node's current prompt, a natural-language description of its function, and accumulated execution examples. In HumanEval, the concrete implementation selects up to four demonstrations and may revise prompts based on recent failures.

### 4.3 Not optimized in the reported work

The reported edge method does not jointly learn model choice, token budget, latency, internal agent topology, tool selection policy, or graph size under an explicit resource constraint. Cost and latency are measured in the appendix but do not enter the objective.

## 5. Training and optimization mechanism

### 5.1 Edge optimization

The paper estimates the gradient with REINFORCE:

`grad_theta E[u(G)] ~= (1/M) sum_i u_hat(G_i) grad_theta log p_theta(G_i)`.

Adam replaces the vanilla gradient-ascent update in the experiments (`beta_1 = 0.9`, `beta_2 = 0.999`, task-dependent learning rate). The repository minimizes the negative log-probability-weighted utility. One optimizer path subtracts a moving-average baseline; the MMLU evaluator path uses raw binary utility. DAG feasibility is enforced during sequential edge sampling, so later edge probabilities are conditioned on earlier accepted edges.

### 5.2 Node optimization

The paper's generic Algorithm 3 treats prompt improvement as an operator `I(h_n, p_n, d_n)`. The HumanEval implementation updates every four problems, labels recent node traces positive or negative from final unit-test success, proposes prompt/demonstration candidates, evaluates them on recent inputs, and keeps the highest-scoring choice. Mini Crosswords uses UCB1 to choose demonstration examples after edge optimization.

### 5.3 Important implementation observations

- The core graph optimizer uses PyTorch autograd only for graph-sampling log probabilities; the LLM itself is a black box.
- Edge sampling order matters because cycle checks can make later edges infeasible.
- The current MMLU driver does not set Python, NumPy, or PyTorch seeds.
- The standalone Crosswords driver sets all three seeds when a numeric run ID is supplied.
- The standalone edge optimizer appears to use the inner batch-loop variable `i` instead of the optimization step when deciding when to display/save. This does not change the gradient update, but it can make artifact cadence differ from the apparent intention.
- The MMLU data loader samples one NumPy permutation and then repeats that same order indefinitely rather than reshuffling each epoch.

These observations are reproducibility risks, not claims that the reported paper results are invalid.

## 6. Benchmarks and reported experiments

### MMLU

- Purpose: demonstrate that edge optimization can suppress adversarial agents and, with role diversity, obtain a small collaborative gain.
- Graphs: equal numbers of truthful input-output agents and deliberately adversarial agents, plus a majority-vote decision node.
- Main settings: 1T1A, 3T3A, 5T5A, and 7T7A; `T` denotes truthful and `A` adversarial.
- Optimization: 200 iterations, four graph samples per iteration, learning rate 0.1, GPT-4 Turbo, temperature 0.2.
- Evaluation compares direct answer, full graph, random graphs, and optimized graph.
- Reported collaborative result: heterogeneous roles improve over the single-agent baseline by `2.1% +/- 1.1%` over five training seeds.

### Mini Crosswords

- Purpose: test whether edge optimization can recombine Tree-of-Thought, Reflexion, and chain/brute-force agents.
- Data: 20 five-by-five crossword problems.
- Utility: best-state word accuracy.
- Optimization: initial edge probability 0.1, learning rate 0.4, 20 sampled graphs per iteration, ten effective update iterations, GPT-3.5 Turbo at temperature 0.
- Reported result: `0.465 +/- 0.0509` initially versus `0.575 +/- 0.0275` after edge optimization (three runs; standard error). Subsequent node optimization reaches `0.668 +/- 0.0060`.
- A randomly selected optimized distribution evaluated with GPT-4 Turbo reaches `0.800 +/- 0.0616`; the paper notes the expense prevented evaluating multiple learned distributions with that model.

### HumanEval

- Purpose: test node-level prompt/demonstration optimization in a ReAct-style code agent.
- The agent writes code, runs internal tests, and revises failures.
- Reported online result: accuracy rises from 0.76 without optimization to `0.88 +/- 0.007` over three repetitions.

### GAIA

- Purpose: demonstrate framework generality on questions requiring reasoning, files, web search, multimodality, and tools.
- The reported experiments do not apply edge or node optimization.
- Best reported system is a seven-agent Tree-of-Thought swarm with prompt-based self-consistency.
- Reported average across GAIA levels: 18.45, versus 9.70 for the cited GPT-4 Turbo baseline; performance is based on five runs.

## 7. Metrics

- MMLU: multiple-choice accuracy; optimization utility is per-question correctness.
- Mini Crosswords: best-state word accuracy, averaged over sampled graphs/problems.
- HumanEval: fraction of generated programs passing unit tests.
- GAIA: benchmark answer accuracy by difficulty level and overall average.
- Resource metrics in the appendix: prompt tokens, completion tokens, estimated USD cost, and wall-clock hours.
- Structural metrics: number of nodes, number of potential edges, learned edge probabilities, and expected realized edge count.

The paper reports means with either standard deviations or standard errors depending on the experiment; this distinction must be preserved in any reproduction.

## 8. Strengths

1. A clean unifying abstraction covers both single-agent reasoning programs and multi-agent systems.
2. Node and edge modularity permits reuse and task-specific composition.
3. Black-box optimization does not require gradients through the LLM.
4. The paper evaluates both topology optimization and prompt-level optimization.
5. Ablations compare optimized graphs with full and random graphs and, for Crosswords, control approximately for edge density.
6. The appendix reports prompts, important hyperparameters, token usage, cost, and time.
7. Code and experiment entry points are publicly available.

## 9. Weaknesses

1. The objective is performance-only; communication, latency, and API cost are reported but not optimized.
2. REINFORCE is sample-inefficient and high variance when every sample invokes paid LLM calls.
3. Sequential DAG construction makes the induced graph distribution order-dependent.
4. Several results use only three or five repetitions; one expensive GPT-4 Crosswords evaluation uses one learned distribution.
5. GAIA demonstrates framework composition rather than graph optimization, so it does not validate the paper's optimizer on tool-rich tasks.
6. The adversarial MMLU setting is controlled but artificial.
7. Model endpoints (`gpt-4-1106-preview`, `gpt-3.5-turbo-1106`) are historical, limiting exact numerical reproduction.
8. Prompts and graph behavior are task-specific despite the common framework.
9. The repository lacks a single immutable experiment manifest that records every seed, model snapshot, prompt, dependency lock, and raw output used for the paper.
10. Some code paths have weak failure semantics: node errors may be logged and omitted rather than making the run fail loudly.

## 10. Reproducibility requirements

At minimum, a faithful reproduction must record:

- the exact repository revision and any local patch;
- Python 3.10 and a resolved dependency lock compatible with PyTorch 2.1-2.2;
- dataset source, version, split, ordering, and example limit;
- model provider and immutable model/version identifier where available;
- all prompts, roles, temperature, maximum tokens, retries, timeouts, and concurrency;
- graph construction, potential-edge ordering, initial probability, thresholding rule, and decision strategy;
- seeds for Python, NumPy, and PyTorch;
- optimization iterations, batch size, learning rate, and any reward baseline;
- raw model responses, graph masks/log probabilities, learned logits, per-example correctness, tokens, latency, errors, and cost;
- repeated runs using predeclared seeds and no cherry-picking.

Exact 2024 numbers should not be promised when the original hosted model snapshots are unavailable. The defensible target is a behavioral reproduction under clearly documented replacement models, followed by paired baseline/proposed comparisons under identical conditions.

## 11. Compute and API requirements

The official package targets Python 3.10 and depends on PyTorch, OpenAI's Python client, NetworkX, NumPy, pandas, TensorBoard, and task/tool-specific packages. OpenAI access is required for the published LLM paths; GAIA may additionally require search APIs and file/web tooling.

The paper's representative resource table includes:

| Experiment | Cost (USD) | Prompt tokens | Completion tokens | Time (h) |
|---|---:|---:|---:|---:|
| MMLU GPTSwarm optimization (3A3T) | 5.32 | 361,812 | 56,770 | 0.90 |
| MMLU GPTSwarm inference (3A3T) | 1.82 | 113,233 | 22,923 | 0.31 |
| HumanEval without optimization | 1.61 | 59,646 | 33,951 | 0.68 |
| HumanEval with optimization | 28.46 | 2,298,140 | 182,594 | 1.49 |
| Crosswords edge optimization (GPT-3.5 Turbo) | 77.42 | 50,394,028 | 13,511,265 | 2.82 |
| Crosswords edge evaluation (GPT-4 Turbo) | 377.54 | 13,137,160 | 8,205,522 | 5.56 |

These are historical provider prices, not current budget estimates. Hardware demand is modest relative to LLM pretraining because the optimizer operates on small logit tensors; API usage and wall-clock concurrency dominate.

## 12. Opportunities for small extensions

The following are promising but remain unselected until Phase 1:

1. **Cost-aware edge optimization:** subtract normalized token/API cost from task utility.
2. **Budget-constrained optimization:** enforce a token, call, or active-edge budget during graph sampling.
3. **Early exit:** stop after sufficient agreement instead of always invoking the whole realized graph.
4. **Confidence-aware routing:** activate additional agents only when an inexpensive first pass is uncertain.
5. **Graph sparsification:** penalize active communication edges while preserving accuracy.
6. **Latency-aware optimization:** add measured critical-path latency to the objective.
7. **Robust optimization:** train graph distributions across noisy or unreliable agent responses.

The strongest Phase 1 candidates are cost-aware edge optimization and a budget-constrained/early-exit variant. They directly address an observed omission, reuse the existing edge parameterization, add measurable trade-offs, and can be tested on the minimal MMLU baseline without reproducing the expensive Crosswords or GAIA suites.

## Primary sources

- Zhuge et al., *GPTSwarm: Language Agents as Optimizable Graphs*, ICML 2024: <https://proceedings.mlr.press/v235/zhuge24a.html>
- Official implementation: <https://github.com/metauto-ai/GPTSwarm>

