# BC-GPTSwarm: 10-minute presentation script

Slides 1-3: existing work and problem statement. Slides 4-10: proposed extension, completed evaluation and limitations. Source: `paper/main_short.tex`, the recorded results and Zhuge et al., ICML 2024, https://proceedings.mlr.press/v235/zhuge24a.html. These notes match the accompanying 10-slide PowerPoint. Timings are a rehearsal guide, including pauses for the equations and results.

## Slide 1 - GPTSwarm (0:00-0:50)

GPTSwarm represents language agents as computational graphs. Operations become nodes, an individual agent is a directed acyclic graph, and a swarm combines several agents. Edges describe information flow between operations. This common representation supports two kinds of optimization: refining the prompts inside nodes and learning the connections between agents. The original work is by Zhuge and colleagues at ICML 2024. Our project builds on its edge optimizer. First I will explain the existing method and the resource-control problem, then the budget-constrained extension and its recorded results.

## Slide 2 - Original edge optimization (0:50-1:40)

Each candidate edge has a Bernoulli inclusion probability controlled by a learned logit. The sampler considers edges in a fixed order and rejects any addition that would introduce a cycle. It executes the resulting graph, scores the final answer, and uses REINFORCE to estimate a policy gradient. Adam updates the edge logits. Language-model execution is a black box: the optimizer learns from observed task utility rather than differentiating through model calls. The objective maximizes expected utility. Prompt optimization is supported separately in the original framework, but our reduced reproduction studies only the edge-learning path.

## Slide 3 - Problem statement (1:40-2:30)

Graphs can differ in resource consumption. Extra collaboration may improve an answer but also consume more tokens, while removing edges may discard useful information. A disconnected graph can appear cheap without being a valid solution. We therefore ask whether the stochastic edge policy can satisfy a declared expected-token budget while retaining task accuracy. The budget is measured in tokens per question and success must be assessed through budget violation as well as accuracy. This is an expected constraint: it does not guarantee that every individual question stays below the target. Cost-aware multi-agent learning already exists; our contribution is a specific extension within GPTSwarm.

## Slide 4 - Constrained formulation (2:30-3:30)

BC-GPTSwarm retains candidate edges, node routines, prompts, cycle rejection and majority vote. It adds token telemetry, feasible conditioning and a learned non-negative multiplier. Cost is the sum of provider-reported input and output tokens across model calls. We maximize expected utility subject to expected cost being at most B. The Lagrangian subtracts lambda times normalized violation, C divided by B minus one. A fixed penalty uses a coefficient chosen in advance; BC learns its multiplier from the observed violation. Latency is recorded separately and is not the optimized constraint. These choices make resource control explicit and testable.

## Slide 5 - Feasibility and conditional gradients (3:30-4:30)

A feasible graph must connect at least one agent to the decision node. That node is a sink, so its incoming edges cannot create cycles. The probability of feasibility is one minus the product of the decision-edge exclusion probabilities. Rejection sampling enforces this event. Sampling alone is insufficient, however: the score-function gradient must use the conditional graph distribution. We subtract log q from the unconditional graph log probability. Without that term, the samples and gradient refer to different distributions. The feasibility-matched accuracy-only and fixed-penalty controls share these validity rules. A direct truthful-path measurement supplies a floor for the operational budget.

## Slide 6 - Learning loop (4:30-5:40)

We sample a feasible graph, execute its agents in topological order, evaluate utility and meter the tokens. The utility advantage is centered within the batch, and the cost penalty is subtracted to form the edge-learning signal. Adam updates the edge logits using the conditional graph score. The dual update raises lambda when mean batch cost exceeds B and projects lambda onto non-negative values. If cost is below target, lambda can decrease. After the fixed optimization allowance, the policy is frozen and held-out evaluation uses recorded seeded feasible draws. A multiplier moving in the expected direction validates the mechanism; it does not prove that the final policy satisfies the constraint.

## Slide 7 - Controlled protocol (5:40-6:35)

The recorded study uses qwen2.5:0.5b through Ollama, with deterministic model settings and a four-token output cap. There are 24 optimization questions, 24 disjoint held-out questions, and three optimizer seeds. Each run receives twelve iterations with batch size four. The graph has one truthful and one adversarial agent. Main comparators are original accuracy-only optimization, feasibility-matched accuracy-only optimization, a fixed penalty, and BC at three budget ratios. The low target is raised by the measured direct-path floor. All comparators share model, prompts, data and execution rules. This is a reduced behavioral reproduction rather than a numerical comparison with the original hosted-model endpoint.

## Slide 8 - Main results (6:35-7:45)

The original baseline averages 0.250 accuracy and 251.1 tokens per question. Middle-budget BC averages 0.236 accuracy and 236.7 tokens. That is a descriptive 5.8 percent token reduction, accompanied by lower mean accuracy and a 4.0 percent expected-budget violation. The fixed penalty consumes slightly fewer tokens than BC but has lower mean accuracy. The low BC target is missed by 28.8 percent; only the high target has aggregate slack. The table reports sample standard deviations over three seeds. Per-query compliance is a different outcome from the expected constraint. None of these point estimates establishes that BC improves accuracy or reliably enforces a tight budget.

## Slide 9 - Budget response (7:45-8:45)

The requested targets change substantially, from 180.4 to 303.3 tokens per question, while observed mean costs stay near 232 to 239. The final multipliers are larger for tighter targets, so the dual mechanism reacts in the intended direction. Yet the final training-batch cost ratios remain above one for low and middle budgets. These observations separate a functioning update rule from demonstrated convergence. Twelve iterations did not establish tight-budget control. The high-budget result has aggregate slack, but the method did not produce a strong monotonic reduction in observed costs as the requested budget tightened.

## Slide 10 - Ablations, limits and next steps (8:45-10:00)

Ablations explore utility centering, dual learning rates and graph size. Raw utility reaches the original mean accuracy with fewer mean tokens, but variance is large. The larger two-truthful, two-adversarial graph raises accuracy to 0.278 at 419.5 tokens and an 84.4 percent budget violation. It is not a successful constrained solution. The focal accuracy difference is minus 1.39 percentage points with a broad confidence interval including effects in either direction. Three seeds cannot establish superiority or equivalence. The supported contribution is an implementation and constraint diagnosis. Future experiments should predeclare longer optimization, use a stronger model and test tasks with more graph-dependent cooperation. Frozen data, raw traces and analysis scripts make the current evidence inspectable.
