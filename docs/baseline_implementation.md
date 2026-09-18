# Phase 3 Baseline Implementation

## Scope

The implemented baseline is the minimum adversarial MMLU path selected in
Phase 0. It contains direct answer, fully connected, random, and learned
accuracy-only graphs. It does not contain the proposed budget constraint.

## Upstream fidelity

The reference is GPTSwarm commit
1e1e5899d51bde690672417cf92aa4de7b210422. The local adapter retains:

- one single-node truthful agent and one single-node adversarial agent;
- candidate edges between every distinct agent plus edges into the decision node;
- deterministic publication-era candidate-edge ordering;
- independent Bernoulli edge logits initialized at 0.5;
- sequential sampling with cycle rejection;
- majority vote with seeded tie breaking;
- raw binary MMLU accuracy as the only optimization utility;
- REINFORCE loss minus log-probability times utility;
- Adam with learning rate 0.1; and
- thresholded learned probabilities for final evaluation.

The upstream-to-local file mapping and MIT license are under
vendor/gptswarm_reference.

## Provider boundary

The provider interface returns raw text, model identifier, prompt tokens,
completion tokens, latency, and cache status. Two implementations exist:

- deterministic mock provider for tests and the software smoke tier;
- Ollama chat provider for local inference.

The Phase 4 configuration disables response caching because latency is an
outcome. Provider errors and unparseable answers are recorded rather than
silently retried into a favorable result.

## Dataset

- Dataset: cais/mmlu
- Configuration: all
- Revision: c30699e8356da336a370243923dbaf21066bb9fe
- Optimization split: dev
- Evaluation split: validation
- Selection seed: 888
- Size: 24 subject-diverse examples per split

Each committed JSONL file and its example IDs have SHA-256 checksums in
data/mmlu/manifest.json. The two splits are disjoint.

## Environment

- Python 3.10.9
- Windows 10 build 26200
- CPU: Intel64 Family 6 Model 186 Stepping 2
- Logical CPUs: 16
- RAM: 16,890,322,944 bytes
- PyTorch 2.8.0+cpu
- Ollama 0.17.1
- Model: qwen2.5:0.5b, Ollama ID a8b0c5157701, 397 MB

Exact Python dependencies are pinned in requirements.txt and pyproject.toml.

## Logging

Every run produces:

- a manifest with Git/config/dataset hashes and environment metadata;
- per-iteration optimizer records;
- per-example predictions, targets, active edges, raw agent responses, token
  counts, latency, parse failures, and provider errors;
- an aggregate CSV row; and
- an entry in experiments/experiment_log.csv.

## Phase 3 validation

- Six offline tests passed.
- One hundred stochastic six-agent samples were checked for acyclicity.
- Mock direct/full/random/optimized smoke conditions completed on five examples.
- Ollama direct/full/random/optimized smoke conditions completed on five
  examples.
- Random disconnected graphs were retained and logged.

Smoke-tier scores validate software only and are not research evidence.
