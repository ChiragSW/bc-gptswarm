# GPTSwarm source reference

- Upstream: https://github.com/metauto-ai/GPTSwarm
- Publication-era commit: 1e1e5899d51bde690672417cf92aa4de7b210422
- License: MIT; see LICENSE.
- Inspected: 2026-09-18.

The executable code in src/gptswarm_ext is a minimal compatibility layer, not
a claim of independent algorithmic invention. It adapts the following upstream
baseline behavior:

| Local component | Publication-era source |
|---|---|
| deterministic potential-edge order and Bernoulli logits | swarm/graph/swarm.py |
| sequential cycle-rejecting sampling | swarm/optimizer/edge_optimizer/parameterization.py |
| MMLU truthful/adversarial prompts | swarm/environment/prompt/mmlu_prompt_set.py |
| majority-vote decision | swarm/environment/operations/final_decision.py |
| raw-accuracy REINFORCE/Adam loop | experiments/evaluator/evaluator.py |
| direct/full/random/optimized matrix | experiments/run_mmlu.py |

Why an adapter is used instead of modifying upstream:

1. the publication code targets retired hosted endpoints;
2. its full dependency surface is unnecessary for the selected MMLU subset;
3. structured token, latency, failure, seed, and edge logging are required;
4. a local Ollama provider and deterministic mock provider are needed.

No Phase 5 budget-constrained objective is present in the baseline adapter.
