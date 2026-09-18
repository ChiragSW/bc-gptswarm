# Phase 4 Baseline Reproduction Results

## Status and claim boundary

The Phase 4 matrix completed all 12 predeclared runs: four conditions over
three graph seeds. These are real local-model observations, not estimates.
They are a controlled behavioral reproduction using qwen2.5:0.5b, not a
numerical replication of the GPT-4 Turbo results in GPTSwarm.

## Aggregate results

Each run evaluates the same 24 validation questions. Values are mean and
sample standard deviation over seeds 11, 22, and 33.

| Method | Accuracy | Total tokens | Selected edges | Failures |
|---|---:|---:|---:|---:|
| Direct | 0.167 +/- 0.000 | 3,523 | 1.000 | 0.0 |
| Full | 0.264 +/- 0.096 | 7,279 | 3.000 | 0.0 |
| Random | 0.250 +/- 0.072 | 4,465 | 1.792 | 4.7 |
| Optimized | 0.250 +/- 0.110 | 6,027 | 2.333 | 0.0 |

Tokens and failures are means across seeds. A failure is any evaluation
question without a valid parsed decision; random disconnected graphs account
for the recorded failures.

## Main comparisons

- Optimized and random graphs have identical mean accuracy: 0.250.
- Optimized is 0.014 below full and 0.083 above direct in mean accuracy.
- Optimized uses 17.2% fewer tokens than full, but 35.0% more than random.
- The lower random token count is partly caused by disconnected graphs:
  random averages 4.7 failed questions out of 24, whereas optimized has none.
- The optimized-minus-random accuracy difference by seed is +0.042, +0.083,
  and -0.125. The sign is not stable.

No statistical significance is claimed from three seeds and 24 questions.

## Learned graphs

Potential-edge order is:

1. truthful to adversarial;
2. adversarial to truthful;
3. truthful to decision; and
4. adversarial to decision.

| Seed | Final probabilities | Thresholded graph | Accuracy |
|---:|---|---|---:|
| 11 | [0.504, 0.423, 0.643, 0.624] | truthful to adversarial; both agents to decision | 0.208 |
| 22 | [0.344, 0.454, 0.640, 0.663] | both agents to decision | 0.375 |
| 33 | [0.502, 0.559, 0.655, 0.405] | truthful to adversarial; truthful to decision | 0.167 |

The truthful-to-decision probability rises above 0.63 for all three seeds.
The adversarial-to-decision edge is suppressed below the 0.5 threshold in only
one seed. The 48 optimization executions per seed produce 12, 13, and 9
positive-utility observations, respectively.

## Consistency with GPTSwarm

The implementation reproduces several qualitative mechanisms:

- edge logits receive finite policy gradients and change from their 0.5 start;
- all sampled and selected graphs remain acyclic;
- useful decision edges can receive higher probability;
- one seed suppresses the adversarial decision edge; and
- thresholded graphs avoid the disconnected failures present in random
  sampling.

It does not reproduce the paper's qualitative optimized-over-random accuracy
gain on average. This is unsurprising but not dismissible: the local model,
1T1A scale, 12 updates, and 24-question evaluation are substantially weaker
than the published setup. The honest Phase 4 conclusion is that the baseline
pipeline behaves sensibly, while accuracy improvement is not established.

## Latency caveat

Mean measured request-latency sums are 35.1, 70.3, 37.1, and 29.1 seconds for
direct, full, random, and optimized conditions. Methods ran in a fixed order,
and the later optimized calls benefited from a warm local model. These values
are retained in raw logs but are not used for comparative latency claims.

## Artifacts

- results/baseline_results.csv: one row per method and seed
- results/processed/baseline_summary.csv: generated aggregate table
- results/raw/phase4_local_mmlu_*.jsonl: manifests, optimization records, and
  per-example traces
- results/figures/baseline_accuracy_tokens.pdf: generated vector figure
- paper/tables/baseline_results.tex: generated LaTeX table
- experiments/experiment_log.csv: complete experiment ledger

All Phase 4 manifests record clean Git commit
6b6e8a53a2a90e89035e821c0d50dd41e0743559.
