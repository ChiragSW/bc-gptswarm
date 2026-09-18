# Baseline Deviation Log

| Item | Original GPTSwarm setting | Phase 4 setting | Reason and consequence |
|---|---|---|---|
| Model | GPT-4 Turbo-era hosted endpoint | qwen2.5:0.5b through Ollama | Original endpoint is not pinned or credentialed; results are behavioral, not numerical reproduction |
| Graph scale | Primary report includes 3 truthful and 3 adversarial agents | 1 truthful and 1 adversarial agent | Minimum graph that exercises adversarial edge selection on CPU |
| Optimization | 200 iterations, batch size 4 | 12 iterations, batch size 4 | Compute-feasible pilot; weaker convergence is expected |
| Evaluation | 153 validation questions | 24 subject-diverse validation questions | CPU-feasible predeclared subset; low statistical power |
| Optimization data | MMLU dev | 24 subject-diverse dev questions | Preserves split separation at reduced scale |
| Execution | Asynchronous hosted calls | Synchronous local calls | Avoids concurrency-dependent local latency; topology and answers remain comparable within this study |
| Answer parsing | First output character | First output character, then standalone A-D fallback | Prevents explanations from becoming automatic failures; every raw response is retained |
| Failures | Some node exceptions can become missing outputs | Explicit failure field; invalid decisions score incorrect | Avoids hidden favorable handling |
| Cost | Historical API dollars | USD 0 local inference | Token counts remain primary resource diagnostic |
| Random graph validity | Unconstrained stochastic realization | Unconstrained stochastic realization | Disconnected graphs remain possible and count as incorrect |

No deviation changes the baseline reward: it remains task accuracy only.
