# BC-GPTSwarm

Reproducible research code for a minimal GPTSwarm MMLU baseline and the planned
budget-constrained extension. Phases 3-4 implement and evaluate only the
accuracy-only baseline. BC-GPTSwarm is not yet implemented.

## Baseline provenance

The compatibility layer follows GPTSwarm commit
1e1e5899d51bde690672417cf92aa4de7b210422. The mapping to upstream files and
license are in vendor/gptswarm_reference.

## Environment

- Python 3.10
- CPU PyTorch
- Ollama 0.17.1
- qwen2.5:0.5b for the credential-free behavioral reproduction

Install the pinned Python packages:

    python -m pip install -r requirements.txt

Install Ollama separately, then obtain the local model:

    ollama pull qwen2.5:0.5b

## Reproduce

Prepare the immutable MMLU snapshots:

    python scripts/prepare_mmlu.py --dev-count 24 --validation-count 24 --seed 888

Run tests and the software smoke tier:

    python -m pytest -q
    python scripts/run_baseline.py --config configs/smoke.yaml

Run the real-model smoke tier and Phase 4 matrix:

    python scripts/run_baseline.py --config configs/ollama_smoke.yaml
    python scripts/run_baseline.py --config configs/baseline.yaml
    python scripts/generate_baseline_artifacts.py

Compile the paper:

    cd paper
    pdflatex main.tex
    bibtex main
    pdflatex main.tex
    pdflatex main.tex

## Layout

- configs: versioned experiment settings
- data/mmlu: committed dataset subsets and checksums
- experiments: append-only experiment ledger
- src/gptswarm_ext: baseline graph, provider, optimizer, and logging code
- results/raw: per-example and per-optimization-step JSONL
- results/processed: generated summaries
- results/figures: generated vector figures
- paper: live LaTeX manuscript
- tests: offline baseline tests
- vendor/gptswarm_reference: upstream provenance and license

## Integrity notes

The local 0.5B model is not numerically comparable to the paper's GPT-4 Turbo
experiments. Results are labeled a small controlled behavioral reproduction.
Failed or disconnected random graphs remain in the logs and count as incorrect.
No result is removed based on outcome.
