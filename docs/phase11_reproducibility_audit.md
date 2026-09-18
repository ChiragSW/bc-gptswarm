# Phase 11 Reproducibility Audit

Audit date: 2026-09-18

## Scope

This audit closes the reproduction-package phase. It validates that another researcher can inspect the committed evidence without model calls, run credential-free software checks, rerun the real-model study with documented dependencies, regenerate every reported analysis artifact, and rebuild the final paper.

## Package additions

- `README.md` now documents the project status, original GPTSwarm provenance, BC-GPTSwarm contribution, architecture, installation, environment variables, frozen dataset, three reproduction levels, all experiment commands, ablations, artifact generation, paper compilation, output structure, and integrity limitations.
- `scripts/verify_reproducibility.py` provides a dependency-free integrity check for the Python version, exact dependency pins, dataset hashes and IDs, experiment matrices, raw trace counts, ledger coverage, and required generated artifacts. It supports human-readable and JSON output.
- Experiment runners now serialize repository-relative raw-trace paths with forward slashes.
- The analysis generator resolves both historical Windows separators and portable POSIX separators, so the committed snapshot can be analyzed on either platform.
- Figure generators omit volatile PDF creation and modification timestamps. All six vector figures were regenerated twice and had identical SHA-256 hashes across consecutive builds.

## Verified committed evidence

The integrity checker reports:

- Python 3.10 and ten exact dependency pins;
- two valid 24-example MMLU subsets with matching SHA-256 hashes and no ID overlap;
- 12 complete baseline runs covering four methods and three seeds;
- 18 complete controlled-comparison runs covering six conditions and three seeds;
- 12 complete additional ablation runs covering four conditions and three seeds;
- 51 experiment-ledger entries containing all 42 core research run IDs;
- an evaluation record for every declared example in every core raw trace; and
- all 17 required processed artifacts, figures, tables, and the final paper.

The checker intentionally verifies the committed evidence before any rerun. Experiment commands overwrite their corresponding result files and append ledger entries, so comparisons with the committed snapshot should be performed in a clean branch or disposable clone.

## Validation results

- `python -m pytest -q`: 14 tests passed.
- `python scripts/verify_reproducibility.py --json`: package status `PASS`, zero failed checks.
- `python scripts/generate_baseline_artifacts.py`: completed.
- `python scripts/generate_study_artifacts.py`: completed using the committed cross-platform trace paths.
- `python scripts/generate_statistical_analysis.py`: completed.
- Two consecutive figure-generation passes produced identical hashes for all six vector PDFs.
- BibTeX and three LaTeX passes completed without errors, undefined citations/references, or overfull boxes.
- The final PDF contains 12 pages and was rendered at 150 DPI. Every page was visually inspected; no clipping, overlap, unreadable glyph, broken figure, table overflow, or bibliography defect was found.
- `git diff --check` completed without whitespace errors.

## Full reproduction order

For the exact real-model workflow, use a clean checkout and run:

```text
python -m pytest -q
python scripts/verify_reproducibility.py
python scripts/run_baseline.py --config configs/ollama_smoke.yaml
python scripts/run_baseline.py --config configs/baseline.yaml
python scripts/generate_baseline_artifacts.py
python scripts/run_comparison.py --config configs/main_experiments.yaml
python scripts/run_comparison.py --config configs/ablations.yaml
python scripts/generate_study_artifacts.py
python scripts/generate_statistical_analysis.py
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Ollama 0.17.1 and `qwen2.5:0.5b` are required only for the three real-model commands. The committed results were not rerun during packaging because Phase 11 changes only documentation, portable path serialization, deterministic figure metadata, and verification; all experiment evidence is already complete and remains unchanged.

## Remaining researcher-specific metadata

The paper remains labeled `Anonymous submission`. Authors and affiliations must be supplied for a non-blind venue. No project-level license has been selected by the repository owner; the bundled upstream GPTSwarm reference retains its recorded MIT license.
