# Submission files

The five-page manuscript describes the completed **qwen2.5:0.5b / Ollama** study. It follows the supplied IEEE Signal Processing Letters Overleaf template: `\documentclass[journal]{IEEEtran}`, a journal author block, running headings, a single-paragraph abstract, alphabetical keywords and an initial drop capital. Pages 1-4 contain the manuscript; page 5 contains references only. Standard template margins and 10-point type are preserved.

- `main_short.tex`: submission manuscript source.
- `main_short.pdf`: compiled five-page manuscript.
- `references.bib`: bibliography shared with the full report.
- `short_budget_response.pdf`: cropped budget-response figure used by the short paper.
- `../outputs/BC-GPTSwarm-overleaf.zip`: self-contained Overleaf upload package.
- `../outputs/BC-GPTSwarm-research-presentation.pptx`: exactly 10 slides; slides 1-3 explain existing work and the problem, slides 4-10 explain the extension.
- `../outputs/BC-GPTSwarm-10-minute-script.md`: speaker notes matching the 10 slides.

Build from the repository's `paper` directory:

```sh
pdflatex -interaction=nonstopmode -halt-on-error main_short.tex
bibtex main_short
pdflatex -interaction=nonstopmode -halt-on-error main_short.tex
pdflatex -interaction=nonstopmode -halt-on-error main_short.tex
```

For Overleaf, create a new project by uploading the ZIP, set `main.tex` as the main document, and select pdfLaTeX. The archive contains the manuscript, bibliography and all three required figures; it does not depend on paths outside the uploaded project. IEEEtran and its bibliography style are standard TeX distribution packages.

The archived results are a reduced behavioral reproduction, not a numerical reproduction of the original GPTSwarm endpoint. The middle target uses 5.8% fewer mean tokens but has lower mean accuracy and remains 4.0% above budget. No new model runs were performed for these submission artifacts.
