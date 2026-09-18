from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    source = ROOT / "results" / "baseline_results.csv"
    frame = pd.read_csv(source)
    order = ["direct", "full", "random", "optimized"]
    summary = (
        frame.groupby("method", sort=False)
        .agg(
            runs=("seed", "count"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            total_tokens_mean=("total_tokens", "mean"),
            latency_seconds_mean=("latency_seconds", "mean"),
            active_edges_mean=("mean_active_edges", "mean"),
            failures_mean=("failure_count", "mean"),
        )
        .reindex(order)
        .reset_index()
    )
    processed = ROOT / "results" / "processed"
    figures = ROOT / "results" / "figures"
    tables = ROOT / "paper" / "tables"
    processed.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    summary.to_csv(processed / "baseline_summary.csv", index=False)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Local MMLU baseline reproduction (mean $\pm$ standard deviation over seeds).}",
        r"\label{tab:baseline-results}",
        r"\small",
        r"\begin{tabular}{lrrr}",
        r"\hline",
        r"Method & Accuracy & Tokens & Failures \\",
        r"\hline",
    ]
    for row in summary.itertuples(index=False):
        accuracy_sd = 0.0 if pd.isna(row.accuracy_std) else row.accuracy_std
        lines.append(
            f"{row.method.title()} & "
            f"{row.accuracy_mean:.3f} $\\pm$ {accuracy_sd:.3f} & "
            f"{row.total_tokens_mean:.0f} & {row.failures_mean:.1f} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
    (tables / "baseline_results.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    figure, axis = plt.subplots(figsize=(4.2, 3.0))
    for row in summary.itertuples(index=False):
        accuracy_sd = 0.0 if pd.isna(row.accuracy_std) else row.accuracy_std
        axis.errorbar(
            row.total_tokens_mean,
            row.accuracy_mean,
            yerr=accuracy_sd,
            marker="o",
            capsize=3,
            label=row.method.title(),
        )
    axis.set_xlabel("Total tokens over 24 questions")
    axis.set_ylabel("Accuracy")
    axis.set_ylim(bottom=0)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(figures / "baseline_accuracy_tokens.pdf")
    plt.close(figure)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
