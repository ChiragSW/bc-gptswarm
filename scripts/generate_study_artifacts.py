from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
T95_DF2 = 4.302652729
PDF_METADATA = {"CreationDate": None, "ModDate": None}

LABELS = {
    "original_accuracy": "Original",
    "feasible_accuracy": "Feasible accuracy",
    "fixed_penalty": "Fixed penalty",
    "bc_budget_050": "BC 0.50",
    "bc_budget_075": "BC 0.75",
    "bc_budget_100": "BC 1.00",
    "bc_raw_utility": "Raw utility",
    "bc_dual_005": "Dual 0.05",
    "bc_dual_050": "Dual 0.50",
    "bc_2t2a": "BC 2T2A",
}


def _summary(frame: pd.DataFrame, group: str = "condition") -> pd.DataFrame:
    metrics = [
        "accuracy",
        "mean_tokens",
        "normalized_budget_violation",
        "compliance_rate",
        "latency_seconds",
        "optimization_time_seconds",
        "mean_active_agents",
        "mean_active_edges",
        "failure_count",
        "dual_final",
    ]
    rows: list[dict[str, Any]] = []
    for name, group_frame in frame.groupby(group, sort=False):
        row: dict[str, Any] = {group: name, "runs": len(group_frame)}
        for metric in metrics:
            values = group_frame[metric].astype(float)
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            half = T95_DF2 * std / math.sqrt(len(values)) if len(values) > 1 else 0.0
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_ci95_low"] = mean - half
            row[f"{metric}_ci95_high"] = mean + half
        rows.append(row)
    return pd.DataFrame(rows)


def _load_records(path: Path, record_type: str) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("record_type") == record_type:
                records.append(record)
    return records


def _result_path(value: object) -> Path:
    """Resolve trace paths written on either Windows or POSIX."""
    return ROOT / Path(str(value).replace("\\", "/"))


def _write_main_table(summary: pd.DataFrame, path: Path) -> None:
    order = [
        "original_accuracy",
        "feasible_accuracy",
        "fixed_penalty",
        "bc_budget_050",
        "bc_budget_075",
        "bc_budget_100",
    ]
    indexed = summary.set_index("condition")
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Controlled local-model comparison (mean $\pm$ sample standard deviation over three seeds). The 95\% intervals are exploratory because $n=3$.}",
        r"\label{tab:main-results}",
        r"\small",
        r"\begin{tabular}{lrrrrrr}",
        r"\hline",
        r"Condition & Accuracy & Tokens/query & Norm. viol. & Compliance & Agents & Opt. sec. \\",
        r"\hline",
    ]
    for condition in order:
        row = indexed.loc[condition]
        lines.append(
            f"{LABELS[condition]} & "
            f"{row.accuracy_mean:.3f} $\\pm$ {row.accuracy_std:.3f} & "
            f"{row.mean_tokens_mean:.1f} $\\pm$ {row.mean_tokens_std:.1f} & "
            f"{row.normalized_budget_violation_mean:+.3f} & "
            f"{row.compliance_rate_mean:.2f} & "
            f"{row.mean_active_agents_mean:.2f} & "
            f"{row.optimization_time_seconds_mean:.1f} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table*}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _ablation_groups(combined: pd.DataFrame) -> pd.DataFrame:
    specs = {
        "mechanism": [
            ("original", "original_accuracy"),
            ("feasible accuracy", "feasible_accuracy"),
            ("fixed penalty", "fixed_penalty"),
            ("dynamic dual", "bc_budget_075"),
        ],
        "budget ratio": [
            ("0.50", "bc_budget_050"),
            ("0.75", "bc_budget_075"),
            ("1.00", "bc_budget_100"),
        ],
        "utility": [
            ("centered", "bc_budget_075"),
            ("raw", "bc_raw_utility"),
        ],
        "dual rate": [
            ("0.05", "bc_dual_005"),
            ("0.20", "bc_budget_075"),
            ("0.50", "bc_dual_050"),
        ],
        "graph scale": [
            ("1T1A", "bc_budget_075"),
            ("2T2A", "bc_2t2a"),
        ],
    }
    rows = []
    for dimension, levels in specs.items():
        for level, condition in levels:
            subset = combined[combined["condition"] == condition].copy()
            subset.insert(0, "ablation_dimension", dimension)
            subset.insert(1, "ablation_level", level)
            rows.append(subset)
    return pd.concat(rows, ignore_index=True)


def _ablation_summary(long_frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dimension, level), group in long_frame.groupby(
        ["ablation_dimension", "ablation_level"], sort=False
    ):
        row = {
            "ablation_dimension": dimension,
            "ablation_level": level,
            "condition": group["condition"].iloc[0],
            "runs": len(group),
        }
        for metric in [
            "accuracy",
            "mean_tokens",
            "normalized_budget_violation",
            "compliance_rate",
            "mean_active_agents",
            "mean_active_edges",
            "dual_final",
        ]:
            row[f"{metric}_mean"] = float(group[metric].astype(float).mean())
            row[f"{metric}_std"] = float(group[metric].astype(float).std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def _write_ablation_table(summary: pd.DataFrame, path: Path) -> None:
    selected = summary[summary["ablation_dimension"].isin(["utility", "dual rate", "graph scale"])]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Ablations at the middle budget (mean over three seeds).}",
        r"\label{tab:ablation-results}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{llrrr}",
        r"\hline",
        r"Factor & Level & Acc. & Tokens & Viol. \\",
        r"\hline",
    ]
    for row in selected.itertuples(index=False):
        lines.append(
            f"{row.ablation_dimension.title()} & {row.ablation_level} & "
            f"{row.accuracy_mean:.3f} & {row.mean_tokens_mean:.1f} & "
            f"{row.normalized_budget_violation_mean:+.3f} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _pareto(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate in summary.itertuples(index=False):
        dominated = False
        for other in summary.itertuples(index=False):
            no_worse = (
                other.mean_tokens_mean <= candidate.mean_tokens_mean
                and other.accuracy_mean >= candidate.accuracy_mean
            )
            strictly_better = (
                other.mean_tokens_mean < candidate.mean_tokens_mean
                or other.accuracy_mean > candidate.accuracy_mean
            )
            if no_worse and strictly_better:
                dominated = True
                break
        row = candidate._asdict()
        row["pareto_efficient"] = not dominated
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mean_tokens_mean")


def _paired_outcomes(main: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    rows = []
    counts = {"baseline_wins": 0, "proposed_wins": 0, "both_correct": 0, "both_wrong": 0}
    for seed in sorted(main["seed"].astype(int).unique()):
        baseline_row = main[
            (main["seed"].astype(int) == seed)
            & (main["condition"] == "original_accuracy")
        ].iloc[0]
        proposed_row = main[
            (main["seed"].astype(int) == seed)
            & (main["condition"] == "bc_budget_075")
        ].iloc[0]
        baseline = {
            record["example_id"]: record
            for record in _load_records(
                _result_path(baseline_row.raw_trace_path),
                "evaluation",
            )
        }
        proposed = {
            record["example_id"]: record
            for record in _load_records(
                _result_path(proposed_row.raw_trace_path),
                "evaluation",
            )
        }
        for example_id in sorted(baseline):
            left = baseline[example_id]
            right = proposed[example_id]
            if left["correct"] and not right["correct"]:
                outcome = "baseline_wins"
            elif right["correct"] and not left["correct"]:
                outcome = "proposed_wins"
            elif right["correct"]:
                outcome = "both_correct"
            else:
                outcome = "both_wrong"
            counts[outcome] += 1
            rows.append(
                {
                    "seed": seed,
                    "example_id": example_id,
                    "subject": left["subject"],
                    "target": left["target"],
                    "baseline_prediction": left["prediction"],
                    "proposed_prediction": right["prediction"],
                    "baseline_tokens": left["total_tokens"],
                    "proposed_tokens": right["total_tokens"],
                    "proposed_budget": float(proposed_row.budget_tokens),
                    "proposed_over_budget": (
                        right["total_tokens"] > float(proposed_row.budget_tokens)
                    ),
                    "baseline_edges": json.dumps(left["edges"]),
                    "proposed_edges": json.dumps(right["edges"]),
                    "outcome": outcome,
                }
            )
    return pd.DataFrame(rows), counts


def _optimization_stability(combined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    constrained = combined[combined["objective"] == "constrained"]
    for run in constrained.itertuples(index=False):
        records = _load_records(_result_path(run.raw_trace_path), "optimization")
        seen: set[int] = set()
        for record in records:
            iteration = int(record["iteration"])
            if iteration in seen:
                continue
            seen.add(iteration)
            rows.append(
                {
                    "experiment_id": run.experiment_id,
                    "condition": run.condition,
                    "seed": int(run.seed),
                    "iteration": iteration,
                    "batch_mean_tokens": float(record["batch_mean_tokens"]),
                    "budget_tokens": float(record["budget_tokens"]),
                    "normalized_batch_cost": (
                        float(record["batch_mean_tokens"])
                        / float(record["budget_tokens"])
                    ),
                    "dual_after": float(record["dual_after"]),
                    "loss": float(record["loss"]),
                }
            )
    return pd.DataFrame(rows)


def _make_figures(
    main_summary: pd.DataFrame,
    combined_summary: pd.DataFrame,
    ablation_summary: pd.DataFrame,
    stability: pd.DataFrame,
    figures: Path,
) -> None:
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9})

    order = [
        "original_accuracy",
        "feasible_accuracy",
        "fixed_penalty",
        "bc_budget_050",
        "bc_budget_075",
        "bc_budget_100",
    ]
    indexed = main_summary.set_index("condition")
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for condition in order:
        row = indexed.loc[condition]
        ax.errorbar(
            row.mean_tokens_mean,
            row.accuracy_mean,
            xerr=row.mean_tokens_std,
            yerr=row.accuracy_std,
            marker="o",
            capsize=2,
            linestyle="none",
            label=LABELS[condition],
        )
    ax.set_xlabel("Mean tokens per question")
    ax.set_ylabel("Accuracy")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(figures / "main_accuracy_tokens.pdf", metadata=PDF_METADATA)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for condition in order:
        row = indexed.loc[condition]
        ax.errorbar(
            row.latency_seconds_mean,
            row.accuracy_mean,
            xerr=row.latency_seconds_std,
            yerr=row.accuracy_std,
            marker="o",
            capsize=2,
            linestyle="none",
            label=LABELS[condition],
        )
    ax.set_xlabel("Summed inference latency over 24 questions (s)")
    ax.set_ylabel("Accuracy")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(figures / "accuracy_latency.pdf", metadata=PDF_METADATA)
    plt.close(fig)

    budget = ablation_summary[ablation_summary["ablation_dimension"] == "budget ratio"].copy()
    budget["ratio"] = budget["ablation_level"].astype(float)
    budget = budget.sort_values("ratio")
    targets = np.array([180.425, 227.46875, 303.2916666667])
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    axes[0].plot(budget["ratio"], budget["mean_tokens_mean"], marker="o", label="Observed")
    axes[0].plot(budget["ratio"], targets, marker="s", linestyle="--", label="Target")
    axes[0].set_xlabel("Budget ratio")
    axes[0].set_ylabel("Mean tokens/question")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.25)
    axes[1].errorbar(
        budget["ratio"],
        budget["accuracy_mean"],
        yerr=budget["accuracy_std"],
        marker="o",
        capsize=3,
    )
    axes[1].set_xlabel("Budget ratio")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "budget_response.pdf", metadata=PDF_METADATA)
    plt.close(fig)

    graph = ablation_summary[ablation_summary["ablation_dimension"] == "graph scale"]
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.8))
    axes[0].bar(graph["ablation_level"], graph["accuracy_mean"], yerr=graph["accuracy_std"], capsize=3)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Agent count vs. accuracy")
    axes[1].bar(graph["ablation_level"], graph["mean_tokens_mean"], yerr=graph["mean_tokens_std"], capsize=3)
    axes[1].axhline(227.46875, color="black", linestyle="--", linewidth=0.8, label="Budget")
    axes[1].set_ylabel("Mean tokens/question")
    axes[1].set_title("Agent count vs. cost")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures / "graph_scale.pdf", metadata=PDF_METADATA)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    for condition in ["bc_budget_050", "bc_budget_075", "bc_budget_100"]:
        subset = stability[stability["condition"] == condition]
        grouped = subset.groupby("iteration").agg(
            cost=("normalized_batch_cost", "mean"),
            dual=("dual_after", "mean"),
        )
        axes[0].plot(grouped.index, grouped["cost"], label=LABELS[condition])
        axes[1].plot(grouped.index, grouped["dual"], label=LABELS[condition])
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Batch tokens / budget")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Dual value")
    axes[1].legend(frameon=False, fontsize=7)
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "optimization_stability.pdf", metadata=PDF_METADATA)
    plt.close(fig)


def main() -> None:
    main_runs = pd.read_csv(ROOT / "results" / "main_results.csv")
    additional = pd.read_csv(
        ROOT / "results" / "processed" / "phase7_additional_runs.csv"
    )
    combined = pd.concat([main_runs, additional], ignore_index=True)

    processed = ROOT / "results" / "processed"
    figures = ROOT / "results" / "figures"
    tables = ROOT / "paper" / "tables"
    processed.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    main_summary = _summary(main_runs)
    combined_summary = _summary(combined)
    main_summary.to_csv(processed / "main_summary.csv", index=False)
    combined_summary.to_csv(processed / "all_condition_summary.csv", index=False)

    ablation_runs = _ablation_groups(combined)
    ablation_runs.to_csv(ROOT / "results" / "ablation_results.csv", index=False)
    ablation_summary = _ablation_summary(ablation_runs)
    ablation_summary.to_csv(processed / "ablation_summary.csv", index=False)

    pareto = _pareto(combined_summary)
    pareto.to_csv(processed / "pareto_front.csv", index=False)
    paired, paired_counts = _paired_outcomes(main_runs)
    paired.to_csv(processed / "paired_outcomes.csv", index=False)
    paired[paired["outcome"].isin(["baseline_wins", "proposed_wins"])].to_csv(
        processed / "disagreement_cases.csv", index=False
    )
    stability = _optimization_stability(combined)
    stability.to_csv(processed / "optimization_stability.csv", index=False)

    correlations = {
        "active_edges_accuracy": float(
            combined["mean_active_edges"].corr(combined["accuracy"])
        ),
        "active_agents_accuracy": float(
            combined["mean_active_agents"].corr(combined["accuracy"])
        ),
        "latency_accuracy": float(combined["latency_seconds"].corr(combined["accuracy"])),
        "tokens_accuracy": float(combined["mean_tokens"].corr(combined["accuracy"])),
    }
    analysis = {
        "paired_original_vs_bc075": paired_counts,
        "run_level_pearson_correlations": correlations,
        "provider_or_graph_failures": int(combined["failure_count"].sum()),
        "pareto_conditions": pareto.loc[pareto["pareto_efficient"], "condition"].tolist(),
        "notes": [
            "All intervals are exploratory Student-t intervals over three seeds.",
            "Correlations are descriptive run-level associations, not causal estimates.",
            "Latency includes provider inference and remains sensitive to local system load.",
        ],
    }
    (processed / "analysis_summary.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    _write_main_table(main_summary, tables / "main_results.tex")
    _write_ablation_table(ablation_summary, tables / "ablation_results.tex")
    _make_figures(
        main_summary,
        combined_summary,
        ablation_summary,
        stability,
        figures,
    )
    print(main_summary.to_string(index=False))
    print(json.dumps(analysis, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
