"""Generate the Phase 9 statistical analysis from committed Phase 6 results."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gptswarm_ext.statistical_analysis import (
    exact_mcnemar,
    paired_summary,
    question_cluster_bootstrap_ci,
)


BASELINE = "original_accuracy"
PROPOSED = "bc_budget_075"
METRICS = (
    ("accuracy", "proportion"),
    ("mean_tokens", "tokens/question"),
    ("latency_seconds", "seconds/run"),
    ("mean_active_agents", "agents/question"),
    ("mean_active_edges", "edges/question"),
)


def _format_effect(metric: str, value: float) -> str:
    if metric == "accuracy":
        return f"{100.0 * value:+.2f} pp"
    if metric == "mean_tokens":
        return f"{value:+.1f} tokens/q"
    if metric == "latency_seconds":
        return f"{value:+.2f} s/run"
    if metric == "normalized_budget_violation":
        return f"{100.0 * value:+.2f}\%"
    return f"{value:+.3f}"


def _format_interval(metric: str, low: float, high: float) -> str:
    if metric in {"accuracy", "normalized_budget_violation"}:
        return f"[{100.0 * low:+.2f}, {100.0 * high:+.2f}]"
    if metric == "mean_tokens":
        return f"[{low:+.1f}, {high:+.1f}]"
    if metric == "latency_seconds":
        return f"[{low:+.2f}, {high:+.2f}]"
    return f"[{low:+.3f}, {high:+.3f}]"


def _write_latex_table(tests: pd.DataFrame, path: Path) -> None:
    labels = {
        "accuracy_seed": "Accuracy",
        "mean_tokens_seed": "Tokens/question",
        "latency_seconds_seed": "Latency/run",
        "normalized_budget_violation_seed": "Budget violation vs. zero",
        "accuracy_mcnemar": "Accuracy (item-level)",
    }
    selected = tests[tests["analysis_id"].isin(labels)].set_index("analysis_id")
    order = [
        "accuracy_seed",
        "mean_tokens_seed",
        "latency_seconds_seed",
        "normalized_budget_violation_seed",
        "accuracy_mcnemar",
    ]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Focal statistical analysis. Effects are BC-GPTSwarm minus the original baseline; seed-level intervals are Student-$t$ intervals with two degrees of freedom.}",
        r"\label{tab:statistical-results}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lclrr}",
        r"\hline",
        r"Outcome & Unit ($n$) & Effect & 95\% CI & Exact $p$ \\",
        r"\hline",
    ]
    for analysis_id in order:
        row = selected.loc[analysis_id]
        unit = "seed (3)" if row["unit"] == "optimizer seed" else "seed--question (72)"
        lines.append(
            f"{labels[analysis_id]} & {unit} & {_format_effect(row['metric'], row['estimate'])} & "
            f"{_format_interval(row['metric'], row['ci95_low'], row['ci95_high'])} & {row['p_value']:.3f} \\\\"
        )
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\begin{minipage}{0.96\textwidth}\footnotesize",
            r"Seed-level $p$-values enumerate all $2^3$ paired sign flips; their minimum possible two-sided value is 0.25. The item-level row uses exact McNemar testing, while its interval is a fixed-seed question-cluster bootstrap and is secondary because questions repeat across three learned policies.",
            r"\end{minipage}",
            r"\end{table*}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_markdown(tests: pd.DataFrame, mcnemar: dict[str, float], path: Path) -> None:
    indexed = tests.set_index("analysis_id")
    accuracy = indexed.loc["accuracy_seed"]
    tokens = indexed.loc["mean_tokens_seed"]
    latency = indexed.loc["latency_seconds_seed"]
    budget = indexed.loc["normalized_budget_violation_seed"]
    item = indexed.loc["accuracy_mcnemar"]
    text = f"""# Phase 9 Statistical Analysis

## Scope and inferential unit

The focal contrast is the original accuracy-only policy versus middle-budget BC-GPTSwarm (`bc_budget_075`). The optimizer seed is the independent unit for learned-policy comparisons, giving only three paired observations. We report paired raw effects, sample standard deviations, exploratory 95% Student-t intervals with two degrees of freedom, Cohen's dz, and exact two-sided sign-flip p-values. With three pairs, the smallest attainable two-sided exact p-value is 0.25.

The ablation matrix is exploratory and was not preregistered as a confirmatory family, so no ablation-wide null-hypothesis tests are performed. This avoids turning many underpowered comparisons into significance claims.

## Seed-level focal comparison

| Outcome | Mean paired effect (BC - baseline) | SD | 95% CI | Cohen's dz | Exact p |
|---|---:|---:|---:|---:|---:|
| Accuracy | {100 * accuracy.estimate:+.2f} pp | {100 * accuracy.sd:.2f} pp | [{100 * accuracy.ci95_low:+.2f}, {100 * accuracy.ci95_high:+.2f}] pp | {accuracy.effect_size:.3f} | {accuracy.p_value:.3f} |
| Tokens/question | {tokens.estimate:+.2f} | {tokens.sd:.2f} | [{tokens.ci95_low:+.2f}, {tokens.ci95_high:+.2f}] | {tokens.effect_size:.3f} | {tokens.p_value:.3f} |
| Evaluation latency/run | {latency.estimate:+.2f} s | {latency.sd:.2f} s | [{latency.ci95_low:+.2f}, {latency.ci95_high:+.2f}] s | {latency.effect_size:.3f} | {latency.p_value:.3f} |
| Normalized budget violation vs zero | {100 * budget.estimate:+.2f}% | {100 * budget.sd:.2f}% | [{100 * budget.ci95_low:+.2f}, {100 * budget.ci95_high:+.2f}]% | {budget.effect_size:.3f} | {budget.p_value:.3f} |

The token effect is {tokens.relative_effect_percent:+.2f}% relative to the baseline mean. Every seed-level interval includes zero, and no exact test can reject a zero effect. Cohen's dz values are included only as descriptive standardized effects; with n=3 they are unstable.

## Secondary paired-answer analysis

Across 72 seed-question pairs, BC-GPTSwarm has two unique wins and the baseline has three. Exact McNemar p = {item.p_value:.3f}. The conditional matched odds ratio is {mcnemar['matched_odds_ratio']:.3f} with exact 95% CI [{mcnemar['odds_ratio_ci_low']:.3f}, {mcnemar['odds_ratio_ci_high']:.3f}]. The paired risk difference is {100 * item.estimate:+.2f} percentage points; a 20,000-resample question-cluster bootstrap gives [{100 * item.ci95_low:+.2f}, {100 * item.ci95_high:+.2f}] percentage points.

This item-level calculation is secondary. It conditions on the three fitted policies and resamples the 24 held-out questions; it does not create 72 independent optimizer replications.

## Interpretation

- The study provides no statistical evidence that middle-budget BC-GPTSwarm changes accuracy, token use, or latency relative to the original policy.
- Failure to reject zero budget violation is not evidence of budget equivalence or compliance. No equivalence margin was predeclared, and two of three seed means exceed the target.
- Broad intervals and the exact-test resolution limit make the null result inconclusive rather than evidence that the methods are identical.
- The Phase 8 descriptive conclusions remain appropriate: the mechanism is operational, but accuracy improvement and tight-budget convergence are not established.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    main_results = pd.read_csv(ROOT / "results" / "main_results.csv")
    paired = pd.read_csv(ROOT / "results" / "processed" / "paired_outcomes.csv")
    focal = main_results[main_results["condition"].isin([BASELINE, PROPOSED])].copy()
    if focal.groupby("condition")["seed"].nunique().to_dict() != {BASELINE: 3, PROPOSED: 3}:
        raise ValueError("expected three matched seeds for each focal condition")

    seed_rows: list[dict[str, object]] = []
    test_rows: list[dict[str, object]] = []
    for metric, units in METRICS:
        wide = focal.pivot(index="seed", columns="condition", values=metric).sort_index()
        if wide.isna().any().any() or list(wide.index) != [11, 22, 33]:
            raise ValueError(f"incomplete focal pairing for {metric}")
        differences = wide[PROPOSED] - wide[BASELINE]
        summary = paired_summary(differences)
        baseline_mean = float(wide[BASELINE].mean())
        relative = 100.0 * summary["mean"] / baseline_mean if baseline_mean else np.nan
        for seed, row in wide.iterrows():
            seed_rows.append(
                {
                    "seed": seed,
                    "metric": metric,
                    "baseline": row[BASELINE],
                    "proposed": row[PROPOSED],
                    "difference_proposed_minus_baseline": row[PROPOSED] - row[BASELINE],
                    "units": units,
                }
            )
        test_rows.append(
            {
                "analysis_id": f"{metric}_seed",
                "scope": "focal exploratory",
                "unit": "optimizer seed",
                "metric": metric,
                "n": 3,
                "estimate": summary["mean"],
                "sd": summary["sd"],
                "ci95_low": summary["ci_low"],
                "ci95_high": summary["ci_high"],
                "effect_size_name": "Cohen dz",
                "effect_size": summary["cohen_dz"],
                "relative_effect_percent": relative,
                "test": "exact paired sign-flip",
                "p_value": summary["p_value"],
                "caveat": "n=3 independent seed pairs; minimum two-sided exact p is 0.25",
            }
        )

    budget_values = focal[focal["condition"] == PROPOSED].sort_values("seed")[
        "normalized_budget_violation"
    ]
    budget_summary = paired_summary(budget_values)
    test_rows.append(
        {
            "analysis_id": "normalized_budget_violation_seed",
            "scope": "focal exploratory",
            "unit": "optimizer seed",
            "metric": "normalized_budget_violation",
            "n": 3,
            "estimate": budget_summary["mean"],
            "sd": budget_summary["sd"],
            "ci95_low": budget_summary["ci_low"],
            "ci95_high": budget_summary["ci_high"],
            "effect_size_name": "Cohen dz",
            "effect_size": budget_summary["cohen_dz"],
            "relative_effect_percent": np.nan,
            "test": "exact one-sample sign-flip against zero",
            "p_value": budget_summary["p_value"],
            "caveat": "failure to reject zero is not an equivalence or compliance test",
        }
    )

    paired["baseline_correct"] = paired["baseline_prediction"] == paired["target"]
    paired["proposed_correct"] = paired["proposed_prediction"] == paired["target"]
    paired["accuracy_difference"] = (
        paired["proposed_correct"].astype(int) - paired["baseline_correct"].astype(int)
    )
    counts = paired["outcome"].value_counts()
    mcnemar = exact_mcnemar(
        proposed_wins=int(counts.get("proposed_wins", 0)),
        baseline_wins=int(counts.get("baseline_wins", 0)),
    )
    question_effects = paired.groupby("example_id")["accuracy_difference"].mean()
    if len(question_effects) != 24 or not (paired.groupby("example_id").size() == 3).all():
        raise ValueError("expected 24 questions observed under all three seeds")
    bootstrap_low, bootstrap_high = question_cluster_bootstrap_ci(question_effects)
    test_rows.append(
        {
            "analysis_id": "accuracy_mcnemar",
            "scope": "secondary item-level",
            "unit": "seed-question pair",
            "metric": "accuracy",
            "n": len(paired),
            "estimate": float(paired["accuracy_difference"].mean()),
            "sd": np.nan,
            "ci95_low": bootstrap_low,
            "ci95_high": bootstrap_high,
            "effect_size_name": "matched odds ratio",
            "effect_size": mcnemar["matched_odds_ratio"],
            "relative_effect_percent": np.nan,
            "test": "exact McNemar",
            "p_value": mcnemar["p_value"],
            "caveat": "secondary; 24 questions repeat across three fitted policies; CI clusters by question",
        }
    )

    processed = ROOT / "results" / "processed"
    tests = pd.DataFrame(test_rows)
    seed_differences = pd.DataFrame(seed_rows)
    seed_differences.to_csv(processed / "seed_paired_differences.csv", index=False)
    tests.to_csv(processed / "statistical_tests.csv", index=False)
    summary = {
        "focal_contrast": f"{PROPOSED} minus {BASELINE}",
        "independent_seed_pairs": 3,
        "minimum_two_sided_sign_flip_p": 0.25,
        "bootstrap_iterations": 20_000,
        "bootstrap_seed": 20_260_918,
        "mcnemar": mcnemar,
        "multiplicity": "No confirmatory family was preregistered; ablation tests were not performed.",
    }
    (processed / "statistical_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_latex_table(tests, ROOT / "paper" / "tables" / "statistical_results.tex")
    _write_markdown(tests, mcnemar, ROOT / "docs" / "phase9_statistical_analysis.md")
    print(tests.to_string(index=False))


if __name__ == "__main__":
    main()
