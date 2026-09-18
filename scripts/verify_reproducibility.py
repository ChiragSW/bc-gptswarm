"""Verify the committed BC-GPTSwarm reproduction package without model calls."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SEEDS = {11, 22, 33}
EXPECTED_MATRICES = {
    "results/baseline_results.csv": (
        "method",
        {"direct", "full", "random", "optimized"},
    ),
    "results/main_results.csv": (
        "condition",
        {
            "original_accuracy",
            "feasible_accuracy",
            "fixed_penalty",
            "bc_budget_050",
            "bc_budget_075",
            "bc_budget_100",
        },
    ),
    "results/processed/phase7_additional_runs.csv": (
        "condition",
        {"bc_raw_utility", "bc_dual_005", "bc_dual_050", "bc_2t2a"},
    ),
}
LOG_COLUMNS = {
    "experiment_id",
    "date",
    "git_commit",
    "method",
    "dataset",
    "task",
    "model",
    "graph_config",
    "seed",
    "temperature",
    "num_agents",
    "num_edges",
    "accuracy",
    "reward",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cost",
    "latency",
    "notes",
}
GENERATED_ARTIFACTS = {
    "results/processed/baseline_summary.csv",
    "results/processed/main_summary.csv",
    "results/processed/ablation_summary.csv",
    "results/processed/analysis_summary.json",
    "results/processed/statistical_tests.csv",
    "results/processed/statistical_summary.json",
    "results/figures/baseline_accuracy_tokens.pdf",
    "results/figures/main_accuracy_tokens.pdf",
    "results/figures/accuracy_latency.pdf",
    "results/figures/budget_response.pdf",
    "results/figures/graph_scale.pdf",
    "results/figures/optimization_stability.pdf",
    "paper/tables/baseline_results.tex",
    "paper/tables/main_results.tex",
    "paper/tables/ablation_results.tex",
    "paper/tables/statistical_results.tex",
    "paper/main.pdf",
}


class Report:
    def __init__(self) -> None:
        self.checks: list[dict[str, str]] = []

    def pass_(self, name: str, detail: str) -> None:
        self.checks.append({"status": "PASS", "check": name, "detail": detail})

    def fail(self, name: str, detail: str) -> None:
        self.checks.append({"status": "FAIL", "check": name, "detail": detail})

    @property
    def failures(self) -> list[dict[str, str]]:
        return [check for check in self.checks if check["status"] == "FAIL"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(relative_path: str) -> list[dict[str, str]]:
    with (ROOT / relative_path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _portable_path(value: str) -> Path:
    return ROOT / Path(value.replace("\\", "/"))


def _check_environment(report: Report) -> None:
    actual = f"{sys.version_info.major}.{sys.version_info.minor}"
    if sys.version_info[:2] == (3, 10):
        report.pass_("python", f"Python {actual} matches the pinned environment")
    else:
        report.fail("python", f"Python {actual} is active; Python 3.10 is required")

    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    unpinned = [line for line in requirements if line.strip() and "==" not in line]
    if unpinned:
        report.fail("dependencies", f"unpinned requirement lines: {unpinned}")
    else:
        report.pass_("dependencies", f"{len(requirements)} exact package pins found")


def _check_dataset(report: Report) -> None:
    manifest_path = ROOT / "data" / "mmlu" / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        report.fail("dataset manifest", str(error))
        return

    observed_ids: dict[str, list[str]] = {}
    for filename, specification in manifest["files"].items():
        path = manifest_path.parent / filename
        if not path.is_file():
            report.fail("dataset files", f"missing {path.relative_to(ROOT)}")
            return
        actual_hash = _sha256(path)
        expected_hash = specification["sha256"]
        with path.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        ids = [row["example_id"] for row in rows]
        observed_ids[filename] = ids
        if actual_hash != expected_hash:
            report.fail("dataset checksum", f"{filename}: {actual_hash} != {expected_hash}")
            return
        if ids != specification["example_ids"]:
            report.fail("dataset identity", f"{filename}: example IDs differ from manifest")
            return

    overlap = set(observed_ids["dev_subset.jsonl"]) & set(
        observed_ids["validation_subset.jsonl"]
    )
    if overlap:
        report.fail("dataset split", f"{len(overlap)} example IDs occur in both splits")
        return
    counts = {name: len(ids) for name, ids in observed_ids.items()}
    report.pass_(
        "dataset",
        f"revision {manifest['revision']}; checksums and IDs valid; counts={counts}",
    )


def _check_trace(row: dict[str, str]) -> str | None:
    raw_value = row.get("raw_trace_path", "")
    if not raw_value:
        return "missing raw_trace_path"
    path = _portable_path(raw_value)
    if not path.is_file() or path.stat().st_size == 0:
        return f"missing or empty trace {raw_value}"
    evaluation_count = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                evaluation_count += record.get("record_type") == "evaluation"
    except (OSError, json.JSONDecodeError) as error:
        return f"invalid trace {raw_value}: {error}"
    expected = int(row["num_examples"])
    if evaluation_count != expected:
        return f"{raw_value}: {evaluation_count} evaluation rows, expected {expected}"
    return None


def _check_matrices(report: Report) -> set[str]:
    all_ids: set[str] = set()
    for relative_path, (label_column, expected_labels) in EXPECTED_MATRICES.items():
        try:
            rows = _rows(relative_path)
        except OSError as error:
            report.fail("experiment matrix", f"{relative_path}: {error}")
            continue
        pairs = Counter((row[label_column], int(row["seed"])) for row in rows)
        expected_pairs = {
            (label, seed) for label in expected_labels for seed in EXPECTED_SEEDS
        }
        observed_pairs = set(pairs)
        duplicates = [pair for pair, count in pairs.items() if count != 1]
        ids = [row["experiment_id"] for row in rows]
        errors = []
        if observed_pairs != expected_pairs:
            errors.append(
                f"missing={sorted(expected_pairs - observed_pairs)}, "
                f"extra={sorted(observed_pairs - expected_pairs)}"
            )
        if duplicates:
            errors.append(f"non-unique condition/seed pairs={duplicates}")
        if len(ids) != len(set(ids)):
            errors.append("duplicate experiment IDs")
        trace_errors = [error for row in rows if (error := _check_trace(row))]
        if trace_errors:
            errors.extend(trace_errors[:3])
        if errors:
            report.fail(relative_path, "; ".join(errors))
        else:
            report.pass_(
                relative_path,
                f"{len(rows)} complete runs; labels={len(expected_labels)}; seeds={sorted(EXPECTED_SEEDS)}",
            )
        all_ids.update(ids)
    return all_ids


def _check_ledger(report: Report, expected_ids: set[str]) -> None:
    try:
        rows = _rows("experiments/experiment_log.csv")
    except OSError as error:
        report.fail("experiment ledger", str(error))
        return
    columns = set(rows[0]) if rows else set()
    missing_columns = LOG_COLUMNS - columns
    if missing_columns:
        report.fail("experiment ledger", f"missing columns {sorted(missing_columns)}")
        return
    counts = Counter(row["experiment_id"] for row in rows)
    missing_ids = expected_ids - set(counts)
    if missing_ids:
        report.fail("experiment ledger", f"missing core run IDs {sorted(missing_ids)}")
        return
    duplicate_count = sum(count - 1 for count in counts.values() if count > 1)
    detail = f"{len(rows)} entries; all {len(expected_ids)} core run IDs present"
    if duplicate_count:
        detail += f"; {duplicate_count} rerun entries share an experiment ID"
    report.pass_("experiment ledger", detail)


def _check_artifacts(report: Report) -> None:
    missing = []
    for relative_path in sorted(GENERATED_ARTIFACTS):
        path = ROOT / relative_path
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(relative_path)
    if missing:
        report.fail("generated artifacts", f"missing or empty: {missing}")
    else:
        report.pass_("generated artifacts", f"{len(GENERATED_ARTIFACTS)} required files present")


def _payload(report: Report) -> dict[str, Any]:
    return {
        "status": "PASS" if not report.failures else "FAIL",
        "checks": report.checks,
        "failures": len(report.failures),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify committed data, experiment matrices, traces, ledger, and artifacts."
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()

    report = Report()
    _check_environment(report)
    _check_dataset(report)
    expected_ids = _check_matrices(report)
    _check_ledger(report, expected_ids)
    _check_artifacts(report)

    payload = _payload(report)
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        for check in report.checks:
            print(f"[{check['status']}] {check['check']}: {check['detail']}")
        print(f"\nReproducibility package: {payload['status']}")
    raise SystemExit(1 if report.failures else 0)


if __name__ == "__main__":
    main()
