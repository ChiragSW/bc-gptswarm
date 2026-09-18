from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .baseline import (
    PROMPT_SPEC,
    UPSTREAM_COMMIT,
    evaluate_condition,
    make_distribution,
    optimize_accuracy_only,
    summarize_traces,
)
from .config import load_config, project_path, set_global_seed
from .data import load_examples
from .providers import build_provider


RESULT_COLUMNS = [
    "experiment_id",
    "timestamp",
    "method",
    "seed",
    "model",
    "dataset",
    "num_examples",
    "accuracy",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "latency_seconds",
    "mean_active_agents",
    "mean_active_edges",
    "failure_count",
    "edge_probabilities",
    "selected_edges",
    "raw_trace_path",
]

LOG_COLUMNS = [
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
]


def _git_state() -> tuple[str, bool]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _append_log(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def run(config_path: str | Path) -> list[dict[str, Any]]:
    config_file = Path(config_path).resolve()
    config = load_config(config_file)
    model_config = dict(config["model"])
    model_config["cache_path"] = str(project_path(model_config["cache_path"]))
    provider = build_provider(model_config)
    dev_examples = load_examples(
        project_path(config["dataset"]["dev_path"]),
        int(config["dataset"]["optimization_limit"]),
    )
    validation_examples = load_examples(
        project_path(config["dataset"]["validation_path"]),
        int(config["dataset"]["eval_limit"]),
    )
    commit, dirty = _git_state()
    timestamp = datetime.now(timezone.utc).isoformat()
    graph_config = config["graph"]
    output_config = config["output"]
    results: list[dict[str, Any]] = []
    log_rows: list[dict[str, Any]] = []

    for method in config["experiment"]["methods"]:
        for seed_value in config["experiment"]["seeds"]:
            seed = int(seed_value)
            set_global_seed(seed)
            agent_ids, distribution = make_distribution(
                int(graph_config["num_truthful_agents"]),
                int(graph_config["num_adversarial_agents"]),
                float(graph_config["initial_probability"]),
            )
            training_records: list[dict[str, Any]] = []
            if method == "optimized":
                training_records = optimize_accuracy_only(
                    distribution=distribution,
                    examples=dev_examples,
                    provider=provider,
                    agent_ids=agent_ids,
                    seed=seed,
                    iterations=int(config["optimization"]["iterations"]),
                    batch_size=int(config["optimization"]["batch_size"]),
                    learning_rate=float(config["optimization"]["learning_rate"]),
                )

            traces, selected_graph = evaluate_condition(
                method=method,
                examples=validation_examples,
                provider=provider,
                distribution=distribution,
                agent_ids=agent_ids,
                seed=seed,
                threshold=float(graph_config["threshold"]),
            )
            summary = summarize_traces(traces)
            experiment_id = (
                f"{config['experiment']['name']}_{method}_seed{seed}"
            )
            raw_path = project_path(output_config["raw_dir"]) / f"{experiment_id}.jsonl"
            manifest = {
                "record_type": "manifest",
                "experiment_id": experiment_id,
                "timestamp": timestamp,
                "git_commit": commit,
                "dirty_worktree": dirty,
                "upstream_gptswarm_commit": UPSTREAM_COMMIT,
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "config_path": str(config_file),
                "config_sha256": _sha256(config_file),
                "dataset_dev_sha256": _sha256(
                    project_path(config["dataset"]["dev_path"])
                ),
                "dataset_validation_sha256": _sha256(
                    project_path(config["dataset"]["validation_path"])
                ),
                "model": model_config,
                "graph": graph_config,
                "optimization": config["optimization"],
                "prompts": PROMPT_SPEC,
            }
            raw_records = [manifest]
            raw_records.extend(
                {"record_type": "optimization", **record}
                for record in training_records
            )
            raw_records.extend(
                {"record_type": "evaluation", **asdict(trace)}
                for trace in traces
            )
            _write_jsonl(raw_path, raw_records)

            probabilities = distribution.probabilities.detach().tolist()
            selected_edges = selected_graph.edges if selected_graph is not None else ()
            result = {
                "experiment_id": experiment_id,
                "timestamp": timestamp,
                "method": method,
                "seed": seed,
                "model": model_config["model"],
                "dataset": "cais/mmlu@c30699e8356da336a370243923dbaf21066bb9fe",
                "num_examples": int(summary["num_examples"]),
                "accuracy": summary["accuracy"],
                "input_tokens": int(summary["input_tokens"]),
                "output_tokens": int(summary["output_tokens"]),
                "total_tokens": int(summary["total_tokens"]),
                "latency_seconds": summary["latency_seconds"],
                "mean_active_agents": summary["mean_active_agents"],
                "mean_active_edges": summary["mean_active_edges"],
                "failure_count": int(summary["failure_count"]),
                "edge_probabilities": json.dumps(probabilities),
                "selected_edges": json.dumps(selected_edges),
                "raw_trace_path": str(raw_path.relative_to(project_path("."))),
            }
            results.append(result)
            log_rows.append(
                {
                    "experiment_id": experiment_id,
                    "date": timestamp,
                    "git_commit": commit,
                    "method": method,
                    "dataset": result["dataset"],
                    "task": "MMLU adversarial 1T1A",
                    "model": model_config["model"],
                    "graph_config": json.dumps(graph_config, sort_keys=True),
                    "seed": seed,
                    "temperature": model_config["temperature"],
                    "num_agents": 1 if method == "direct" else len(agent_ids),
                    "num_edges": summary["mean_active_edges"],
                    "accuracy": summary["accuracy"],
                    "reward": summary["accuracy"],
                    "input_tokens": int(summary["input_tokens"]),
                    "output_tokens": int(summary["output_tokens"]),
                    "total_tokens": int(summary["total_tokens"]),
                    "cost": 0.0,
                    "latency": summary["latency_seconds"],
                    "notes": (
                        "local behavioral reproduction; "
                        f"failures={int(summary['failure_count'])}; dirty={dirty}"
                    ),
                }
            )
            print(
                f"{experiment_id}: accuracy={summary['accuracy']:.3f}, "
                f"tokens={summary['total_tokens']:.0f}, "
                f"failures={summary['failure_count']:.0f}"
            )

    _write_csv(project_path(output_config["baseline_csv"]), RESULT_COLUMNS, results)
    _append_log(project_path(output_config["experiment_log"]), log_rows)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
