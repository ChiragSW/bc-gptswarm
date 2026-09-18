from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .baseline import (
    PROMPT_SPEC,
    UPSTREAM_COMMIT,
    execute_direct,
    make_distribution,
    summarize_traces,
)
from .config import load_config, project_path, set_global_seed
from .constrained import (
    calibrate_direct_cost,
    evaluate_feasible_policy,
    optimize_budget_constrained,
)
from .data import load_examples
from .providers import build_provider
from .runner import _append_log, _git_state, _sha256, _write_csv, _write_jsonl


PROPOSED_RESULT_COLUMNS = [
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
    "mean_tokens",
    "budget_tokens",
    "normalized_budget_violation",
    "absolute_budget_error",
    "compliance_rate",
    "latency_seconds",
    "mean_active_agents",
    "mean_active_edges",
    "failure_count",
    "dual_final",
    "edge_probabilities",
    "evaluation_edges",
    "raw_trace_path",
]


def run(config_path: str | Path) -> list[dict[str, Any]]:
    config_file = Path(config_path).resolve()
    config = load_config(config_file)
    if "constraint" not in config:
        raise ValueError("Missing configuration section: constraint")

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
    direct_calibration = [execute_direct(example, provider) for example in dev_examples]
    direct_cost_floor = calibrate_direct_cost(direct_calibration)
    budget_tokens = float(config["constraint"]["budget_tokens"])
    if budget_tokens < direct_cost_floor:
        raise ValueError(
            f"Token budget {budget_tokens:.3f} is below the measured direct-path "
            f"cost floor {direct_cost_floor:.3f}"
        )

    commit, dirty = _git_state()
    timestamp = datetime.now(timezone.utc).isoformat()
    graph_config = config["graph"]
    optimization_config = config["optimization"]
    output_config = config["output"]
    results: list[dict[str, Any]] = []
    log_rows: list[dict[str, Any]] = []

    for seed_value in config["experiment"]["seeds"]:
        seed = int(seed_value)
        set_global_seed(seed)
        agent_ids, distribution = make_distribution(
            int(graph_config["num_truthful_agents"]),
            int(graph_config["num_adversarial_agents"]),
            float(graph_config["initial_probability"]),
        )
        training_records, state = optimize_budget_constrained(
            distribution=distribution,
            examples=dev_examples,
            provider=provider,
            agent_ids=agent_ids,
            seed=seed,
            iterations=int(optimization_config["iterations"]),
            batch_size=int(optimization_config["batch_size"]),
            primal_learning_rate=float(optimization_config["primal_learning_rate"]),
            dual_learning_rate=float(optimization_config["dual_learning_rate"]),
            budget_tokens=budget_tokens,
            initial_dual_value=float(
                optimization_config.get("initial_dual_value", 0.0)
            ),
            center_utility=bool(optimization_config.get("center_utility", True)),
        )
        traces, evaluation_graphs = evaluate_feasible_policy(
            distribution=distribution,
            examples=validation_examples,
            provider=provider,
            agent_ids=agent_ids,
            seed=seed,
        )
        summary = summarize_traces(traces)
        mean_tokens = float(np.mean([trace.total_tokens for trace in traces]))
        normalized_violation = mean_tokens / budget_tokens - 1.0
        compliance_rate = float(
            np.mean([trace.total_tokens <= budget_tokens for trace in traces])
        )
        experiment_id = f"{config['experiment']['name']}_seed{seed}"
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
            "optimization": optimization_config,
            "constraint": config["constraint"],
            "measured_direct_cost_floor": direct_cost_floor,
            "evaluation_rule": (
                "one deterministic-by-seed draw per held-out example from the "
                "frozen feasible policy"
            ),
            "prompts": PROMPT_SPEC,
        }
        raw_records = [manifest]
        raw_records.extend(
            {"record_type": "optimization", **record}
            for record in training_records
        )
        raw_records.extend(
            {"record_type": "evaluation", **asdict(trace)} for trace in traces
        )
        _write_jsonl(raw_path, raw_records)

        probabilities = distribution.probabilities.detach().tolist()
        result = {
            "experiment_id": experiment_id,
            "timestamp": timestamp,
            "method": "budget_constrained",
            "seed": seed,
            "model": model_config["model"],
            "dataset": "cais/mmlu@c30699e8356da336a370243923dbaf21066bb9fe",
            "num_examples": int(summary["num_examples"]),
            "accuracy": summary["accuracy"],
            "input_tokens": int(summary["input_tokens"]),
            "output_tokens": int(summary["output_tokens"]),
            "total_tokens": int(summary["total_tokens"]),
            "mean_tokens": mean_tokens,
            "budget_tokens": budget_tokens,
            "normalized_budget_violation": normalized_violation,
            "absolute_budget_error": abs(mean_tokens - budget_tokens),
            "compliance_rate": compliance_rate,
            "latency_seconds": summary["latency_seconds"],
            "mean_active_agents": summary["mean_active_agents"],
            "mean_active_edges": summary["mean_active_edges"],
            "failure_count": int(summary["failure_count"]),
            "dual_final": state.dual_value,
            "edge_probabilities": json.dumps(probabilities),
            "evaluation_edges": json.dumps(
                [sample.edges for sample in evaluation_graphs]
            ),
            "raw_trace_path": str(raw_path.relative_to(project_path("."))),
        }
        results.append(result)
        log_rows.append(
            {
                "experiment_id": experiment_id,
                "date": timestamp,
                "git_commit": commit,
                "method": "budget_constrained",
                "dataset": result["dataset"],
                "task": "MMLU adversarial 1T1A software smoke",
                "model": model_config["model"],
                "graph_config": json.dumps(graph_config, sort_keys=True),
                "seed": seed,
                "temperature": model_config["temperature"],
                "num_agents": len(agent_ids),
                "num_edges": summary["mean_active_edges"],
                "accuracy": summary["accuracy"],
                "reward": summary["accuracy"],
                "input_tokens": int(summary["input_tokens"]),
                "output_tokens": int(summary["output_tokens"]),
                "total_tokens": int(summary["total_tokens"]),
                "cost": 0.0,
                "latency": summary["latency_seconds"],
                "notes": (
                    f"software smoke only; budget={budget_tokens}; "
                    f"violation={normalized_violation:.6f}; "
                    f"lambda={state.dual_value:.6f}; "
                    f"failures={int(summary['failure_count'])}; dirty={dirty}"
                ),
            }
        )
        print(
            f"{experiment_id}: accuracy={summary['accuracy']:.3f}, "
            f"mean_tokens={mean_tokens:.1f}/{budget_tokens:.1f}, "
            f"lambda={state.dual_value:.3f}, "
            f"failures={summary['failure_count']:.0f}"
        )

    _write_csv(
        project_path(output_config["proposed_csv"]),
        PROPOSED_RESULT_COLUMNS,
        results,
    )
    _append_log(project_path(output_config["experiment_log"]), log_rows)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/proposed_smoke.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
