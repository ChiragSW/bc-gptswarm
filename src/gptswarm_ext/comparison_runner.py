from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .baseline import (
    PROMPT_SPEC,
    UPSTREAM_COMMIT,
    evaluate_condition,
    execute_direct,
    make_distribution,
    optimize_accuracy_only,
    summarize_traces,
)
from .config import load_config, project_path, set_global_seed
from .constrained import (
    calibrate_direct_cost,
    evaluate_feasible_policy,
    optimize_graph_policy,
)
from .data import load_examples
from .providers import build_provider
from .runner import _append_log, _git_state, _sha256, _write_csv, _write_jsonl


RESULT_COLUMNS = [
    "experiment_id",
    "timestamp",
    "study",
    "condition",
    "objective",
    "seed",
    "execution_order",
    "model",
    "dataset",
    "num_examples",
    "num_truthful_agents",
    "num_adversarial_agents",
    "potential_edges",
    "budget_ratio",
    "budget_tokens",
    "direct_cost_floor",
    "fixed_penalty",
    "dual_learning_rate",
    "center_utility",
    "accuracy",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "mean_tokens",
    "normalized_budget_violation",
    "absolute_budget_error",
    "compliance_rate",
    "api_cost",
    "latency_seconds",
    "optimization_time_seconds",
    "mean_active_agents",
    "mean_active_edges",
    "failure_count",
    "dual_final",
    "edge_probabilities",
    "evaluation_edges",
    "raw_trace_path",
]


def _merged_graph(base: dict[str, Any], condition: dict[str, Any]) -> dict[str, Any]:
    graph = dict(base)
    graph.update(condition.get("graph", {}))
    return graph


def _budget_for_condition(
    condition: dict[str, Any],
    reference_tokens: float,
    direct_cost_floor: float,
    floor_margin: float,
) -> float:
    ratio = float(condition.get("budget_ratio", 1.0))
    return max(reference_tokens * ratio, direct_cost_floor * floor_margin)


def run(config_path: str | Path) -> list[dict[str, Any]]:
    config_file = Path(config_path).resolve()
    config = load_config(config_file)
    required = {"conditions", "resource"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Missing configuration sections: {sorted(missing)}")

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
    direct_traces = [execute_direct(example, provider) for example in dev_examples]
    direct_cost_floor = calibrate_direct_cost(direct_traces)
    reference_tokens = float(config["resource"]["reference_tokens_per_example"])
    floor_margin = float(config["resource"].get("direct_floor_margin", 1.05))
    commit, dirty = _git_state()
    study = str(config["experiment"]["study"])
    experiment_name = str(config["experiment"]["name"])
    base_graph = config["graph"]
    optimization = config["optimization"]
    output = config["output"]
    conditions = list(config["conditions"])
    seeds = [int(seed) for seed in config["experiment"]["seeds"]]
    results: list[dict[str, Any]] = []
    log_rows: list[dict[str, Any]] = []

    calibration_path = (
        project_path(output["raw_dir"]) / f"{experiment_name}_calibration.jsonl"
    )
    _write_jsonl(
        calibration_path,
        [
            {
                "record_type": "calibration_manifest",
                "study": study,
                "git_commit": commit,
                "dirty_worktree": dirty,
                "reference_tokens_per_example": reference_tokens,
                "direct_cost_floor": direct_cost_floor,
                "floor_margin": floor_margin,
                "prompts": PROMPT_SPEC,
            },
            *(
                {"record_type": "direct_calibration", **asdict(trace)}
                for trace in direct_traces
            ),
        ],
    )

    for seed_index, seed in enumerate(seeds):
        shift = seed_index % len(conditions)
        ordered_conditions = conditions[shift:] + conditions[:shift]
        for order_index, condition in enumerate(ordered_conditions):
            set_global_seed(seed)
            condition_name = str(condition["name"])
            objective = str(condition["objective"])
            graph_config = _merged_graph(base_graph, condition)
            budget_tokens = _budget_for_condition(
                condition,
                reference_tokens,
                direct_cost_floor,
                floor_margin,
            )
            agent_ids, distribution = make_distribution(
                int(graph_config["num_truthful_agents"]),
                int(graph_config["num_adversarial_agents"]),
                float(graph_config["initial_probability"]),
            )
            started = time.perf_counter()
            if objective == "original_accuracy":
                training_records = optimize_accuracy_only(
                    distribution=distribution,
                    examples=dev_examples,
                    provider=provider,
                    agent_ids=agent_ids,
                    seed=seed,
                    iterations=int(optimization["iterations"]),
                    batch_size=int(optimization["batch_size"]),
                    learning_rate=float(optimization["primal_learning_rate"]),
                )
                state_dual = 0.0
            else:
                training_records, state = optimize_graph_policy(
                    distribution=distribution,
                    examples=dev_examples,
                    provider=provider,
                    agent_ids=agent_ids,
                    seed=seed,
                    iterations=int(optimization["iterations"]),
                    batch_size=int(optimization["batch_size"]),
                    primal_learning_rate=float(optimization["primal_learning_rate"]),
                    dual_learning_rate=float(
                        condition.get(
                            "dual_learning_rate",
                            optimization["dual_learning_rate"],
                        )
                    ),
                    budget_tokens=budget_tokens,
                    initial_dual_value=float(
                        condition.get(
                            "initial_dual_value",
                            optimization["initial_dual_value"],
                        )
                    ),
                    center_utility=bool(
                        condition.get("center_utility", optimization["center_utility"])
                    ),
                    objective=objective,
                    fixed_penalty=float(condition.get("fixed_penalty", 0.0)),
                )
                state_dual = state.dual_value
            optimization_time = time.perf_counter() - started

            if objective == "original_accuracy":
                traces, fixed_graph = evaluate_condition(
                    method="optimized",
                    examples=validation_examples,
                    provider=provider,
                    distribution=distribution,
                    agent_ids=agent_ids,
                    seed=seed,
                    threshold=float(graph_config["threshold"]),
                )
                evaluation_edges: list[Any] = [
                    fixed_graph.edges if fixed_graph is not None else ()
                ] * len(traces)
                evaluation_rule = "thresholded publication-style policy"
            else:
                traces, evaluation_graphs = evaluate_feasible_policy(
                    distribution=distribution,
                    examples=validation_examples,
                    provider=provider,
                    agent_ids=agent_ids,
                    seed=seed,
                )
                evaluation_edges = [sample.edges for sample in evaluation_graphs]
                evaluation_rule = "seeded draws from frozen feasible policy"

            summary = summarize_traces(traces)
            mean_tokens = float(np.mean([trace.total_tokens for trace in traces]))
            normalized_violation = mean_tokens / budget_tokens - 1.0
            compliance_rate = float(
                np.mean([trace.total_tokens <= budget_tokens for trace in traces])
            )
            timestamp = datetime.now(timezone.utc).isoformat()
            experiment_id = f"{experiment_name}_{condition_name}_seed{seed}"
            raw_path = project_path(output["raw_dir"]) / f"{experiment_id}.jsonl"
            manifest = {
                "record_type": "manifest",
                "experiment_id": experiment_id,
                "timestamp": timestamp,
                "study": study,
                "condition": condition,
                "execution_order": order_index,
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
                "optimization": optimization,
                "budget_tokens": budget_tokens,
                "direct_cost_floor": direct_cost_floor,
                "reference_tokens_per_example": reference_tokens,
                "evaluation_rule": evaluation_rule,
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
                "study": study,
                "condition": condition_name,
                "objective": objective,
                "seed": seed,
                "execution_order": order_index,
                "model": model_config["model"],
                "dataset": "cais/mmlu@c30699e8356da336a370243923dbaf21066bb9fe",
                "num_examples": int(summary["num_examples"]),
                "num_truthful_agents": int(graph_config["num_truthful_agents"]),
                "num_adversarial_agents": int(
                    graph_config["num_adversarial_agents"]
                ),
                "potential_edges": len(distribution.potential_edges),
                "budget_ratio": float(condition.get("budget_ratio", 1.0)),
                "budget_tokens": budget_tokens,
                "direct_cost_floor": direct_cost_floor,
                "fixed_penalty": float(condition.get("fixed_penalty", 0.0)),
                "dual_learning_rate": float(
                    condition.get(
                        "dual_learning_rate", optimization["dual_learning_rate"]
                    )
                ),
                "center_utility": bool(
                    condition.get("center_utility", optimization["center_utility"])
                ),
                "accuracy": summary["accuracy"],
                "input_tokens": int(summary["input_tokens"]),
                "output_tokens": int(summary["output_tokens"]),
                "total_tokens": int(summary["total_tokens"]),
                "mean_tokens": mean_tokens,
                "normalized_budget_violation": normalized_violation,
                "absolute_budget_error": abs(mean_tokens - budget_tokens),
                "compliance_rate": compliance_rate,
                "api_cost": 0.0,
                "latency_seconds": summary["latency_seconds"],
                "optimization_time_seconds": optimization_time,
                "mean_active_agents": summary["mean_active_agents"],
                "mean_active_edges": summary["mean_active_edges"],
                "failure_count": int(summary["failure_count"]),
                "dual_final": state_dual,
                "edge_probabilities": json.dumps(probabilities),
                "evaluation_edges": json.dumps(evaluation_edges),
                "raw_trace_path": raw_path.relative_to(project_path(".")).as_posix(),
            }
            results.append(result)
            _write_csv(project_path(output["run_csv"]), RESULT_COLUMNS, results)
            log_rows.append(
                {
                    "experiment_id": experiment_id,
                    "date": timestamp,
                    "git_commit": commit,
                    "method": condition_name,
                    "dataset": result["dataset"],
                    "task": f"MMLU adversarial {len(agent_ids)}-agent {study}",
                    "model": model_config["model"],
                    "graph_config": json.dumps(graph_config, sort_keys=True),
                    "seed": seed,
                    "temperature": model_config["temperature"],
                    "num_agents": summary["mean_active_agents"],
                    "num_edges": summary["mean_active_edges"],
                    "accuracy": summary["accuracy"],
                    "reward": summary["accuracy"],
                    "input_tokens": int(summary["input_tokens"]),
                    "output_tokens": int(summary["output_tokens"]),
                    "total_tokens": int(summary["total_tokens"]),
                    "cost": 0.0,
                    "latency": summary["latency_seconds"],
                    "notes": (
                        f"budget={budget_tokens:.3f}; "
                        f"violation={normalized_violation:.6f}; "
                        f"lambda={state_dual:.6f}; "
                        f"optimization_seconds={optimization_time:.3f}; "
                        f"failures={int(summary['failure_count'])}; dirty={dirty}"
                    ),
                }
            )
            print(
                f"{experiment_id}: accuracy={summary['accuracy']:.3f}, "
                f"tokens={mean_tokens:.1f}/{budget_tokens:.1f}, "
                f"lambda={state_dual:.3f}, failures={summary['failure_count']:.0f}"
            )

    _append_log(project_path(output["experiment_log"]), log_rows)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
