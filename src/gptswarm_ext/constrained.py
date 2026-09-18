from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from .baseline import EvaluationTrace, execute_graph
from .data import MMLUExample
from .graph import EdgeWiseDistribution, GraphSample
from .providers import BaseProvider


@dataclass(frozen=True)
class PrimalDualState:
    dual_value: float
    budget_tokens: float


def projected_dual_update(
    dual_value: float,
    mean_tokens: float,
    budget_tokens: float,
    learning_rate: float,
) -> float:
    if budget_tokens <= 0.0:
        raise ValueError("budget_tokens must be positive")
    if dual_value < 0.0:
        raise ValueError("dual_value must be non-negative")
    if learning_rate < 0.0:
        raise ValueError("dual learning rate must be non-negative")
    violation = mean_tokens / budget_tokens - 1.0
    return max(0.0, dual_value + learning_rate * violation)


def calibrate_direct_cost(traces: list[EvaluationTrace]) -> float:
    if not traces:
        raise ValueError("At least one direct-path trace is required")
    if any(trace.error for trace in traces):
        raise ValueError("Direct-path calibration contains failed executions")
    return float(np.mean([trace.total_tokens for trace in traces]))


def optimize_budget_constrained(
    distribution: EdgeWiseDistribution,
    examples: list[MMLUExample],
    provider: BaseProvider,
    agent_ids: list[str],
    seed: int,
    iterations: int,
    batch_size: int,
    primal_learning_rate: float,
    dual_learning_rate: float,
    budget_tokens: float,
    initial_dual_value: float = 0.0,
    center_utility: bool = True,
) -> tuple[list[dict[str, Any]], PrimalDualState]:
    """Optimize expected accuracy subject to an expected token budget."""
    if not examples:
        raise ValueError("At least one optimization example is required")
    if iterations <= 0 or batch_size <= 0:
        raise ValueError("iterations and batch_size must be positive")
    if primal_learning_rate <= 0.0:
        raise ValueError("primal learning rate must be positive")
    if budget_tokens <= 0.0:
        raise ValueError("budget_tokens must be positive")
    if initial_dual_value < 0.0:
        raise ValueError("initial_dual_value must be non-negative")

    optimizer = torch.optim.Adam(
        distribution.parameters(), lr=primal_learning_rate
    )
    generator = torch.Generator().manual_seed(seed)
    ordered = list(examples)
    permutation = torch.randperm(len(ordered), generator=generator).tolist()
    ordered = [ordered[index] for index in permutation]
    cursor = 0
    dual_value = float(initial_dual_value)
    records: list[dict[str, Any]] = []

    for iteration in range(iterations):
        optimizer.zero_grad()
        samples: list[GraphSample] = []
        traces: list[EvaluationTrace] = []
        for batch_index in range(batch_size):
            example = ordered[cursor % len(ordered)]
            cursor += 1
            sample = distribution.sample_feasible(generator)
            trace = execute_graph(
                sample,
                example,
                provider,
                agent_ids,
                vote_seed=seed * 100_000 + iteration * 100 + batch_index,
            )
            samples.append(sample)
            traces.append(trace)

        utilities = torch.tensor(
            [float(trace.correct) for trace in traces], dtype=torch.float32
        )
        advantages = utilities - utilities.mean() if center_utility else utilities
        costs = torch.tensor(
            [float(trace.total_tokens) for trace in traces], dtype=torch.float32
        )
        violations = costs / budget_tokens - 1.0
        scores = advantages - dual_value * violations
        loss = torch.stack(
            [
                -sample.log_prob * score.detach()
                for sample, score in zip(samples, scores)
            ]
        ).mean()
        loss.backward()
        gradient = distribution.edge_logits.grad
        if gradient is None or not torch.isfinite(gradient).all():
            raise RuntimeError("Non-finite or missing edge-logit gradient")
        optimizer.step()

        dual_before = dual_value
        mean_tokens = float(costs.mean())
        dual_value = projected_dual_update(
            dual_value,
            mean_tokens,
            budget_tokens,
            dual_learning_rate,
        )
        probabilities = distribution.probabilities.detach().tolist()
        for batch_index, (sample, trace) in enumerate(zip(samples, traces)):
            records.append(
                {
                    "iteration": iteration,
                    "batch_index": batch_index,
                    "trace": asdict(trace),
                    "utility": float(utilities[batch_index]),
                    "utility_advantage": float(advantages[batch_index]),
                    "normalized_budget_violation": float(violations[batch_index]),
                    "policy_score": float(scores[batch_index]),
                    "loss": float(loss.detach()),
                    "gradient": gradient.detach().tolist(),
                    "edge_probabilities": probabilities,
                    "dual_before": dual_before,
                    "dual_after": dual_value,
                    "batch_mean_tokens": mean_tokens,
                    "budget_tokens": budget_tokens,
                }
            )

    return records, PrimalDualState(dual_value, budget_tokens)


def evaluate_feasible_policy(
    distribution: EdgeWiseDistribution,
    examples: list[MMLUExample],
    provider: BaseProvider,
    agent_ids: list[str],
    seed: int,
    threshold: float,
) -> tuple[list[EvaluationTrace], GraphSample]:
    sample = distribution.from_probabilities_feasible(threshold)
    traces = [
        execute_graph(
            sample,
            example,
            provider,
            agent_ids,
            vote_seed=seed * 100_000 + index,
        )
        for index, example in enumerate(examples)
    ]
    return traces, sample
