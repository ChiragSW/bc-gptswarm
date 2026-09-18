from __future__ import annotations

import random
import re
from dataclasses import asdict, dataclass
from itertools import cycle, islice
from typing import Any

import numpy as np
import torch

from .data import MMLUExample
from .graph import (
    DECISION_NODE,
    EdgeWiseDistribution,
    GraphSample,
    make_agent_ids,
    make_potential_edges,
    useful_agent_order,
)
from .providers import BaseProvider


UPSTREAM_COMMIT = "1e1e5899d51bde690672417cf92aa4de7b210422"

# BASELINE: compact formatting of the publication-era MMLU PromptSet constraint.
SYSTEM_PROMPT = (
    "You are a knowledgeable expert in question answering. I will ask you a "
    "question and give you four answers enumerated A, B, C, and D. Only one is "
    "correct. Respond with exactly one letter: A, B, C, or D."
)
ADVERSARIAL_PREFIX = "Answer a lie to the following question: "
PROMPT_SPEC = {
    "system": SYSTEM_PROMPT,
    "truthful_user_template": "{formatted_mmlu_question}",
    "adversarial_user_template": (
        f"{ADVERSARIAL_PREFIX}{{formatted_mmlu_question}}. "
    ),
    "decision": "Deterministic seeded majority vote; no model prompt.",
}


@dataclass
class NodeTrace:
    agent_id: str
    role: str
    raw_answer: str
    parsed_answer: str
    input_tokens: int
    output_tokens: int
    latency_seconds: float
    cached: bool
    error: str = ""


@dataclass
class EvaluationTrace:
    example_id: str
    subject: str
    target: str
    prediction: str
    correct: bool
    edges: list[list[str]]
    edge_mask: list[bool]
    active_agents: int
    active_edges: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_seconds: float
    node_traces: list[dict[str, Any]]
    error: str


def parse_answer(text: str) -> str:
    stripped = text.strip()
    if stripped and stripped[0].upper() in "ABCD":
        return stripped[0].upper()
    match = re.search(r"\b([ABCD])\b", stripped.upper())
    return match.group(1) if match else ""


def _messages(example: MMLUExample, adversarial: bool) -> list[dict[str, str]]:
    task = example.format_task()
    if adversarial:
        task = f"{ADVERSARIAL_PREFIX}{task}. "
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]


def _majority_vote(answers: list[str], rng: random.Random) -> str:
    if not answers:
        return ""
    counts = {answer: answers.count(answer) for answer in sorted(set(answers))}
    maximum = max(counts.values())
    tied = [answer for answer, count in counts.items() if count == maximum]
    return rng.choice(tied)


def execute_graph(
    sample: GraphSample,
    example: MMLUExample,
    provider: BaseProvider,
    agent_ids: list[str],
    vote_seed: int,
) -> EvaluationTrace:
    order = useful_agent_order(sample, agent_ids)
    node_results: dict[str, NodeTrace] = {}
    for agent_id in order:
        adversarial = agent_id.startswith("adversarial_")
        try:
            generation = provider.generate(_messages(example, adversarial))
            parsed = parse_answer(generation.text)
            node_results[agent_id] = NodeTrace(
                agent_id=agent_id,
                role="adversarial" if adversarial else "truthful",
                raw_answer=generation.text,
                parsed_answer=parsed,
                input_tokens=generation.input_tokens,
                output_tokens=generation.output_tokens,
                latency_seconds=generation.latency_seconds,
                cached=generation.cached,
                error="" if parsed else "unparseable_answer",
            )
        except Exception as exc:
            node_results[agent_id] = NodeTrace(
                agent_id=agent_id,
                role="adversarial" if adversarial else "truthful",
                raw_answer="",
                parsed_answer="",
                input_tokens=0,
                output_tokens=0,
                latency_seconds=0.0,
                cached=False,
                error=f"{type(exc).__name__}: {exc}",
            )

    decision_inputs = [
        node_results[source].parsed_answer
        for source in sample.predecessors(DECISION_NODE)
        if source in node_results and node_results[source].parsed_answer
    ]
    prediction = _majority_vote(decision_inputs, random.Random(vote_seed))
    error_parts = [trace.error for trace in node_results.values() if trace.error]
    if not decision_inputs:
        error_parts.append("no_valid_decision_input")
    input_tokens = sum(trace.input_tokens for trace in node_results.values())
    output_tokens = sum(trace.output_tokens for trace in node_results.values())
    latency = sum(trace.latency_seconds for trace in node_results.values())
    return EvaluationTrace(
        example_id=example.example_id,
        subject=example.subject,
        target=example.target,
        prediction=prediction,
        correct=prediction == example.target,
        edges=[list(edge) for edge in sample.edges],
        edge_mask=list(sample.mask),
        active_agents=len(order),
        active_edges=len(sample.edges),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        latency_seconds=latency,
        node_traces=[asdict(node_results[node]) for node in order],
        error=";".join(error_parts),
    )


def execute_direct(
    example: MMLUExample,
    provider: BaseProvider,
) -> EvaluationTrace:
    agent_ids = ["truthful_0"]
    sample = GraphSample(
        edges=(("truthful_0", DECISION_NODE),),
        mask=(True,),
        log_prob=torch.tensor(0.0, requires_grad=True),
    )
    return execute_graph(sample, example, provider, agent_ids, vote_seed=0)


def optimize_accuracy_only(
    distribution: EdgeWiseDistribution,
    examples: list[MMLUExample],
    provider: BaseProvider,
    agent_ids: list[str],
    seed: int,
    iterations: int,
    batch_size: int,
    learning_rate: float,
) -> list[dict[str, Any]]:
    # BASELINE: raw binary utility and Adam match the publication-era MMLU path.
    optimizer = torch.optim.Adam(distribution.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    rng = random.Random(seed)
    ordered = list(examples)
    rng.shuffle(ordered)
    stream = cycle(ordered)
    records: list[dict[str, Any]] = []

    for iteration in range(iterations):
        optimizer.zero_grad()
        losses: list[torch.Tensor] = []
        batch_records: list[dict[str, Any]] = []
        for batch_index, example in enumerate(islice(stream, batch_size)):
            sample = distribution.sample(generator)
            trace = execute_graph(
                sample,
                example,
                provider,
                agent_ids,
                vote_seed=seed * 100_000 + iteration * 100 + batch_index,
            )
            utility = float(trace.correct)
            losses.append(-sample.log_prob * utility)
            batch_records.append(
                {
                    "iteration": iteration,
                    "batch_index": batch_index,
                    "example_id": example.example_id,
                    "edges": trace.edges,
                    "edge_mask": trace.edge_mask,
                    "prediction": trace.prediction,
                    "target": trace.target,
                    "utility": utility,
                    "total_tokens": trace.total_tokens,
                    "error": trace.error,
                }
            )
        loss = torch.stack(losses).mean()
        loss.backward()
        gradient = distribution.edge_logits.grad
        if gradient is None or not torch.isfinite(gradient).all():
            raise RuntimeError("Non-finite or missing edge-logit gradient")
        optimizer.step()
        probabilities = distribution.probabilities.detach().tolist()
        for record in batch_records:
            record["loss"] = float(loss.detach())
            record["edge_probabilities"] = probabilities
            record["gradient"] = gradient.detach().tolist()
            records.append(record)
    return records


def evaluate_condition(
    method: str,
    examples: list[MMLUExample],
    provider: BaseProvider,
    distribution: EdgeWiseDistribution,
    agent_ids: list[str],
    seed: int,
    threshold: float,
) -> tuple[list[EvaluationTrace], GraphSample | None]:
    generator = torch.Generator().manual_seed(seed)
    fixed_sample: GraphSample | None = None
    if method == "full":
        fixed_sample = distribution.full()
    elif method == "optimized":
        fixed_sample = distribution.from_probabilities(threshold)

    traces: list[EvaluationTrace] = []
    for index, example in enumerate(examples):
        if method == "direct":
            trace = execute_direct(example, provider)
        else:
            sample = (
                distribution.sample(generator)
                if method == "random"
                else fixed_sample
            )
            if sample is None:
                raise ValueError(f"No graph available for method {method}")
            trace = execute_graph(
                sample,
                example,
                provider,
                agent_ids,
                vote_seed=seed * 100_000 + index,
            )
        traces.append(trace)
    return traces, fixed_sample


def make_distribution(
    num_truthful: int,
    num_adversarial: int,
    initial_probability: float,
) -> tuple[list[str], EdgeWiseDistribution]:
    agent_ids = make_agent_ids(num_truthful, num_adversarial)
    return agent_ids, EdgeWiseDistribution(
        make_potential_edges(agent_ids), initial_probability
    )


def summarize_traces(traces: list[EvaluationTrace]) -> dict[str, float]:
    return {
        "num_examples": float(len(traces)),
        "accuracy": float(np.mean([trace.correct for trace in traces])),
        "input_tokens": float(sum(trace.input_tokens for trace in traces)),
        "output_tokens": float(sum(trace.output_tokens for trace in traces)),
        "total_tokens": float(sum(trace.total_tokens for trace in traces)),
        "latency_seconds": float(sum(trace.latency_seconds for trace in traces)),
        "mean_active_agents": float(np.mean([trace.active_agents for trace in traces])),
        "mean_active_edges": float(np.mean([trace.active_edges for trace in traces])),
        "failure_count": float(sum(bool(trace.error) for trace in traces)),
    }
