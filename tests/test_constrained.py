import math

import torch

from gptswarm_ext.baseline import make_distribution
from gptswarm_ext.constrained import (
    optimize_budget_constrained,
    optimize_graph_policy,
    projected_dual_update,
)
from gptswarm_ext.data import MMLUExample
from gptswarm_ext.graph import DECISION_NODE, is_acyclic
from gptswarm_ext.providers import BaseProvider, Generation


class CostAwareProvider(BaseProvider):
    def generate(self, messages):
        adversarial = "Answer a lie" in messages[-1]["content"]
        return Generation(
            text="B" if adversarial else "A",
            input_tokens=39 if adversarial else 9,
            output_tokens=1,
            latency_seconds=0.0,
            model="cost-aware-test",
        )


def _example() -> MMLUExample:
    return MMLUExample(
        example_id="train",
        split="dev",
        source_index=0,
        subject="test",
        question="Choose A.",
        choices=("correct", "wrong", "wrong", "wrong"),
        answer=0,
    )


def test_feasible_sampler_never_disconnects_decision_node():
    _, distribution = make_distribution(1, 1, 0.2)
    generator = torch.Generator().manual_seed(31)
    for _ in range(200):
        sample = distribution.sample_feasible(generator)
        assert sample.predecessors(DECISION_NODE)
        assert is_acyclic(sample.edges)
        assert torch.isfinite(sample.log_prob)


def test_threshold_repair_uses_most_probable_decision_edge():
    _, distribution = make_distribution(1, 1, 0.5)
    with torch.no_grad():
        distribution.edge_logits.copy_(torch.tensor([-8.0, -8.0, -3.0, -2.0]))
    sample = distribution.from_probabilities_feasible(threshold=0.5)
    assert sample.predecessors(DECISION_NODE) == ["adversarial_0"]
    assert is_acyclic(sample.edges)


def test_projected_dual_update_increases_on_violation_and_stays_nonnegative():
    assert projected_dual_update(0.2, 120.0, 100.0, 0.5) == 0.3
    assert projected_dual_update(0.1, 0.0, 100.0, 1.0) == 0.0


def test_constrained_optimizer_has_finite_gradients_and_valid_trajectories():
    agent_ids, distribution = make_distribution(1, 1, 0.5)
    records, state = optimize_budget_constrained(
        distribution=distribution,
        examples=[_example()],
        provider=CostAwareProvider(),
        agent_ids=agent_ids,
        seed=13,
        iterations=30,
        batch_size=8,
        primal_learning_rate=0.1,
        dual_learning_rate=0.2,
        budget_tokens=20.0,
    )
    probabilities = distribution.probabilities.detach().tolist()
    assert len(records) == 240
    assert state.dual_value > 0.0
    assert all(not record["trace"]["error"] for record in records)
    assert all(math.isfinite(value) for value in probabilities)
    assert probabilities[2] > probabilities[3]


def test_control_objectives_keep_dual_fixed_at_zero():
    for objective in ("accuracy", "fixed_penalty"):
        agent_ids, distribution = make_distribution(1, 1, 0.5)
        records, state = optimize_graph_policy(
            distribution=distribution,
            examples=[_example()],
            provider=CostAwareProvider(),
            agent_ids=agent_ids,
            seed=5,
            iterations=3,
            batch_size=4,
            primal_learning_rate=0.1,
            dual_learning_rate=0.2,
            budget_tokens=20.0,
            initial_dual_value=2.0,
            objective=objective,
            fixed_penalty=0.25,
        )
        assert state.dual_value == 0.0
        assert {record["objective"] for record in records} == {objective}
