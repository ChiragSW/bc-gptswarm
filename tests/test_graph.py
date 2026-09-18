import torch

from gptswarm_ext.graph import (
    EdgeWiseDistribution,
    is_acyclic,
    make_agent_ids,
    make_potential_edges,
)


def test_sampled_and_full_graphs_are_acyclic():
    agent_ids = make_agent_ids(3, 3)
    distribution = EdgeWiseDistribution(make_potential_edges(agent_ids), 0.5)
    generator = torch.Generator().manual_seed(123)
    assert is_acyclic(distribution.full().edges)
    for _ in range(100):
        assert is_acyclic(distribution.sample(generator).edges)


def test_potential_edge_order_matches_baseline_construction():
    agent_ids = make_agent_ids(1, 1)
    assert make_potential_edges(agent_ids) == [
        ("truthful_0", "adversarial_0"),
        ("adversarial_0", "truthful_0"),
        ("truthful_0", "decision"),
        ("adversarial_0", "decision"),
    ]
