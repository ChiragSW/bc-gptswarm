from __future__ import annotations

from dataclasses import dataclass

import torch


DECISION_NODE = "decision"
Edge = tuple[str, str]


def make_agent_ids(num_truthful: int, num_adversarial: int) -> list[str]:
    truthful = [f"truthful_{index}" for index in range(num_truthful)]
    adversarial = [f"adversarial_{index}" for index in range(num_adversarial)]
    return truthful + adversarial


def make_potential_edges(agent_ids: list[str]) -> list[Edge]:
    # BASELINE: matches publication-era Swarm.organize order: all directed
    # inter-agent pairs first, followed by every agent-to-decision edge.
    edges = [
        (source, target)
        for source in agent_ids
        for target in agent_ids
        if source != target
    ]
    edges.extend((agent_id, DECISION_NODE) for agent_id in agent_ids)
    return edges


def would_create_cycle(edges: list[Edge], candidate: Edge) -> bool:
    source, target = candidate
    if source == target:
        return True
    adjacency: dict[str, list[str]] = {}
    for edge_source, edge_target in edges:
        adjacency.setdefault(edge_source, []).append(edge_target)
    stack = [target]
    visited: set[str] = set()
    while stack:
        node = stack.pop()
        if node == source:
            return True
        if node in visited:
            continue
        visited.add(node)
        stack.extend(adjacency.get(node, []))
    return False


def is_acyclic(edges: tuple[Edge, ...]) -> bool:
    accepted: list[Edge] = []
    for edge in edges:
        if would_create_cycle(accepted, edge):
            return False
        accepted.append(edge)
    return True


@dataclass(frozen=True)
class GraphSample:
    edges: tuple[Edge, ...]
    mask: tuple[bool, ...]
    log_prob: torch.Tensor

    def predecessors(self, node: str) -> list[str]:
        return [source for source, target in self.edges if target == node]


class EdgeWiseDistribution(torch.nn.Module):
    """Faithful minimal form of GPTSwarm's Bernoulli edge distribution."""

    def __init__(self, potential_edges: list[Edge], initial_probability: float = 0.5):
        super().__init__()
        if not 0.0 < initial_probability < 1.0:
            raise ValueError("initial_probability must be strictly between zero and one")
        self.potential_edges = tuple(potential_edges)
        initial_logit = torch.log(
            torch.tensor(initial_probability / (1.0 - initial_probability))
        )
        self.edge_logits = torch.nn.Parameter(
            torch.full((len(potential_edges),), float(initial_logit))
        )

    @property
    def probabilities(self) -> torch.Tensor:
        return torch.sigmoid(self.edge_logits)

    def sample(self, generator: torch.Generator) -> GraphSample:
        accepted: list[Edge] = []
        mask: list[bool] = []
        log_probs: list[torch.Tensor] = [
            torch.tensor(0.0, dtype=self.edge_logits.dtype, requires_grad=True)
        ]
        for edge, logit in zip(self.potential_edges, self.edge_logits):
            if would_create_cycle(accepted, edge):
                mask.append(False)
                continue
            probability = torch.sigmoid(logit)
            selected = bool(torch.rand((), generator=generator) < probability)
            mask.append(selected)
            log_probs.append(torch.log(probability if selected else 1.0 - probability))
            if selected:
                accepted.append(edge)
        return GraphSample(tuple(accepted), tuple(mask), torch.stack(log_probs).sum())

    def sample_feasible(
        self,
        generator: torch.Generator,
        max_attempts: int = 1_000,
    ) -> GraphSample:
        """Sample the Bernoulli DAG policy conditioned on a decision input.

        Rejection sampling alone would draw the right conditional distribution,
        but its uncorrected score function would omit the gradient of the
        conditioning normalizer. Decision-node edges cannot create cycles, so
        the feasibility probability is available in closed form.
        """
        decision_indices = [
            index
            for index, (_, target) in enumerate(self.potential_edges)
            if target == DECISION_NODE
        ]
        if not decision_indices:
            raise ValueError("No candidate edge reaches the decision node")
        decision_probabilities = self.probabilities[decision_indices]
        feasibility_probability = 1.0 - torch.prod(1.0 - decision_probabilities)
        if not bool(feasibility_probability.detach() > 0.0):
            raise RuntimeError("Feasible graph probability is zero")

        for _ in range(max_attempts):
            sample = self.sample(generator)
            if sample.predecessors(DECISION_NODE):
                return GraphSample(
                    edges=sample.edges,
                    mask=sample.mask,
                    log_prob=sample.log_prob - torch.log(feasibility_probability),
                )
        raise RuntimeError(
            f"Failed to sample a feasible graph in {max_attempts} attempts"
        )

    def full(self) -> GraphSample:
        accepted: list[Edge] = []
        mask: list[bool] = []
        for edge in self.potential_edges:
            selected = not would_create_cycle(accepted, edge)
            mask.append(selected)
            if selected:
                accepted.append(edge)
        return GraphSample(
            tuple(accepted),
            tuple(mask),
            torch.tensor(0.0, requires_grad=True),
        )

    def from_probabilities(self, threshold: float = 0.5) -> GraphSample:
        accepted: list[Edge] = []
        mask: list[bool] = []
        for edge, probability in zip(self.potential_edges, self.probabilities.detach()):
            selected = bool(probability > threshold) and not would_create_cycle(
                accepted, edge
            )
            mask.append(selected)
            if selected:
                accepted.append(edge)
        return GraphSample(
            tuple(accepted),
            tuple(mask),
            torch.tensor(0.0, requires_grad=True),
        )

    def from_probabilities_feasible(self, threshold: float = 0.5) -> GraphSample:
        """Threshold the policy while guaranteeing a valid decision input."""
        sample = self.from_probabilities(threshold)
        if sample.predecessors(DECISION_NODE):
            return sample

        decision_indices = [
            index
            for index, (_, target) in enumerate(self.potential_edges)
            if target == DECISION_NODE
        ]
        if not decision_indices:
            raise ValueError("No candidate edge reaches the decision node")
        best_index = max(
            decision_indices,
            key=lambda index: float(self.probabilities[index].detach()),
        )
        edges = list(sample.edges)
        edge = self.potential_edges[best_index]
        if would_create_cycle(edges, edge):
            raise RuntimeError("Decision-edge feasibility repair created a cycle")
        edges.append(edge)
        mask = list(sample.mask)
        mask[best_index] = True
        return GraphSample(
            tuple(edges),
            tuple(mask),
            torch.tensor(0.0, requires_grad=True),
        )


def useful_agent_order(sample: GraphSample, agent_ids: list[str]) -> list[str]:
    reverse: dict[str, list[str]] = {}
    for source, target in sample.edges:
        reverse.setdefault(target, []).append(source)
    useful: set[str] = set()
    stack = list(reverse.get(DECISION_NODE, []))
    while stack:
        node = stack.pop()
        if node in useful:
            continue
        useful.add(node)
        stack.extend(reverse.get(node, []))

    indegree = {node: 0 for node in useful}
    successors: dict[str, list[str]] = {node: [] for node in useful}
    for source, target in sample.edges:
        if source in useful and target in useful:
            successors[source].append(target)
            indegree[target] += 1
    queue = [node for node in agent_ids if node in useful and indegree[node] == 0]
    order: list[str] = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        for successor in successors[node]:
            indegree[successor] -= 1
            if indegree[successor] == 0:
                queue.append(successor)
    if len(order) != len(useful):
        raise RuntimeError("Realized graph is cyclic")
    return order
