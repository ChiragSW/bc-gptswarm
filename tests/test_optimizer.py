from gptswarm_ext.baseline import make_distribution, optimize_accuracy_only
from gptswarm_ext.data import MMLUExample
from gptswarm_ext.providers import BaseProvider, Generation


class AlwaysCorrectProvider(BaseProvider):
    def generate(self, messages):
        return Generation("A", 10, 1, 0.0, "test")


def test_accuracy_optimizer_updates_edge_logits_with_finite_gradient():
    example = MMLUExample(
        example_id="train",
        split="dev",
        source_index=0,
        subject="test",
        question="Choose A.",
        choices=("correct", "wrong", "wrong", "wrong"),
        answer=0,
    )
    agent_ids, distribution = make_distribution(1, 1, 0.2)
    before = distribution.probabilities.detach().clone()
    records = optimize_accuracy_only(
        distribution,
        [example],
        AlwaysCorrectProvider(),
        agent_ids,
        seed=19,
        iterations=40,
        batch_size=4,
        learning_rate=0.1,
    )
    after = distribution.probabilities.detach()
    assert len(records) == 160
    assert not before.equal(after)
    assert after[-2:].max() > before[-2:].max()
