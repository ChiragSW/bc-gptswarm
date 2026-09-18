from gptswarm_ext.baseline import (
    evaluate_condition,
    make_distribution,
    parse_answer,
)
from gptswarm_ext.data import MMLUExample
from gptswarm_ext.providers import MockProvider


def example() -> MMLUExample:
    return MMLUExample(
        example_id="example",
        split="validation",
        source_index=0,
        subject="test",
        question="Which letter is first?",
        choices=("A", "B", "C", "D"),
        answer=0,
    )


def test_answer_parser_matches_original_first_character_behavior():
    assert parse_answer("boreal") == "B"
    assert parse_answer("The answer is C.") == "C"
    assert parse_answer("unknown") == ""


def test_all_conditions_execute_with_mock_provider():
    provider = MockProvider()
    for method in ("direct", "full", "random", "optimized"):
        agent_ids, distribution = make_distribution(1, 1, 0.5)
        traces, _ = evaluate_condition(
            method,
            [example()],
            provider,
            distribution,
            agent_ids,
            seed=7,
            threshold=0.5,
        )
        assert len(traces) == 1
        assert traces[0].total_tokens >= 0
        assert traces[0].target == "A"
