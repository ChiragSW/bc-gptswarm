import json

from gptswarm_ext.data import load_examples


def test_jsonl_loader_and_formatter(tmp_path):
    path = tmp_path / "sample.jsonl"
    row = {
        "example_id": "abc",
        "split": "dev",
        "source_index": 3,
        "subject": "math",
        "question": "2+2?",
        "choices": ["1", "2", "3", "4"],
        "answer": 3,
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    loaded = load_examples(path)
    assert loaded[0].target == "D"
    assert "Option D: 4" in loaded[0].format_task()
