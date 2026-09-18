from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


LETTERS = ("A", "B", "C", "D")


@dataclass(frozen=True)
class MMLUExample:
    example_id: str
    split: str
    source_index: int
    subject: str
    question: str
    choices: tuple[str, str, str, str]
    answer: int

    @property
    def target(self) -> str:
        return LETTERS[self.answer]

    def format_task(self) -> str:
        options = "\n".join(
            f"Option {letter}: {choice}"
            for letter, choice in zip(LETTERS, self.choices)
        )
        return f"{self.question}\n{options}"


def load_examples(path: str | Path, limit: int | None = None) -> list[MMLUExample]:
    examples: list[MMLUExample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            examples.append(
                MMLUExample(
                    example_id=row["example_id"],
                    split=row["split"],
                    source_index=int(row["source_index"]),
                    subject=row["subject"],
                    question=row["question"],
                    choices=tuple(row["choices"]),
                    answer=int(row["answer"]),
                )
            )
            if limit is not None and len(examples) >= limit:
                break
    if not examples:
        raise ValueError(f"No examples loaded from {path}")
    return examples
