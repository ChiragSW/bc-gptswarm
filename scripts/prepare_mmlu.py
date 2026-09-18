from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import load_dataset


DATASET_ID = "cais/mmlu"
DATASET_REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"
ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_subject_diverse(split: Any, split_name: str, count: int, seed: int):
    first_by_subject: dict[str, tuple[int, dict[str, Any]]] = {}
    for index, row in enumerate(split):
        first_by_subject.setdefault(row["subject"], (index, row))
    subjects = sorted(
        first_by_subject,
        key=lambda subject: hashlib.sha256(
            f"{seed}:{split_name}:{subject}".encode()
        ).hexdigest(),
    )
    if count > len(subjects):
        raise ValueError(
            f"Requested {count} examples but only {len(subjects)} subjects exist"
        )
    selected = []
    for subject in subjects[:count]:
        source_index, row = first_by_subject[subject]
        identity = json.dumps(
            {
                "split": split_name,
                "source_index": source_index,
                "subject": subject,
                "question": row["question"],
                "choices": row["choices"],
                "answer": row["answer"],
            },
            sort_keys=True,
        )
        selected.append(
            {
                "example_id": hashlib.sha256(identity.encode()).hexdigest()[:16],
                "split": split_name,
                "source_index": source_index,
                "subject": subject,
                "question": row["question"],
                "choices": list(row["choices"]),
                "answer": int(row["answer"]),
            }
        )
    return selected


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-count", type=int, default=24)
    parser.add_argument("--validation-count", type=int, default=24)
    parser.add_argument("--seed", type=int, default=888)
    args = parser.parse_args()

    dataset = load_dataset(
        DATASET_ID,
        "all",
        revision=DATASET_REVISION,
    )
    output_dir = ROOT / "data" / "mmlu"
    output_dir.mkdir(parents=True, exist_ok=True)
    dev_rows = select_subject_diverse(dataset["dev"], "dev", args.dev_count, args.seed)
    validation_rows = select_subject_diverse(
        dataset["validation"], "validation", args.validation_count, args.seed
    )
    dev_path = output_dir / "dev_subset.jsonl"
    validation_path = output_dir / "validation_subset.jsonl"
    write_jsonl(dev_path, dev_rows)
    write_jsonl(validation_path, validation_rows)

    manifest = {
        "dataset": DATASET_ID,
        "config": "all",
        "revision": DATASET_REVISION,
        "license": "MIT",
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "selection": {
            "strategy": "first row from subjects ordered by seeded SHA-256",
            "seed": args.seed,
            "dev_count": len(dev_rows),
            "validation_count": len(validation_rows),
            "split_overlap": False,
        },
        "files": {
            dev_path.name: {
                "sha256": sha256_file(dev_path),
                "example_ids": [row["example_id"] for row in dev_rows],
                "subjects": [row["subject"] for row in dev_rows],
            },
            validation_path.name: {
                "sha256": sha256_file(validation_path),
                "example_ids": [row["example_id"] for row in validation_rows],
                "subjects": [row["subject"] for row in validation_rows],
            },
        },
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
