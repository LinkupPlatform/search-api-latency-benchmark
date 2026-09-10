"""Repository-local SimpleQA Verified dataset loading."""

import csv
import random
from pathlib import Path
from typing import Any


def load_simpleqa_dataset(path: Path, *, limit: int, seed: int) -> list[dict[str, Any]]:
    """Load a reproducible SimpleQA sample from the checked-in CSV."""
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    if limit > len(rows):
        raise ValueError(f"dataset limit {limit} exceeds dataset size {len(rows)}")
    selected = random.Random(seed).sample(rows, limit)
    return [
        {
            "original_index": int(row["original_index"]),
            "problem": row["problem"],
            "answer": row["answer"],
            "topic": row["topic"],
            "answer_type": row["answer_type"],
            "multi_step": row["multi_step"].lower() == "true",
            "requires_reasoning": row["requires_reasoning"].lower() == "true",
            "urls": [url.strip() for url in row["urls"].split(",") if url.strip()],
        }
        for row in selected
    ]
