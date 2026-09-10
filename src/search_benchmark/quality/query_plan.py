"""Generate and persist reusable search-query plans for SimpleQA."""

import hashlib
import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm

from search_benchmark.llm import openai_json
from search_benchmark.storage import write_json_atomic

QUERY_SCHEMA = {
    "type": "object",
    "properties": {
        "queries": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 6,
        }
    },
    "required": ["queries"],
    "additionalProperties": False,
}


class QueryPlanMismatchError(ValueError):
    """Raised when a saved plan was generated for different benchmark inputs."""


def dataset_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def query_plan_sha256(query_plan: dict[int, list[str]]) -> str:
    serialized = json.dumps(
        query_plan, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(serialized).hexdigest()


def generate_search_queries(question: str, *, api_key: str, model: str) -> list[str]:
    result = openai_json(
        api_key=api_key,
        model=model,
        schema_name="search_queries",
        schema=QUERY_SCHEMA,
        messages=[
            {
                "role": "user",
                "content": (
                    "Convert a factual question into the smallest useful set of web "
                    "search queries. Write each query in concise search-engine keyword "
                    "style, not as a natural-language question: omit interrogatives such "
                    "as who, what, when, where, why, and how. Each query must seek exactly "
                    "one fact, be short, stand alone, and preserve names, dates, and "
                    "disambiguating details. "
                    "Use multiple queries only when separate facts must be found and "
                    "combined. Cover every aspect needed to answer the question. Do not "
                    "answer the question and do not assume the answer.\n\n"
                    f"Question: {question}"
                ),
            }
        ],
    )
    queries = [query.strip() for query in result["queries"] if query.strip()]
    return list(dict.fromkeys(queries))


def build_query_plan(
    samples: list[dict[str, Any]],
    *,
    api_key: str,
    model: str,
    existing_plan: dict[int, list[str]] | None = None,
    checkpoint: Callable[[dict[int, list[str]]], None] | None = None,
    max_workers: int = 5,
) -> dict[int, list[str]]:
    query_plan = dict(existing_plan or {})
    pending_samples = [
        sample for sample in samples if sample["original_index"] not in query_plan
    ]
    with (
        tqdm(
            total=len(samples),
            initial=len(samples) - len(pending_samples),
            desc="Generating search queries",
            unit="query",
        ) as progress,
        ThreadPoolExecutor(max_workers=max_workers) as executor,
    ):
        futures = {
            executor.submit(
                generate_search_queries,
                sample["problem"],
                api_key=api_key,
                model=model,
            ): sample["original_index"]
            for sample in pending_samples
        }
        for position, future in enumerate(as_completed(futures), 1):
            query_plan[futures[future]] = future.result()
            progress.update(1)
            if position % 50 == 0:
                if checkpoint:
                    checkpoint(query_plan)
    if checkpoint:
        checkpoint(query_plan)
    return query_plan


def write_query_plan(
    path: Path,
    *,
    dataset: Path,
    limit: int,
    seed: int,
    model: str,
    queries_by_index: dict[int, list[str]],
) -> None:
    write_json_atomic(
        path,
        {
            "dataset": str(dataset),
            "dataset_sha256": dataset_sha256(dataset),
            "dataset_limit": limit,
            "seed": seed,
            "model": model,
            "queries_by_index": queries_by_index,
        },
    )


def load_query_plan(
    path: Path,
    *,
    dataset: Path | None = None,
    model: str | None = None,
) -> dict[int, list[str]]:
    try:
        saved = json.loads(path.read_text())
        raw_plan = saved["queries_by_index"]
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as error:
        raise ValueError(
            f"Cannot read query plan from {path}; run simpleqa-generate-queries first"
        ) from error
    mismatches = []
    if dataset is not None:
        expected_hash = dataset_sha256(dataset)
        if saved.get("dataset_sha256") != expected_hash:
            mismatches.append("dataset")
    if model is not None and saved.get("model") != model:
        mismatches.append("model")
    if mismatches:
        raise QueryPlanMismatchError(
            f"Query plan {path} does not match the requested "
            + " and ".join(mismatches)
        )
    if not isinstance(raw_plan, dict):
        raise ValueError(f"Query plan in {path} must contain an object")
    plan = {}
    for index, queries in raw_plan.items():
        if not isinstance(queries, list) or not all(
            isinstance(query, str) and query.strip() for query in queries
        ):
            raise ValueError(
                f"Query plan in {path} has invalid queries for index {index}"
            )
        plan[int(index)] = queries
    return plan
