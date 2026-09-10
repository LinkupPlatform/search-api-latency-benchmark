"""End-to-end web-search quality benchmark using SimpleQA Verified."""

import argparse
import http.client
import json
import os
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from search_benchmark.arguments import nonnegative_float, positive_int
from search_benchmark.latency import percentile
from search_benchmark.llm import openai_json
from search_benchmark.providers import API_KEY_ENV, PROVIDERS, build_request
from search_benchmark.quality.answers import answer_question
from search_benchmark.quality.dataset import load_simpleqa_dataset
from search_benchmark.quality.grading import (
    SIMPLEQA_VERIFIED_GRADER_TEMPLATE,
    calculate_metrics,
)
from search_benchmark.quality.query_plan import load_query_plan, query_plan_sha256
from search_benchmark.storage import write_json_atomic

DEFAULT_MODEL = "gpt-5.6-luna"
DATA_DIRECTORY = Path("data")

load_dotenv()

GRADE_SCHEMA = {
    "type": "object",
    "properties": {
        "grade": {
            "type": "string",
            "enum": ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"],
        }
    },
    "required": ["grade"],
    "additionalProperties": False,
}


def execute_search(provider: str, api_key: str, query: str) -> dict[str, Any]:
    spec = build_request(provider, api_key, query)
    connection = http.client.HTTPSConnection(spec.hostname, timeout=30)
    started = time.perf_counter()
    try:
        connection.request(spec.method, spec.path, body=spec.body, headers=spec.headers)
        response = connection.getresponse()
        payload = response.read()
        elapsed_ms = (time.perf_counter() - started) * 1000
    finally:
        connection.close()
    if response.status != 200:
        return {
            "query": query,
            "status": response.status,
            "elapsed_ms": round(elapsed_ms, 1),
            "error": payload.decode(errors="replace")[:1000],
            "results": [],
        }
    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError:
        return {
            "query": query,
            "status": response.status,
            "elapsed_ms": round(elapsed_ms, 1),
            "error": "invalid JSON response",
            "results": [],
        }
    return {
        "query": query,
        "status": response.status,
        "elapsed_ms": round(elapsed_ms, 1),
        "results": extract_search_results(provider, decoded),
    }


def execute_search_safely(provider: str, api_key: str, query: str) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        return execute_search(provider, api_key, query)
    except Exception as error:
        return {
            "query": query,
            "status": None,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
            "error": type(error).__name__,
            "results": [],
        }


def _compact_results(
    items: Any, text_keys: tuple[str, ...], url_keys: tuple[str, ...]
) -> list[dict[str, str]]:
    if not isinstance(items, list):
        return []
    compacted = []
    for item in items:
        if not isinstance(item, dict):
            continue
        text_parts = []
        for key in text_keys:
            value = item.get(key)
            if isinstance(value, list):
                text_parts.extend(str(part) for part in value)
            elif value:
                text_parts.append(str(value))
        url = next((str(item[key]) for key in url_keys if item.get(key)), "")
        if text_parts or url:
            compacted.append({"text": "\n".join(text_parts), "url": url})
    return compacted


def extract_search_results(provider: str, payload: Any) -> list[dict[str, str]]:
    if not isinstance(payload, dict):
        return []
    if provider == "brave-search":
        items = payload.get("web", {}).get("results", [])
        return _compact_results(
            items, ("title", "description", "extra_snippets"), ("url",)
        )
    if provider == "serper":
        return _compact_results(
            payload.get("organic", []), ("title", "snippet"), ("link",)
        )

    items = payload.get("results")
    if not isinstance(items, list):
        items = payload.get("data")
    return _compact_results(
        items,
        (
            "title",
            "name",
            "description",
            "snippet",
            "content",
            "text",
            "highlights",
            "excerpts",
        ),
        ("url", "link"),
    )


def grade_answer(
    question: str,
    gold_answer: str,
    predicted_answer: str,
    *,
    api_key: str,
    model: str,
) -> str:
    prompt = SIMPLEQA_VERIFIED_GRADER_TEMPLATE.format(
        question=question,
        criterion=gold_answer,
        answer=predicted_answer,
    )
    result = openai_json(
        api_key=api_key,
        model=model,
        schema_name="simpleqa_grade",
        schema=GRADE_SCHEMA,
        messages=[{"role": "user", "content": prompt}],
    )
    return str(result["grade"])


def latency_metrics(searches: list[dict[str, Any]]) -> dict[str, float]:
    durations = [float(search["elapsed_ms"]) for search in searches]
    return {
        "latency_p50_ms": round(percentile(durations, 0.50), 1),
        "latency_p75_ms": round(percentile(durations, 0.75), 1),
        "latency_p90_ms": round(percentile(durations, 0.90), 1),
    }


def format_metrics_table(metrics: dict[str, dict[str, float | int]]) -> str:
    headers = (
        "Provider",
        "Accuracy",
        "F-score",
        "Latency p50",
        "Latency p75",
        "Latency p90",
    )
    rows = [
        (
            provider,
            f"{float(values['correct_rate']):.1%}",
            f"{float(values['f_score']):.3f}",
            f"{float(values['latency_p50_ms']):.1f} ms",
            f"{float(values['latency_p75_ms']):.1f} ms",
            f"{float(values['latency_p90_ms']):.1f} ms",
        )
        for provider, values in metrics.items()
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]

    def render(row: tuple[str, ...]) -> str:
        return (
            "| "
            + " | ".join(value.ljust(widths[index]) for index, value in enumerate(row))
            + " |"
        )

    separator = "|-" + "-|-".join("-" * width for width in widths) + "-|"
    return "\n".join([render(headers), separator, *(render(row) for row in rows)])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate search providers on SimpleQA Verified."
    )
    parser.add_argument("--execution", choices=("local", "gcloud"), default="local")
    parser.add_argument("--providers", nargs="+", choices=PROVIDERS, required=True)
    parser.add_argument("--dataset-limit", type=positive_int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--pace-seconds", type=nonnegative_float, default=1.0)
    parser.add_argument(
        "--dataset", type=Path, default=DATA_DIRECTORY / "simpleqa_verified.csv"
    )
    parser.add_argument(
        "--query-plan", type=Path, default=DATA_DIRECTORY / "simpleqa-query-plan.json"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/simpleqa-quality.json")
    )
    parser.add_argument(
        "--scout-file",
        type=Path,
        default=DATA_DIRECTORY / "region-scouts.json",
    )
    return parser.parse_args()


def load_or_create_local_result(
    path: Path, *, metadata: dict[str, Any]
) -> dict[str, Any]:
    if not path.exists():
        return {
            "created_at": datetime.now(UTC).isoformat(),
            **metadata,
            "searches": {},
            "rows": [],
            "metrics": {},
        }
    try:
        result = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"Cannot resume invalid result file {path}") from error
    if not isinstance(result, dict):
        raise ValueError(f"Cannot resume result file {path}: expected an object")
    mismatches = [
        key for key, expected in metadata.items() if result.get(key) != expected
    ]
    if mismatches:
        raise ValueError(f"Cannot resume {path}; mismatched " + ", ".join(mismatches))
    if not isinstance(result.get("rows"), list) or not all(
        isinstance(row, dict) for row in result["rows"]
    ):
        raise ValueError(f"Cannot resume {path}: rows must be a list")
    if not isinstance(result.get("metrics"), dict):
        result["metrics"] = {}
    if not isinstance(result.get("searches"), dict):
        result["searches"] = {}
    return result


def main() -> None:
    args = parse_args()
    if args.execution == "gcloud":
        project = os.environ.get("GCP_PROJECT")
        if not project:
            raise ValueError("Set GCP_PROJECT")
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError(
                "Set local OPENAI_API_KEY to generate one shared query plan "
                "before dispatching GCP jobs"
            )
        provider_keys = {}
        for provider in args.providers:
            api_key = os.environ.get(API_KEY_ENV[provider])
            if not api_key:
                raise ValueError(f"Set {API_KEY_ENV[provider]} for {provider}")
            provider_keys[provider] = api_key.strip()
        samples = load_simpleqa_dataset(
            args.dataset, limit=args.dataset_limit, seed=args.seed
        )
        full_query_plan = load_query_plan(
            args.query_plan, dataset=args.dataset, model=args.model
        )
        selected_indexes = [sample["original_index"] for sample in samples]
        missing_indexes = [
            index for index in selected_indexes if index not in full_query_plan
        ]
        if missing_indexes:
            raise ValueError(
                "Query plan is missing selected dataset indexes: "
                + ", ".join(str(index) for index in missing_indexes)
            )
        query_plan = {index: full_query_plan[index] for index in selected_indexes}
        from search_benchmark.cloud import run_quality_jobs

        metrics = run_quality_jobs(
            providers=args.providers,
            dataset_limit=args.dataset_limit,
            seed=args.seed,
            model=args.model,
            pace_seconds=args.pace_seconds,
            project=project,
            scout_file=args.scout_file,
            query_plan=query_plan,
            api_keys={"openai": openai_api_key.strip(), **provider_keys},
        )
        result = {
            "created_at": datetime.now(UTC).isoformat(),
            "execution": "gcloud",
            "dataset": str(args.dataset),
            "dataset_limit": args.dataset_limit,
            "seed": args.seed,
            "model": args.model,
            "providers": args.providers,
            "metrics": metrics,
        }
        write_json_atomic(args.output, result)
        print(format_metrics_table(metrics), flush=True)
        return

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("Set OPENAI_API_KEY")
    provider_keys = {}
    for provider in args.providers:
        key = os.environ.get(API_KEY_ENV[provider])
        if not key:
            raise ValueError(f"Set {API_KEY_ENV[provider]} for {provider}")
        provider_keys[provider] = key.strip()

    samples = load_simpleqa_dataset(
        args.dataset, limit=args.dataset_limit, seed=args.seed
    )
    encoded_query_plan = os.environ.get("SIMPLEQA_QUERY_PLAN")
    if encoded_query_plan:
        from search_benchmark.cloud import decode_query_plan

        queries_by_index = decode_query_plan(encoded_query_plan)
        missing = [
            sample["original_index"]
            for sample in samples
            if sample["original_index"] not in queries_by_index
        ]
        if missing:
            raise ValueError(f"Shared query plan is missing sample indexes: {missing}")
    else:
        queries_by_index = load_query_plan(
            args.query_plan, dataset=args.dataset, model=args.model
        )
    result = load_or_create_local_result(
        args.output,
        metadata={
            "execution": "local",
            "dataset": str(args.dataset),
            "dataset_limit": args.dataset_limit,
            "seed": args.seed,
            "model": args.model,
            "providers": args.providers,
            "query_plan_sha256": query_plan_sha256(queries_by_index),
        },
    )

    for provider in args.providers:
        saved_searches = result["searches"].get(provider, [])
        if not isinstance(saved_searches, list):
            saved_searches = []
        search_cache: dict[str, dict[str, Any]] = {
            search["query"]: search
            for search in saved_searches
            if isinstance(search, dict) and isinstance(search.get("query"), str)
        }
        samples_by_query: dict[str, list[int]] = defaultdict(list)
        for sample in samples:
            for query in queries_by_index[sample["original_index"]]:
                samples_by_query[query].append(sample["original_index"])

        for position, query in enumerate(samples_by_query, 1):
            if query not in search_cache:
                search_cache[query] = execute_search_safely(
                    provider, provider_keys[provider], query
                )
            print(
                f"{provider}: searched {position}/{len(samples_by_query)}",
                flush=True,
            )
            if position < len(samples_by_query):
                time.sleep(args.pace_seconds)
        result["searches"][provider] = list(search_cache.values())
        write_json_atomic(args.output, result)

        for position, sample in enumerate(samples, 1):
            generated_queries = queries_by_index[sample["original_index"]]
            searches = [search_cache[query] for query in generated_queries]
            row = next(
                (
                    candidate
                    for candidate in result["rows"]
                    if candidate.get("provider") == provider
                    and candidate.get("original_index") == sample["original_index"]
                ),
                None,
            )
            if row is None:
                predicted_answer = answer_question(
                    sample["problem"],
                    searches,
                    api_key=openai_api_key,
                    model=args.model,
                )
                row = {
                    **sample,
                    "provider": provider,
                    "generated_queries": generated_queries,
                    "searches": searches,
                    "predicted_answer": predicted_answer,
                    "grade": None,
                }
                result["rows"].append(row)
                write_json_atomic(args.output, result)
            if row.get("grade") is None:
                row["grade"] = grade_answer(
                    sample["problem"],
                    sample["answer"],
                    str(row["predicted_answer"]),
                    api_key=openai_api_key,
                    model=args.model,
                )
            write_json_atomic(args.output, result)
            print(
                f"{provider}: answered and graded {position}/{len(samples)}",
                flush=True,
            )

        grades = [
            row["grade"]
            for row in result["rows"]
            if row["provider"] == provider and isinstance(row.get("grade"), str)
        ]
        result["metrics"][provider] = calculate_metrics(grades)
        result["metrics"][provider].update(latency_metrics(list(search_cache.values())))
        write_json_atomic(args.output, result)

    print(format_metrics_table(result["metrics"]), flush=True)
    if os.environ.get("BENCHMARK_MACHINE_OUTPUT") == "1":
        for provider in args.providers:
            print(
                "RESULT_JSON="
                + json.dumps(result["metrics"][provider], separators=(",", ":")),
                flush=True,
            )


if __name__ == "__main__":
    main()
