import collections
import http.client
import math
import statistics
from pathlib import Path
from typing import cast

from search_benchmark.providers import RequestSpec


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("Cannot calculate a percentile without values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def load_queries(path: Path) -> list[str]:
    queries = [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if not queries:
        raise ValueError(f"No queries found in {path}")
    return queries


def query_at(
    queries: list[str],
    index: int,
    cache_key: str | None = None,
) -> str:
    query = queries[index % len(queries)]
    if cache_key:
        return f"{query} {cache_key}"
    return query


def execute(
    connection: http.client.HTTPSConnection,
    spec: RequestSpec,
) -> tuple[int, int]:
    connection.request(
        spec.method,
        spec.path,
        body=spec.body,
        headers=spec.headers,
    )
    response = connection.getresponse()
    payload = response.read()
    return response.status, len(payload)


def summarize(
    *,
    provider: str,
    region: str,
    rows: list[dict[str, object]],
    warmups: int,
    pace_seconds: float,
    cache_key: str | None,
) -> dict[str, object]:
    successful = [row for row in rows if row["ok"]]
    durations = [cast(float, row["elapsed_ms"]) for row in rows]
    failure_counts = collections.Counter(
        str(row.get("status") or row.get("error", "unknown"))
        for row in rows
        if not row["ok"]
    )
    return {
        "provider": provider,
        "region": region,
        "boundary": "client wall clock through full response body",
        "cache_key": cache_key,
        "warmups": warmups,
        "pace_seconds": pace_seconds,
        "timed_calls": len(rows),
        "successful": len(successful),
        "failed": len(rows) - len(successful),
        "failure_counts": dict(failure_counts),
        "min_ms": round(min(durations), 1),
        "p50_ms": round(percentile(durations, 0.5), 1),
        "p90_ms": round(percentile(durations, 0.9), 1),
        "p95_ms": round(percentile(durations, 0.95), 1),
        "p99_ms": round(percentile(durations, 0.99), 1),
        "max_ms": round(max(durations), 1),
        "mean_ms": round(statistics.fmean(durations), 1),
        "mean_response_bytes": (
            round(
                statistics.fmean(
                    cast(float, row["response_bytes"]) for row in successful
                )
            )
            if successful
            else None
        ),
    }
