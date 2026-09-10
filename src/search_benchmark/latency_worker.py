"""Internal Cloud Run worker for one provider-region latency measurement."""

import argparse
import http.client
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from search_benchmark.arguments import nonnegative_float, nonnegative_int, positive_int
from search_benchmark.latency import execute, load_queries, query_at, summarize
from search_benchmark.providers import API_KEY_ENV, PROVIDERS, build_request

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    provider_default = os.environ.get("PROVIDER")
    parser.add_argument(
        "--provider",
        choices=PROVIDERS,
        default=provider_default,
        required=provider_default is None,
    )
    parser.add_argument(
        "--region", default=os.environ.get("BENCHMARK_REGION", "unknown")
    )
    parser.add_argument(
        "--calls",
        type=positive_int,
        default=positive_int(os.environ.get("TIMED_CALLS", "500")),
    )
    parser.add_argument(
        "--warmups",
        type=nonnegative_int,
        default=nonnegative_int(os.environ.get("WARMUPS", "20")),
    )
    parser.add_argument(
        "--pace-seconds",
        type=nonnegative_float,
        default=nonnegative_float(os.environ.get("PACE_SECONDS", "1")),
    )
    parser.add_argument("--queries", type=Path, default=Path("data/queries.txt"))
    parser.add_argument(
        "--cache-key", help="Append the Region Scout cache key to each query."
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.environ.get("API_KEY") or os.environ.get(API_KEY_ENV[args.provider])
    if not api_key:
        raise ValueError(f"Set API_KEY or {API_KEY_ENV[args.provider]}")
    queries = load_queries(args.queries)
    first_spec = build_request(
        args.provider, api_key.strip(), query_at(queries, 0, args.cache_key)
    )
    connection = http.client.HTTPSConnection(first_spec.hostname, timeout=30)

    for index in range(args.warmups):
        spec = build_request(
            args.provider,
            api_key.strip(),
            query_at(queries, -index - 1, args.cache_key),
        )
        status, _ = execute(connection, spec)
        if status != 200:
            raise RuntimeError(f"Warmup failed with HTTP {status}")
    print(f"Completed {args.warmups} warmups", flush=True)

    rows: list[dict[str, object]] = []
    for index in range(args.calls):
        query = query_at(queries, index, args.cache_key)
        spec = build_request(args.provider, api_key.strip(), query)
        started = time.perf_counter()
        try:
            status, response_bytes = execute(connection, spec)
            elapsed_ms = (time.perf_counter() - started) * 1000
            rows.append(
                {
                    "index": index,
                    "query": query,
                    "ok": status == 200,
                    "status": status,
                    "elapsed_ms": elapsed_ms,
                    "response_bytes": response_bytes,
                }
            )
        except Exception as error:
            elapsed_ms = (time.perf_counter() - started) * 1000
            rows.append(
                {
                    "index": index,
                    "query": query,
                    "ok": False,
                    "status": None,
                    "elapsed_ms": elapsed_ms,
                    "error": type(error).__name__,
                }
            )
            connection.close()
            connection = http.client.HTTPSConnection(spec.hostname, timeout=30)
        if (index + 1) % 25 == 0:
            print(f"Completed {index + 1}/{args.calls}", flush=True)
        time.sleep(args.pace_seconds)

    summary = summarize(
        provider=args.provider,
        region=args.region,
        rows=rows,
        warmups=args.warmups,
        pace_seconds=args.pace_seconds,
        cache_key=args.cache_key,
    )
    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "summary": summary,
        "rows": rows,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
    print("RESULT_JSON=" + json.dumps(summary, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
