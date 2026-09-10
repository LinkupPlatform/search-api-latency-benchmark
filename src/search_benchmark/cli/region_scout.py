"""Reproducible four-region Cloud Run scouting for every search provider."""

import argparse
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from search_benchmark.arguments import nonnegative_float, nonnegative_int, positive_int
from search_benchmark.cloud import (
    PROVIDER_SECRET,
    prepare_cloud_run,
    read_execution_metrics,
    run_command,
)
from search_benchmark.providers import API_KEY_ENV, PROVIDERS
from search_benchmark.storage import write_json_atomic

DEFAULT_REGIONS = ("us-east4", "us-central1", "us-west2", "europe-west1")

load_dotenv()


def cache_key_for_region(started_at: datetime, region: str) -> str:
    digest = hashlib.sha256(f"{started_at.isoformat()}:{region}".encode()).digest()
    return "".join(chr(ord("a") + byte % 26) for byte in digest[:2])


def scout_provider_region(
    *,
    provider: str,
    region: str,
    project: str,
    image: str,
    calls: int,
    warmups: int,
    pace_seconds: float,
    cache_key: str,
) -> dict[str, Any]:
    job_name = f"region-scout-{provider}".replace("_", "-")
    worker_args = ",".join(
        [
            "run",
            "--frozen",
            "--no-dev",
            "python",
            "-m",
            "search_benchmark.latency_worker",
            "--provider",
            provider,
            "--region",
            f"gcp-{region}",
            "--calls",
            str(calls),
            "--warmups",
            str(warmups),
            "--pace-seconds",
            str(pace_seconds),
            "--queries",
            "/app/data/queries.txt",
            "--cache-key",
            cache_key,
        ]
    )
    run_command(
        [
            "gcloud",
            "run",
            "jobs",
            "deploy",
            job_name,
            f"--image={image}",
            f"--region={region}",
            f"--project={project}",
            "--command=uv",
            f"--args={worker_args}",
            f"--set-secrets=API_KEY={PROVIDER_SECRET[provider]}:latest",
            "--task-timeout=900s",
            "--max-retries=0",
            "--memory=512Mi",
            "--cpu=1",
            "--quiet",
        ],
        stream=True,
    )
    execution_output = run_command(
        [
            "gcloud",
            "run",
            "jobs",
            "execute",
            job_name,
            f"--region={region}",
            f"--project={project}",
            "--wait",
            "--quiet",
        ],
        stream=True,
    )
    return read_execution_metrics(
        project=project,
        job_name=job_name,
        region=region,
        execution_output=execution_output,
        stream=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scout providers from four GCP regions and select each lowest p50."
    )
    parser.add_argument("--providers", nargs="+", choices=PROVIDERS, default=PROVIDERS)
    parser.add_argument("--regions", nargs="+", default=DEFAULT_REGIONS)
    parser.add_argument("--queries", type=positive_int, default=50)
    parser.add_argument("--warmups", type=nonnegative_int, default=5)
    parser.add_argument("--pace-seconds", type=nonnegative_float, default=1.0)
    parser.add_argument("--output", type=Path, default=Path("data/region-scouts.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started_at = datetime.now(UTC)
    project = os.environ.get("GCP_PROJECT")
    if not project:
        raise ValueError("Set GCP_PROJECT")
    provider_keys = {}
    for provider in args.providers:
        api_key = os.environ.get(API_KEY_ENV[provider])
        if not api_key:
            raise ValueError(f"Set {API_KEY_ENV[provider]} for {provider}")
        provider_keys[provider] = api_key.strip()
    image = prepare_cloud_run(project=project, api_keys=provider_keys, stream=True)
    results = []
    cache_keys = {
        region: cache_key_for_region(started_at, region) for region in args.regions
    }
    for region in args.regions:
        print(
            f"Testing {region} for {', '.join(args.providers)} with cache key "
            f"{cache_keys[region]} "
            f"({args.queries} queries each)...",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=len(args.providers)) as executor:
            futures = {
                executor.submit(
                    scout_provider_region,
                    provider=provider,
                    region=region,
                    project=project,
                    image=image,
                    calls=args.queries,
                    warmups=args.warmups,
                    pace_seconds=args.pace_seconds,
                    cache_key=cache_keys[region],
                ): provider
                for provider in args.providers
            }
            for future in as_completed(futures):
                provider = futures[future]
                summary = future.result()
                results.append({"provider": provider, "region": region, **summary})
                write_json_atomic(
                    args.output,
                    {
                        "created_at": started_at.isoformat(),
                        "method": {
                            "queries": args.queries,
                            "warmups": args.warmups,
                            "pace_seconds": args.pace_seconds,
                            "regions": args.regions,
                            "cache_keys": cache_keys,
                        },
                        "results": results,
                    },
                )
                print(f"{provider} {region}: p50={summary['p50_ms']} ms", flush=True)

    print(f"Wrote region scout results to {args.output}", flush=True)


if __name__ == "__main__":
    main()
