"""GCP Cloud Run Jobs orchestration for the quality benchmark."""

import base64
import gzip
import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from search_benchmark.providers import API_KEY_ENV

PROVIDER_SECRET = {
    "linkup-flash": "linkup-api-key",
    "exa-instant": "exa-api-key",
    "parallel-turbo": "parallel-api-key",
    "tavily-ultra-fast": "tavily-api-key",
    "brave-search": "brave-api-key",
    "serper": "serper-api-key",
}
OPENAI_SECRET = "openai-api-key"
ARTIFACT_REGION = "us-central1"
ARTIFACT_REPOSITORY = "search-api-latency-benchmark"
IMAGE_NAME = "benchmark"
MACHINE_PREFIX = "RESULT_JSON="
QUERY_PLAN_ENV = "SIMPLEQA_QUERY_PLAN"
ANSI_ESCAPE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


def run_command(
    command: list[str], *, input: str | None = None, stream: bool = False
) -> str:
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        input=input,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.returncode:
        if stream and completed.stdout:
            print(completed.stdout, end="", flush=True)
        raise subprocess.CalledProcessError(
            completed.returncode, command, output=completed.stdout
        )
    return completed.stdout


def ensure_secret(
    *,
    project: str,
    name: str,
    value: str,
    accessor_service_account: str,
    stream: bool = False,
) -> None:
    """Create a secret or append the local value as its newest version."""
    if stream:
        print(f"Configuring Secret Manager access for {name}...", flush=True)
    try:
        run_command(
            ["gcloud", "secrets", "describe", name, f"--project={project}"],
            stream=stream,
        )
    except subprocess.CalledProcessError:
        run_command(
            [
                "gcloud",
                "secrets",
                "create",
                name,
                f"--project={project}",
                "--replication-policy=automatic",
                "--data-file=-",
            ],
            input=value,
            stream=stream,
        )
    else:
        run_command(
            [
                "gcloud",
                "secrets",
                "versions",
                "add",
                name,
                f"--project={project}",
                "--data-file=-",
            ],
            input=value,
            stream=stream,
        )
    run_command(
        [
            "gcloud",
            "secrets",
            "add-iam-policy-binding",
            name,
            f"--project={project}",
            f"--member=serviceAccount:{accessor_service_account}",
            "--role=roles/secretmanager.secretAccessor",
        ],
        stream=stream,
    )


def build_image(project: str, *, stream: bool = False) -> str:
    """Ensure the fixed Artifact Registry repository and build the project image."""
    if stream:
        print("Preparing the container repository...", flush=True)
    try:
        run_command(
            [
                "gcloud",
                "artifacts",
                "repositories",
                "describe",
                ARTIFACT_REPOSITORY,
                f"--location={ARTIFACT_REGION}",
                f"--project={project}",
            ],
            stream=stream,
        )
    except subprocess.CalledProcessError:
        run_command(
            [
                "gcloud",
                "artifacts",
                "repositories",
                "create",
                ARTIFACT_REPOSITORY,
                "--repository-format=docker",
                f"--location={ARTIFACT_REGION}",
                f"--project={project}",
            ],
            stream=stream,
        )
    image = (
        f"{ARTIFACT_REGION}-docker.pkg.dev/{project}/{ARTIFACT_REPOSITORY}/"
        f"{IMAGE_NAME}:latest"
    )
    if stream:
        print("Building and publishing the container image...", flush=True)
    run_command(
        [
            "gcloud",
            "builds",
            "submit",
            ".",
            f"--tag={image}",
            f"--project={project}",
        ],
        stream=stream,
    )
    return image


def prepare_cloud_run(
    *, project: str, api_keys: dict[str, str], stream: bool = False
) -> str:
    if not shutil.which("gcloud"):
        raise RuntimeError("gcloud is required for Cloud Run execution")
    if stream:
        print("Preparing Cloud Run credentials and image...", flush=True)
    project_number = run_command(
        [
            "gcloud",
            "projects",
            "describe",
            project,
            "--format=value(projectNumber)",
        ],
        stream=stream,
    ).strip()
    if not project_number:
        raise RuntimeError(f"Could not determine the project number for {project}")
    runtime_service_account = f"{project_number}-compute@developer.gserviceaccount.com"
    for provider, api_key in api_keys.items():
        secret_name = (
            OPENAI_SECRET if provider == "openai" else PROVIDER_SECRET[provider]
        )
        ensure_secret(
            project=project,
            name=secret_name,
            value=api_key,
            accessor_service_account=runtime_service_account,
            stream=stream,
        )
    return build_image(project, stream=stream)


def select_regions(scout_file: Path) -> dict[str, str]:
    """Select each provider's lowest-p50 region from a region scout run record."""
    try:
        results = json.loads(scout_file.read_text())["results"]
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as error:
        raise ValueError(
            f"Cannot read region scout results from {scout_file}; run region-scout first"
        ) from error
    if not isinstance(results, list):
        raise ValueError(f"Region scout results in {scout_file} must be a list")

    candidates: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        if not isinstance(result, dict):
            continue
        provider = result.get("provider")
        timed_calls = result.get("timed_calls")
        successful = result.get("successful")
        failed = result.get("failed")
        is_healthy = (
            isinstance(timed_calls, int)
            and timed_calls > 0
            and successful == timed_calls
            and failed == 0
        )
        if (
            isinstance(provider, str)
            and isinstance(result.get("p50_ms"), (int, float))
            and is_healthy
        ):
            candidates.setdefault(provider, []).append(result)
    return {
        provider: str(
            min(rows, key=lambda row: float(row["p50_ms"]))["region"]
        ).removeprefix("gcp-")
        for provider, rows in candidates.items()
    }


def parse_metrics_log(output: str) -> dict[str, float | int]:
    for line in output.splitlines():
        if MACHINE_PREFIX in line:
            return json.loads(line.split(MACHINE_PREFIX, 1)[1])
    raise RuntimeError("Cloud Run job did not emit RESULT_JSON metrics")


def read_execution_metrics(
    *,
    project: str,
    job_name: str,
    region: str,
    execution_output: str,
    attempts: int = 12,
    retry_seconds: int = 5,
    stream: bool = False,
) -> dict[str, float | int]:
    """Wait for and return metrics emitted by one completed Cloud Run execution."""
    execution_output = ANSI_ESCAPE.sub("", execution_output)
    match = re.search(r"Execution \[([^\]]+)\]", execution_output)
    if not match:
        raise RuntimeError("Could not determine the Cloud Run execution name")
    execution_name = match.group(1)
    query = (
        'resource.type="cloud_run_job" '
        f'AND resource.labels.job_name="{job_name}" '
        f'AND resource.labels.location="{region}" '
        f'AND labels."run.googleapis.com/execution_name"="{execution_name}" '
        'AND textPayload:"RESULT_JSON="'
    )
    for attempt in range(1, attempts + 1):
        output = run_command(
            [
                "gcloud",
                "logging",
                "read",
                query,
                f"--project={project}",
                "--limit=1",
                "--format=value(textPayload)",
            ],
            stream=stream,
        )
        try:
            return parse_metrics_log(output)
        except RuntimeError:
            if attempt == attempts:
                break
            print(
                f"Waiting for logs from {execution_name} ({attempt}/{attempts})...",
                flush=True,
            )
            time.sleep(retry_seconds)
    raise RuntimeError(
        f"Cloud Run execution {execution_name} completed but did not emit RESULT_JSON"
    )


def encode_query_plan(query_plan: dict[int, list[str]]) -> str:
    serialized = json.dumps(query_plan, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(gzip.compress(serialized)).decode().rstrip("=")


def decode_query_plan(encoded: str) -> dict[int, list[str]]:
    padded = encoded + "=" * (-len(encoded) % 4)
    decoded = gzip.decompress(base64.urlsafe_b64decode(padded))
    return {int(index): list(queries) for index, queries in json.loads(decoded).items()}


def run_quality_jobs(
    *,
    providers: list[str],
    dataset_limit: int,
    seed: int,
    model: str,
    pace_seconds: float,
    project: str,
    scout_file: Path,
    query_plan: dict[int, list[str]],
    api_keys: dict[str, str],
) -> dict[str, dict[str, float | int]]:
    regions = select_regions(scout_file)
    missing = [provider for provider in providers if provider not in regions]
    if missing:
        raise ValueError(
            "No scouted region for "
            + ", ".join(missing)
            + f"; run region-scout to update {scout_file}"
        )

    metrics = {}
    encoded_query_plan = encode_query_plan(query_plan)
    if len(encoded_query_plan) > 30_000:
        raise ValueError(
            "Compressed query plan exceeds the Cloud Run environment limit; "
            "reduce --dataset-limit"
        )
    unique_searches = {query for queries in query_plan.values() for query in queries}
    print(
        f"Preparing SimpleQA Cloud Run jobs for {len(providers)} providers...",
        flush=True,
    )
    image = prepare_cloud_run(project=project, api_keys=api_keys, stream=True)
    for provider in providers:
        region = regions[provider]
        job_name = "simpleqa-" + provider.replace("_", "-")
        print(
            f"Testing {provider} in {region} "
            f"({dataset_limit} questions, {len(unique_searches)} unique searches)...",
            flush=True,
        )
        worker_args = ",".join(
            [
                "run",
                "--frozen",
                "--no-dev",
                "simpleqa-quality",
                "--providers",
                provider,
                "--dataset-limit",
                str(dataset_limit),
                "--seed",
                str(seed),
                "--model",
                model,
                "--pace-seconds",
                str(pace_seconds),
                "--output",
                "/tmp/simpleqa-quality.json",
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
                "--set-env-vars="
                f"BENCHMARK_MACHINE_OUTPUT=1,{QUERY_PLAN_ENV}={encoded_query_plan}",
                "--set-secrets="
                f"OPENAI_API_KEY={OPENAI_SECRET}:latest,"
                f"{API_KEY_ENV[provider]}={PROVIDER_SECRET[provider]}:latest",
                "--task-timeout=7200s",
                "--max-retries=0",
                "--memory=1Gi",
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
        metrics[provider] = read_execution_metrics(
            project=project,
            job_name=job_name,
            region=region,
            execution_output=execution_output,
        )
        provider_metrics = metrics[provider]
        print(
            f"{provider} complete in {region}: "
            f"p50={float(provider_metrics['latency_p50_ms']):.1f} ms, "
            f"F-score={float(provider_metrics['f_score']):.3f}, "
            f"accuracy={float(provider_metrics['correct_rate']):.1%}",
            flush=True,
        )
    return metrics
