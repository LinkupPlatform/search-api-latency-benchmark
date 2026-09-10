# Regional Search API Latency Benchmark

Measure web-search latency and search-augmented SimpleQA Verified quality locally or from GCP.
Latency is wall-clock time from request start through the complete response body; warmups, pacing,
and Cloud Run startup are excluded.

## Providers

- Linkup Flash
- Exa Instant
- Parallel Turbo
- Tavily Ultra Fast
- Brave Search
- Serper

## Setup

```bash
uv sync --frozen
cp .env.example .env
```

Set the required provider keys in `.env`.

## GCP Cloud Run setup

```bash
gcloud auth login
gcloud services enable \
  secretmanager.googleapis.com \
  cloudbuild.googleapis.com \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  logging.googleapis.com \
  --project="$GCP_PROJECT"
```

Set `GCP_PROJECT` and API keys in `.env`. Commands sync secrets, build the image, and deploy jobs.

If Cloud Build lacks permission:

```bash
gcloud projects add-iam-policy-binding "$GCP_PROJECT" \
  --member="user:YOUR_EMAIL" \
  --role="roles/cloudbuild.builds.editor"
```

If the build cannot access its archive or push its image:

```bash
PROJECT_NUMBER="$(gcloud projects describe "$GCP_PROJECT" --format='value(projectNumber)')"
BUILD_SERVICE_ACCOUNT="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

gcloud storage buckets add-iam-policy-binding "gs://${GCP_PROJECT}_cloudbuild" \
  --member="serviceAccount:${BUILD_SERVICE_ACCOUNT}" \
  --role="roles/storage.objectViewer"

gcloud artifacts repositories add-iam-policy-binding search-api-latency-benchmark \
  --location=us-central1 \
  --project="$GCP_PROJECT" \
  --member="serviceAccount:${BUILD_SERVICE_ACCOUNT}" \
  --role="roles/artifactregistry.writer"
```

## SimpleQA Verified search-quality benchmark

For each sampled question, the harness generates concise search queries without seeing the gold
answer, searches every provider, generates an answer from normalized results, and grades it.
The saved plan is reused; duplicate queries run once per provider. Latency measures web-search calls
only, not LLM planning, answering, or grading.

### Dataset provenance and citation

`data/simpleqa_verified.csv` is the 1,000-row `eval` split of
[Google's SimpleQA Verified](https://huggingface.co/datasets/google/simpleqa-verified), checked in
for reproducibility. It is [MIT licensed](https://huggingface.co/datasets/google/simpleqa-verified/blob/main/README.md);
its SHA-256 is
`b5db21155444763543fe31b67e7cf28ce2bb225742a5b889421c5f182e2f92f5`.

Citation: [Haas et al. (2025), *SimpleQA Verified*](https://arxiv.org/abs/2509.07968).

Generate the plan once per dataset, sample, seed, and model:

```bash
uv run simpleqa-generate-queries
```

It writes `data/simpleqa-query-plan.json`, runs up to five requests concurrently, and checkpoints
every 50 rows. Local execution reuses that plan:

```bash
uv run simpleqa-quality \
  --providers linkup-flash tavily-ultra-fast \
  --output results/simpleqa-quality.json
```

Cloud execution uses each provider's lowest-p50 region from `data/region-scouts.json`:

```bash
uv run simpleqa-quality \
  --execution gcloud \
  --providers linkup-flash exa-instant parallel-turbo tavily-ultra-fast brave-search
```

Cloud runs fail when a provider has no scouted region. Every worker receives the same selected plan;
completion prints p50, F-score, and accuracy.

**Options.** Choose any of `linkup-flash`, `exa-instant`, `parallel-turbo`,
`tavily-ultra-fast`, `brave-search`, and `serper`. The LLM defaults to `gpt-5.6-luna`; change it
with `--model`. Both commands select 100 rows with seed `42` by default; use `--dataset-limit` and
`--seed` to change that.

**Results.** Local runs atomically save queries, normalized results, answers, and grades after each
answer. Cloud runs save combined metrics locally and progress in Cloud Logging, but not worker rows.
Both print accuracy, F-score, and search p50/p75/p90; JSON also includes correct, incorrect,
not-attempted, and correct-given-attempted counts. Accuracy is `correct / total questions`; search
latency covers each unique provider request (including failures), not LLM query planning, answering,
or grading.

**Scope.** SimpleQA Verified is a no-tools factuality benchmark. This retrieval-augmented use
measures search-provider performance and is not comparable to its official leaderboard. This
harness uses the official autorater prompt with GPT-5.6 Luna, rather than the paper's GPT-4.1
autorater.

## Regional scouting

Runs 50 queries after 5 warmups for every provider in four regions. Each region gets a distinct
two-letter cache key shared by its providers, reducing cache reuse across regions and runs:

```bash
uv run region-scout
```

Default regions: `us-east4`, `us-central1`, `us-west2`, and `europe-west1`. Results replace
`data/region-scouts.json` atomically and select the cloud quality region for each provider.
Regions run sequentially; provider jobs run in parallel within a region; requests within a job stay
sequential. This is a provider-friendly regional measurement, not a same-origin comparison.

## Interpretation limits

- Providers have different defaults, response sizes, and result formats.
- Cache-busting and pacing can affect results; pacing is outside the timer.
- Results measure warm, sequential latency—not cold connections or concurrent load.
- Latency and quality are separate measures; failures remain in latency percentiles.

## Test

```bash
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run python -m unittest discover -s tests -v
```
