FROM ghcr.io/astral-sh/uv:0.13.5@sha256:32a7bb39b496babbd997b0a39a4d66c126bdaf5661b302bf6f06abda8deca6ff AS uv

FROM python:3.13-slim-bookworm@sha256:e32be0e55814a08977501ca863271d9bd02443710db8f2afaf8372e2c4508530

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
  && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir "pip==26.2.2"

COPY --from=uv /uv /uvx /bin/

WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY src ./src
RUN uv sync --frozen --no-dev

COPY data ./data

ENTRYPOINT ["uv", "run", "--frozen", "--no-dev", "python"]
CMD ["-m", "search_benchmark.latency_worker"]
