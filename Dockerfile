FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY src ./src
COPY data ./data

ENTRYPOINT ["uv", "run", "--frozen", "--no-dev", "python"]
CMD ["-m", "search_benchmark.latency_worker"]
