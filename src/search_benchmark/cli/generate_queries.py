"""CLI for generating a reusable SimpleQA web-search query plan."""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from search_benchmark.arguments import positive_int
from search_benchmark.quality.dataset import load_simpleqa_dataset
from search_benchmark.quality.query_plan import (
    QueryPlanMismatchError,
    build_query_plan,
    load_query_plan,
    write_query_plan,
)

DEFAULT_MODEL = "gpt-5.6-luna"
DATA_DIRECTORY = Path("data")

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate reusable web-search queries for SimpleQA rows."
    )
    parser.add_argument(
        "--dataset", type=Path, default=DATA_DIRECTORY / "simpleqa_verified.csv"
    )
    parser.add_argument("--dataset-limit", type=positive_int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--output", type=Path, default=DATA_DIRECTORY / "simpleqa-query-plan.json"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY")
    samples = load_simpleqa_dataset(
        args.dataset, limit=args.dataset_limit, seed=args.seed
    )
    try:
        existing_plan = (
            load_query_plan(args.output, dataset=args.dataset, model=args.model)
            if args.output.exists()
            else {}
        )
    except QueryPlanMismatchError as error:
        print(f"{error}; regenerating it.", flush=True)
        existing_plan = {}

    def checkpoint(query_plan: dict[int, list[str]]) -> None:
        write_query_plan(
            args.output,
            dataset=args.dataset,
            limit=args.dataset_limit,
            seed=args.seed,
            model=args.model,
            queries_by_index=query_plan,
        )

    build_query_plan(
        samples,
        api_key=api_key,
        model=args.model,
        existing_plan=existing_plan,
        checkpoint=checkpoint,
    )
    print(f"Wrote query plan to {args.output}", flush=True)


if __name__ == "__main__":
    main()
