import tempfile
import unittest
from pathlib import Path

from search_benchmark.quality.query_plan import (
    QueryPlanMismatchError,
    load_query_plan,
    write_query_plan,
)


class QueryPlanValidationTest(unittest.TestCase):
    def test_rejects_different_dataset_or_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset.csv"
            dataset.write_text("original_index,problem\n1,First\n")
            plan = root / "plan.json"
            write_query_plan(
                plan,
                dataset=dataset,
                limit=1,
                seed=42,
                model="model-a",
                queries_by_index={1: ["first query"]},
            )

            self.assertEqual(
                load_query_plan(plan, dataset=dataset, model="model-a"),
                {1: ["first query"]},
            )
            with self.assertRaises(QueryPlanMismatchError):
                load_query_plan(plan, dataset=dataset, model="model-b")

            dataset.write_text("original_index,problem\n1,Changed\n")
            with self.assertRaises(QueryPlanMismatchError):
                load_query_plan(plan, dataset=dataset, model="model-a")

    def test_rejects_string_as_query_list(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            path.write_text('{"queries_by_index":{"1":"not-a-list"}}')
            with self.assertRaises(ValueError):
                load_query_plan(path)


if __name__ == "__main__":
    unittest.main()
