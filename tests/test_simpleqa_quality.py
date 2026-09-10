import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from search_benchmark.arguments import nonnegative_int
from search_benchmark.cli.quality import (
    execute_search_safely,
    extract_search_results,
    format_metrics_table,
    latency_metrics,
    main,
    nonnegative_float,
    parse_args,
    positive_int,
)
from search_benchmark.llm import openai_json
from search_benchmark.quality.answers import answer_question
from search_benchmark.quality.grading import calculate_metrics


class MetricsTest(unittest.TestCase):
    def test_simpleqa_f_score(self) -> None:
        metrics = calculate_metrics(["CORRECT", "INCORRECT", "NOT_ATTEMPTED"])
        self.assertAlmostEqual(metrics["correct_rate"], 1 / 3)
        self.assertEqual(metrics["correct_given_attempted"], 0.5)
        self.assertAlmostEqual(metrics["f_score"], 0.4)

    def test_empty_grades(self) -> None:
        self.assertEqual(calculate_metrics([])["f_score"], 0)

    def test_latency_percentiles_include_every_search(self) -> None:
        metrics = latency_metrics(
            [
                {"elapsed_ms": 100},
                {"elapsed_ms": 200},
                {"elapsed_ms": 300},
                {"elapsed_ms": 900, "error": "TimeoutError"},
            ]
        )
        self.assertEqual(metrics["latency_p50_ms"], 250)
        self.assertEqual(metrics["latency_p75_ms"], 450)
        self.assertEqual(metrics["latency_p90_ms"], 720)

    def test_metrics_table(self) -> None:
        table = format_metrics_table(
            {
                "serper": {
                    "correct_rate": 0.75,
                    "f_score": 0.6,
                    "latency_p50_ms": 100,
                    "latency_p75_ms": 150,
                    "latency_p90_ms": 200,
                }
            }
        )
        self.assertIn("Accuracy", table)
        self.assertIn("75.0%", table)
        self.assertIn("150.0 ms", table)


class SearchResultExtractionTest(unittest.TestCase):
    def test_brave_results(self) -> None:
        results = extract_search_results(
            "brave-search",
            {
                "web": {
                    "results": [
                        {
                            "title": "Title",
                            "description": "Description",
                            "extra_snippets": ["Extra"],
                            "url": "https://example.com",
                        }
                    ]
                }
            },
        )
        self.assertEqual(results[0]["url"], "https://example.com")
        self.assertIn("Extra", results[0]["text"])

    def test_serper_results(self) -> None:
        results = extract_search_results(
            "serper",
            {
                "organic": [
                    {
                        "title": "Title",
                        "snippet": "Snippet",
                        "link": "https://example.com",
                    }
                ]
            },
        )
        self.assertEqual(
            results, [{"text": "Title\nSnippet", "url": "https://example.com"}]
        )

    def test_exa_highlights(self) -> None:
        results = extract_search_results(
            "exa-instant",
            {
                "results": [
                    {
                        "title": "Title",
                        "highlights": ["First fact", "Second fact"],
                        "url": "https://example.com",
                    }
                ]
            },
        )
        self.assertIn("First fact", results[0]["text"])


class ArgumentValidationTest(unittest.TestCase):
    def test_positive_int(self) -> None:
        self.assertEqual(positive_int("3"), 3)
        with self.assertRaises(Exception):
            positive_int("0")

    def test_nonnegative_float(self) -> None:
        self.assertEqual(nonnegative_float("0"), 0)
        with self.assertRaises(Exception):
            nonnegative_float("-1")

    def test_nonnegative_int(self) -> None:
        self.assertEqual(nonnegative_int("0"), 0)
        with self.assertRaises(Exception):
            nonnegative_int("-1")

    @patch("sys.argv", ["simpleqa-quality", "--providers", "serper"])
    def test_dataset_defaults(self) -> None:
        args = parse_args()
        self.assertEqual(args.dataset_limit, 100)
        self.assertEqual(args.seed, 42)


class SearchFailureTest(unittest.TestCase):
    @patch("search_benchmark.cli.quality.execute_search")
    def test_search_exception_becomes_auditable_failure(self, mock_execute) -> None:
        mock_execute.side_effect = TimeoutError()
        result = execute_search_safely("serper", "key", "query")
        self.assertEqual(result["error"], "TimeoutError")
        self.assertEqual(result["results"], [])


class AnswerGenerationTest(unittest.TestCase):
    @patch("search_benchmark.quality.answers.openai_json")
    def test_answer_uses_search_evidence(self, mock_openai_json) -> None:
        mock_openai_json.return_value = {"answer": "Paris"}
        answer = answer_question(
            "What is the capital of France?",
            [
                {
                    "query": "France capital",
                    "results": [{"text": "Paris is the capital.", "url": "example"}],
                }
            ],
            api_key="key",
            model="gpt-5.6-luna",
        )
        self.assertEqual(answer, "Paris")
        messages = mock_openai_json.call_args.kwargs["messages"]
        self.assertEqual([message["role"] for message in messages], ["user"])
        self.assertIn("France capital", messages[0]["content"])


class ResponsesApiTest(unittest.TestCase):
    @patch("search_benchmark.llm.OpenAI")
    def test_requests_structured_output_from_responses_api(
        self, mock_openai: MagicMock
    ) -> None:
        mock_openai.return_value.responses.create.return_value.output_text = (
            '{"answer":"Paris"}'
        )

        result = openai_json(
            api_key="key",
            model="gpt-5.6-luna",
            messages=[{"role": "user", "content": "What is the capital of France?"}],
            schema_name="answer",
            schema={
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        )

        self.assertEqual(result, {"answer": "Paris"})
        mock_openai.assert_called_once_with(api_key="key", timeout=120)
        request = mock_openai.return_value.responses.create.call_args.kwargs
        self.assertEqual(request["text"]["format"]["type"], "json_schema")
        self.assertEqual(
            request["input"][0]["content"], "What is the capital of France?"
        )


class PipelineTest(unittest.TestCase):
    @patch("search_benchmark.cloud.run_quality_jobs")
    @patch("search_benchmark.cli.quality.load_query_plan")
    @patch("search_benchmark.cli.quality.load_simpleqa_dataset")
    @patch("search_benchmark.cli.quality.parse_args")
    def test_cloud_execution_sends_only_selected_query_plan_rows(
        self, mock_args, mock_dataset, mock_plan, mock_run_quality
    ) -> None:
        samples = [
            {"original_index": 1, "problem": "First", "answer": "one"},
            {"original_index": 3, "problem": "Third", "answer": "three"},
        ]
        mock_dataset.return_value = samples
        mock_plan.return_value = {
            1: ["first query"],
            2: ["unused query"],
            3: ["third query"],
        }
        mock_run_quality.return_value = {
            "serper": {
                "correct_rate": 1,
                "f_score": 1,
                "latency_p50_ms": 100,
                "latency_p75_ms": 100,
                "latency_p90_ms": 100,
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            mock_args.return_value = SimpleNamespace(
                execution="gcloud",
                providers=["serper"],
                dataset_limit=2,
                seed=42,
                model="gpt-5.6-luna",
                pace_seconds=0,
                output=output,
                scout_file=Path("region-scouts.json"),
                dataset=Path("simpleqa_verified.csv"),
                query_plan=Path("simpleqa-query-plan.json"),
            )
            with patch.dict(
                "os.environ",
                {
                    "GCP_PROJECT": "project",
                    "OPENAI_API_KEY": "openai",
                    "SERPER_API_KEY": "serper",
                },
                clear=True,
            ):
                main()

        self.assertEqual(
            mock_run_quality.call_args.kwargs["query_plan"],
            {1: ["first query"], 3: ["third query"]},
        )

    @patch("search_benchmark.cli.quality.grade_answer", return_value="CORRECT")
    @patch("search_benchmark.cli.quality.answer_question", return_value="Paris")
    @patch("search_benchmark.cli.quality.execute_search_safely")
    @patch(
        "search_benchmark.cli.quality.load_query_plan",
        return_value={1: ["France capital"]},
    )
    @patch("search_benchmark.cli.quality.load_simpleqa_dataset")
    @patch("search_benchmark.cli.quality.parse_args")
    def test_one_command_runs_same_queries_for_each_provider(
        self,
        mock_args,
        mock_dataset,
        mock_plan,
        mock_search,
        mock_answer,
        mock_grade,
    ) -> None:
        sample = {
            "original_index": 1,
            "problem": "What is the capital of France?",
            "answer": "Paris",
            "topic": "Geography",
            "answer_type": "Place",
            "multi_step": False,
            "requires_reasoning": False,
            "urls": ["a", "b"],
        }
        mock_dataset.return_value = [sample]
        mock_search.side_effect = lambda provider, key, query: {
            "query": query,
            "status": 200,
            "elapsed_ms": 1,
            "results": [{"text": "Paris", "url": "example"}],
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            mock_args.return_value = SimpleNamespace(
                execution="local",
                providers=["linkup-flash", "serper"],
                dataset_limit=1,
                seed=42,
                model="gpt-5.6-luna",
                pace_seconds=0,
                output=output,
                gcloud_project=None,
                gcloud_image=None,
                scout_file=Path("region-scouts.json"),
                dataset=Path("simpleqa_verified.csv"),
                query_plan=Path("simpleqa-query-plan.json"),
            )
            with patch.dict(
                "os.environ",
                {
                    "OPENAI_API_KEY": "openai",
                    "LINKUP_API_KEY": "linkup",
                    "SERPER_API_KEY": "serper",
                },
                clear=True,
            ):
                main()
            result = json.loads(output.read_text())

        self.assertEqual(mock_plan.call_count, 1)
        self.assertEqual(
            [call.args[2] for call in mock_search.call_args_list],
            ["France capital", "France capital"],
        )
        self.assertEqual(result["metrics"]["linkup-flash"]["correct"], 1)
        self.assertEqual(result["metrics"]["serper"]["correct"], 1)
        self.assertEqual(result["metrics"]["linkup-flash"]["latency_p50_ms"], 1)

    @patch(
        "search_benchmark.cli.quality.grade_answer",
        side_effect=[TimeoutError(), "CORRECT"],
    )
    @patch("search_benchmark.cli.quality.answer_question", return_value="Paris")
    @patch("search_benchmark.cli.quality.execute_search_safely")
    @patch(
        "search_benchmark.cli.quality.load_query_plan",
        return_value={1: ["France capital"]},
    )
    @patch("search_benchmark.cli.quality.load_simpleqa_dataset")
    @patch("search_benchmark.cli.quality.parse_args")
    def test_resume_reuses_search_and_answer_after_grading_failure(
        self,
        mock_args,
        mock_dataset,
        mock_plan,
        mock_search,
        mock_answer,
        mock_grade,
    ) -> None:
        mock_dataset.return_value = [
            {
                "original_index": 1,
                "problem": "What is the capital of France?",
                "answer": "Paris",
                "topic": "Geography",
                "answer_type": "Place",
                "multi_step": False,
                "requires_reasoning": False,
                "urls": [],
            }
        ]
        mock_search.return_value = {
            "query": "France capital",
            "status": 200,
            "elapsed_ms": 1,
            "results": [{"text": "Paris", "url": "example"}],
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            mock_args.return_value = SimpleNamespace(
                execution="local",
                providers=["serper"],
                dataset_limit=1,
                seed=42,
                model="gpt-5.6-luna",
                pace_seconds=0,
                output=output,
                scout_file=Path("region-scouts.json"),
                dataset=Path("simpleqa_verified.csv"),
                query_plan=Path("simpleqa-query-plan.json"),
            )
            environment = {
                "OPENAI_API_KEY": "openai",
                "SERPER_API_KEY": "serper",
            }
            with patch.dict("os.environ", environment, clear=True):
                with self.assertRaises(TimeoutError):
                    main()
                partial = json.loads(output.read_text())
                self.assertEqual(partial["rows"][0]["predicted_answer"], "Paris")
                self.assertIsNone(partial["rows"][0]["grade"])
                main()
            result = json.loads(output.read_text())

        self.assertEqual(mock_search.call_count, 1)
        self.assertEqual(mock_answer.call_count, 1)
        self.assertEqual(mock_grade.call_count, 2)
        self.assertEqual(result["rows"][0]["grade"], "CORRECT")
        self.assertEqual(result["metrics"]["serper"]["correct"], 1)


if __name__ == "__main__":
    unittest.main()
