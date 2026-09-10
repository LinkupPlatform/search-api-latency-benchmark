import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from search_benchmark.cli.region_scout import main as scout_main
from search_benchmark.cloud import (
    decode_query_plan,
    encode_query_plan,
    parse_metrics_log,
    prepare_cloud_run,
    run_quality_jobs,
    select_regions,
)


class CloudRunnerTest(unittest.TestCase):
    def test_select_regions_uses_lowest_p50_from_scout_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "region-scouts.json"
            path.write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "provider": "serper",
                                "region": "gcp-us-east4",
                                "p50_ms": 120,
                                "timed_calls": 50,
                                "successful": 50,
                                "failed": 0,
                            },
                            {
                                "provider": "serper",
                                "region": "gcp-us-west2",
                                "p50_ms": 100,
                                "timed_calls": 50,
                                "successful": 50,
                                "failed": 0,
                            },
                        ]
                    }
                )
            )
            self.assertEqual(select_regions(path), {"serper": "us-west2"})

    def test_select_regions_ignores_fast_failed_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "region-scouts.json"
            path.write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "provider": "serper",
                                "region": "gcp-us-east4",
                                "p50_ms": 10,
                                "timed_calls": 50,
                                "successful": 0,
                                "failed": 50,
                            },
                            {
                                "provider": "serper",
                                "region": "gcp-us-west2",
                                "p50_ms": 100,
                                "timed_calls": 50,
                                "successful": 50,
                                "failed": 0,
                            },
                        ]
                    }
                )
            )
            self.assertEqual(select_regions(path), {"serper": "us-west2"})

    def test_parse_metrics_log(self) -> None:
        metrics = parse_metrics_log('log prefix RESULT_JSON={"p50_ms":123.4}')
        self.assertEqual(metrics["p50_ms"], 123.4)

    def test_query_plan_round_trip(self) -> None:
        plan = {5: ["first query", "second query"]}
        self.assertEqual(decode_query_plan(encode_query_plan(plan)), plan)

    @patch("search_benchmark.cloud.shutil.which", return_value="/usr/bin/gcloud")
    @patch("search_benchmark.cloud.run_command")
    def test_cloud_setup_versions_local_keys_and_builds_image(
        self, mock_command, mock_which
    ) -> None:
        image = prepare_cloud_run(
            project="project",
            api_keys={"openai": "openai-key", "serper": "serper-key"},
        )

        self.assertEqual(
            image,
            "us-central1-docker.pkg.dev/project/search-api-latency-benchmark/"
            "benchmark:latest",
        )
        version_calls = [
            call for call in mock_command.call_args_list if "versions" in call.args[0]
        ]
        self.assertEqual(
            [call.kwargs["input"] for call in version_calls],
            ["openai-key", "serper-key"],
        )
        self.assertTrue(
            any(
                "builds" in call.args[0] and "submit" in call.args[0]
                for call in mock_command.call_args_list
            )
        )

    @patch("search_benchmark.cloud.shutil.which", return_value="/usr/bin/gcloud")
    @patch("search_benchmark.cloud.prepare_cloud_run", return_value="image")
    @patch("search_benchmark.cloud.run_command")
    def test_quality_job_uses_selected_provider_region(
        self, mock_command, mock_prepare, mock_which
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            scout_file = Path(directory) / "region-scouts.json"
            scout_file.write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "provider": "serper",
                                "region": "us-east4",
                                "p50_ms": 100,
                                "timed_calls": 50,
                                "successful": 50,
                                "failed": 0,
                            }
                        ]
                    }
                )
            )
            mock_command.side_effect = [
                "",
                "Execution [\x1b[1msimpleqa-serper-abc123\x1b[m] has successfully completed.",
                (
                    'RESULT_JSON={"correct_rate":1,"f_score":1,'
                    '"latency_p50_ms":100,"latency_p75_ms":120,'
                    '"latency_p90_ms":150}'
                ),
            ]
            metrics = run_quality_jobs(
                providers=["serper"],
                dataset_limit=100,
                seed=42,
                model="gpt-5.6-luna",
                pace_seconds=1,
                project="project",
                scout_file=scout_file,
                query_plan={5: ["shared query"]},
                api_keys={"openai": "openai", "serper": "serper"},
            )
        deploy_command = mock_command.call_args_list[0].args[0]
        self.assertIn("--region=us-east4", deploy_command)
        self.assertTrue(
            any("SIMPLEQA_QUERY_PLAN=" in argument for argument in deploy_command)
        )
        self.assertEqual(metrics["serper"]["latency_p90_ms"], 150)


class ScoutTest(unittest.TestCase):
    @patch("search_benchmark.cli.region_scout.scout_provider_region")
    @patch("search_benchmark.cli.region_scout.prepare_cloud_run", return_value="image")
    @patch("search_benchmark.cli.region_scout.parse_args")
    def test_scout_selects_lowest_p50(
        self, mock_args, mock_prepare, mock_scout
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "scouts.json"
            mock_args.return_value = SimpleNamespace(
                providers=["serper"],
                regions=["us-east4", "europe-west1"],
                queries=50,
                warmups=5,
                pace_seconds=1,
                output=output,
            )
            mock_scout.side_effect = [
                {"p50_ms": 100, "p90_ms": 120},
                {"p50_ms": 200, "p90_ms": 220},
            ]
            with patch.dict(
                "os.environ", {"GCP_PROJECT": "project", "SERPER_API_KEY": "key"}
            ):
                scout_main()
            run_record = json.loads(output.read_text())
        self.assertEqual(run_record["method"]["queries"], 50)
        self.assertEqual(len(run_record["results"]), 2)


if __name__ == "__main__":
    unittest.main()
