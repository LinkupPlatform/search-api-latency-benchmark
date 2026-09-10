import json
import unittest

from search_benchmark.latency import percentile, query_at, summarize
from search_benchmark.providers import build_request


class PercentileTest(unittest.TestCase):
    def test_interpolates_percentile(self) -> None:
        self.assertEqual(percentile([100, 200, 300, 400], 0.5), 250)
        self.assertEqual(percentile([100, 200, 300, 400], 0.9), 370)


class QueryTest(unittest.TestCase):
    def test_queries_are_not_modified_without_a_cache_key(self) -> None:
        queries = ["first", "second"]
        self.assertEqual(query_at(queries, 3), "second")

    def test_cache_key_is_shared_across_queries(self) -> None:
        self.assertEqual(
            query_at(["first", "second"], 1, cache_key="ab"),
            "second ab",
        )


class SummaryTest(unittest.TestCase):
    def test_failures_are_included_in_latency_percentiles(self) -> None:
        summary = summarize(
            provider="test",
            region="local",
            rows=[
                {"ok": True, "elapsed_ms": 100, "response_bytes": 10},
                {"ok": False, "elapsed_ms": 900, "status": 429},
            ],
            warmups=0,
            pace_seconds=1,
            cache_key="ab",
        )
        self.assertEqual(summary["p50_ms"], 500)
        self.assertEqual(summary["failed"], 1)

    def test_all_failures_still_produce_a_summary(self) -> None:
        summary = summarize(
            provider="test",
            region="local",
            rows=[{"ok": False, "elapsed_ms": 100, "status": 401}],
            warmups=0,
            pace_seconds=0,
            cache_key=None,
        )
        self.assertEqual(summary["failed"], 1)
        self.assertIsNone(summary["mean_response_bytes"])


class RequestShapeTest(unittest.TestCase):
    def test_linkup_flash(self) -> None:
        request = build_request("linkup-flash", "key", "query")
        self.assertEqual(request.hostname, "api.linkup.so")
        self.assertEqual(
            json.loads(request.body or ""),
            {"q": "query", "depth": "flash", "outputType": "searchResults"},
        )

    def test_exa_instant_uses_highlights(self) -> None:
        request = build_request("exa-instant", "key", "query")
        payload = json.loads(request.body or "")
        self.assertEqual(payload["type"], "instant")
        self.assertEqual(payload["contents"], {"highlights": True})
        self.assertNotIn("numResults", payload)
        self.assertNotIn("text", payload["contents"])

    def test_parallel_turbo(self) -> None:
        request = build_request("parallel-turbo", "key", "query")
        self.assertEqual(
            json.loads(request.body or ""),
            {
                "objective": "query",
                "search_queries": ["query"],
                "mode": "turbo",
            },
        )

    def test_tavily_ultra_fast_includes_native_content(self) -> None:
        request = build_request("tavily-ultra-fast", "key", "query")
        payload = json.loads(request.body or "")
        self.assertEqual(payload["search_depth"], "ultra-fast")
        self.assertFalse(payload["include_raw_content"])
        self.assertNotIn("max_results", payload)
        self.assertNotIn("content", payload)

    def test_brave_uses_default_result_count_and_extra_snippets(self) -> None:
        request = build_request("brave-search", "key", "query")
        self.assertNotIn("count=", request.path)
        self.assertIn("extra_snippets=true", request.path)

    def test_serper_is_not_serpapi(self) -> None:
        request = build_request("serper", "key", "query")
        self.assertEqual(request.hostname, "google.serper.dev")
        self.assertNotIn("num", json.loads(request.body or ""))


if __name__ == "__main__":
    unittest.main()
