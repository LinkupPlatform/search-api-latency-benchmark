"""Search provider configuration and HTTP request construction."""

import json
import urllib.parse
from dataclasses import dataclass

PROVIDERS = (
    "linkup-flash",
    "exa-instant",
    "parallel-turbo",
    "tavily-ultra-fast",
    "brave-search",
    "serper",
)
API_KEY_ENV = {
    "linkup-flash": "LINKUP_API_KEY",
    "exa-instant": "EXA_API_KEY",
    "parallel-turbo": "PARALLEL_API_KEY",
    "tavily-ultra-fast": "TAVILY_API_KEY",
    "brave-search": "BRAVE_API_KEY",
    "serper": "SERPER_API_KEY",
}


@dataclass(frozen=True)
class RequestSpec:
    hostname: str
    method: str
    path: str
    body: str | None
    headers: dict[str, str]


def build_request(provider: str, api_key: str, query: str) -> RequestSpec:
    common_headers = {
        "content-type": "application/json",
        "user-agent": "regional-search-latency-benchmark/1.0",
    }
    if provider == "linkup-flash":
        return RequestSpec(
            hostname="api.linkup.so",
            method="POST",
            path="/v1/search",
            body=json.dumps(
                {
                    "q": query,
                    "depth": "flash",
                    "outputType": "searchResults",
                }
            ),
            headers={
                **common_headers,
                "authorization": f"Bearer {api_key}",
            },
        )
    if provider == "exa-instant":
        return RequestSpec(
            hostname="api.exa.ai",
            method="POST",
            path="/search",
            body=json.dumps(
                {
                    "query": query,
                    "type": "instant",
                    "contents": {"highlights": True},
                }
            ),
            headers={
                **common_headers,
                "x-api-key": api_key,
            },
        )
    if provider == "parallel-turbo":
        return RequestSpec(
            hostname="api.parallel.ai",
            method="POST",
            path="/v1/search",
            body=json.dumps(
                {
                    "objective": query,
                    "search_queries": [query],
                    "mode": "turbo",
                }
            ),
            headers={
                **common_headers,
                "x-api-key": api_key,
            },
        )
    if provider == "tavily-ultra-fast":
        return RequestSpec(
            hostname="api.tavily.com",
            method="POST",
            path="/search",
            body=json.dumps(
                {
                    "query": query,
                    "search_depth": "ultra-fast",
                    "include_answer": False,
                    "include_raw_content": False,
                    "include_images": False,
                }
            ),
            headers={
                **common_headers,
                "authorization": f"Bearer {api_key}",
            },
        )
    if provider == "brave-search":
        return RequestSpec(
            hostname="api.search.brave.com",
            method="GET",
            path="/res/v1/web/search?"
            + urllib.parse.urlencode(
                {
                    "q": query,
                    "country": "us",
                    "extra_snippets": "true",
                    "text_decorations": "false",
                }
            ),
            body=None,
            headers={
                "accept": "application/json",
                "user-agent": common_headers["user-agent"],
                "x-subscription-token": api_key,
            },
        )
    if provider == "serper":
        return RequestSpec(
            hostname="google.serper.dev",
            method="POST",
            path="/search",
            body=json.dumps({"q": query}),
            headers={
                **common_headers,
                "x-api-key": api_key,
            },
        )
    raise ValueError(f"Unsupported provider: {provider}")
