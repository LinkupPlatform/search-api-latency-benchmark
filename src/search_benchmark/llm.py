"""Minimal OpenAI Responses API client for structured outputs."""

import json
from typing import Any, cast

from openai import OpenAI


def openai_json(
    *,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    schema_name: str,
    schema: dict[str, Any],
    timeout: int = 120,
) -> dict[str, Any]:
    client = OpenAI(api_key=api_key, timeout=timeout)
    response = client.responses.create(
        model=model,
        input=cast(Any, messages),
        reasoning={"effort": "none"},
        text={
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "strict": True,
                "schema": schema,
            }
        },
    )
    if not response.output_text:
        raise RuntimeError("OpenAI Responses API returned no text output")
    return json.loads(response.output_text)
