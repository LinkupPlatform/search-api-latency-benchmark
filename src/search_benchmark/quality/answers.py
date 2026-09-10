"""Generate one SimpleQA answer from normalized web-search results."""

import json
from typing import Any

from search_benchmark.llm import openai_json

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


def answer_question(
    question: str,
    searches: list[dict[str, Any]],
    *,
    api_key: str,
    model: str,
) -> str:
    evidence = [
        {
            "query": search["query"],
            "results": search["results"],
            **({"error": search["error"]} if search.get("error") else {}),
        }
        for search in searches
    ]
    result = openai_json(
        api_key=api_key,
        model=model,
        schema_name="simpleqa_answer",
        schema=ANSWER_SCHEMA,
        messages=[
            {
                "role": "user",
                "content": (
                    "Answer the factual question using only the supplied web search "
                    "results. Give one concise direct answer with no citations or "
                    "explanation. If the evidence is insufficient or conflicting, "
                    'answer exactly "I don\'t know."\n\n'
                    + json.dumps(
                        {"question": question, "search_results": evidence},
                        ensure_ascii=False,
                    )
                ),
            },
        ],
    )
    return str(result["answer"]).strip()
