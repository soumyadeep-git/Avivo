"""
Lightweight benchmark script for demoing retrieval quality against representative prompts.
"""

import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_engine import RAGEngine


SAMPLE_QUERIES = [
    "What are path parameters in FastAPI?",
    "How do I declare query parameters?",
    "When should I use async def in FastAPI?",
]


async def main() -> None:
    rag_engine = RAGEngine()
    results = []

    for query in SAMPLE_QUERIES:
        result = await rag_engine.query(user_id=0, query_text=query)
        results.append(
            {
                "query": query,
                "grounded": result.get("grounded", False),
                "cached": result.get("cached", False),
                "sources": result.get("sources", []),
                "answer_preview": (result.get("answer", "") or "")[:220],
            }
        )

    print(json.dumps(results, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    asyncio.run(main())
