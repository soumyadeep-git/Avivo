from rag_engine import RAGEngine
from vector_store import RetrievedChunk


def test_build_retrieval_context_prefers_relevant_chunks() -> None:
    engine = RAGEngine.__new__(RAGEngine)
    retrieved = [
        RetrievedChunk(
            chunk_id="1",
            content="Path parameters are declared in the route path.",
            metadata={"path": "tutorial/path-params.md", "section": "Path Parameters"},
            score=0.91,
        ),
        RetrievedChunk(
            chunk_id="2",
            content="Low quality fallback chunk.",
            metadata={"path": "misc.md", "section": "Misc"},
            score=0.10,
        ),
    ]

    context = engine._build_retrieval_context(retrieved)

    assert context["has_relevant_context"] is True
    assert context["sources"][0] == "tutorial/path-params.md -> Path Parameters"
    assert "similarity=0.910" in context["context"]
