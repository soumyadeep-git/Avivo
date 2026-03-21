from bot import build_answer_message


def test_build_answer_message_includes_evidence_sources_and_snippets() -> None:
    result = {
        "answer": "Path parameters are parts of the URL path.",
        "sources": ["tutorial/path-params.md -> Path Parameters"],
        "source_snippets": ["tutorial/path-params.md -> Path Parameters: Path params are values captured from the URL."],
        "cached": True,
        "grounded": True,
    }

    message = build_answer_message(result)

    assert "Answer" in message
    assert "Evidence:" in message
    assert "Served from semantic cache" in message
    assert "Sources:" in message
    assert "Retrieved snippets:" in message
