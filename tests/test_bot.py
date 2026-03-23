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

    assert "<b>Answer</b>" in message
    assert "<b>Evidence</b>" in message
    assert "Served from semantic cache" in message
    assert "<b>Sources</b>" in message
    assert "<b>Retrieved snippets</b>" in message


def test_build_answer_message_renders_fenced_code_as_pre_block() -> None:
    result = {
        "answer": "Use this example:\n```python\nfrom fastapi import FastAPI\napp = FastAPI()\n```",
        "sources": [],
        "source_snippets": [],
        "cached": False,
        "grounded": True,
    }

    message = build_answer_message(result)

    assert "<b>Python code</b>" in message
    assert "<pre>from fastapi import FastAPI\napp = FastAPI()</pre>" in message
