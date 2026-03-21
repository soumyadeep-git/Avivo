from ingest import build_chunks, chunk_text, split_sections


def test_split_sections_uses_markdown_headings() -> None:
    text = "# Intro\nHello\n## Details\nMore text"
    sections = split_sections(text)

    assert sections[0]["heading"] == "Intro"
    assert sections[1]["heading"] == "Details"


def test_chunk_text_prefers_natural_boundaries() -> None:
    text = "Paragraph one.\n\nParagraph two has more content for chunking."
    chunks = chunk_text(text, chunk_size=20, overlap=5)

    assert len(chunks) >= 2
    assert all(chunk.strip() for chunk in chunks)


def test_build_chunks_preserves_section_metadata() -> None:
    text = "# API\nLine one.\nLine two.\n# FAQ\nLine three."
    chunks = build_chunks(text, chunk_size=50, overlap=10)

    assert len(chunks) == 2
    assert chunks[0]["heading"] == "API"
    assert chunks[1]["heading"] == "FAQ"
