"""
Ingestion script for the Mini-RAG System.
Reads files from the data directory, chunks them with light markdown awareness,
generates embeddings locally, and stores them in the configured vector backend.
"""

import hashlib
import os
from pathlib import Path
from typing import Dict, List
import uuid

from config import settings
from vector_store import IngestionChunk, get_vector_store

def split_sections(text: str) -> List[Dict[str, str]]:
    """Split markdown-ish text into sections using heading boundaries when possible."""
    stripped_text = text.strip()
    if not stripped_text:
        return []

    sections: List[Dict[str, str]] = []
    current_heading = "Introduction"
    current_lines: List[str] = []

    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            if current_lines:
                sections.append(
                    {
                        "heading": current_heading,
                        "content": "\n".join(current_lines).strip(),
                    }
                )
                current_lines = []
            current_heading = line.lstrip("#").strip() or "Untitled Section"
            continue

        current_lines.append(line)

    if current_lines:
        sections.append(
            {
                "heading": current_heading,
                "content": "\n".join(current_lines).strip(),
            }
        )

    return [section for section in sections if section["content"]]


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split long text into chunks while preferring paragraph and sentence boundaries."""
    chunks: List[str] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)

        if end < text_length:
            boundary_candidates = [
                text.rfind("\n\n", start, end),
                text.rfind("\n", start, end),
                text.rfind(". ", start, end),
                text.rfind(" ", start, end),
            ]
            best_boundary = max(boundary_candidates)
            if best_boundary > start + chunk_size // 2:
                end = best_boundary + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = max(end - overlap, start + 1)

    return chunks


def build_chunks(text: str, chunk_size: int, overlap: int) -> List[Dict[str, str]]:
    """Create metadata-rich chunks to improve retrieval and citations."""
    chunks: List[Dict[str, str]] = []

    for section in split_sections(text):
        section_heading = section["heading"]
        section_content = section["content"]
        for section_chunk_index, chunk in enumerate(
            chunk_text(section_content, chunk_size=chunk_size, overlap=overlap)
        ):
            chunks.append(
                {
                    "heading": section_heading,
                    "content": chunk,
                    "section_chunk_index": str(section_chunk_index),
                }
            )

    return chunks

def get_document_files(data_dir: str) -> List[str]:
    """Retrieves all markdown and text files from the data directory."""
    root = Path(data_dir)
    files = list(root.rglob("*.txt")) + list(root.rglob("*.md"))
    # Exclude instructional READMEs and opt-out files like `foo.skip.md`.
    return sorted(
        str(file_path)
        for file_path in files
        if str(file_path) != str(root / "README.md") and ".skip." not in file_path.name
    )


def main() -> None:
    """Main execution function for ingestion pipeline."""
    if not os.path.exists(settings.data_dir):
        print(
            f"Directory '{settings.data_dir}' not found. Please create it and add your documents."
        )
        return

    files = get_document_files(settings.data_dir)
    if not files:
        print(f"No documents found in '{settings.data_dir}' directory.")
        return

    print(f"Found {len(files)} files to ingest.")

    vector_store = get_vector_store()
    indexed_documents = 0
    skipped_documents = 0
    failed_documents = 0
    upserted_chunks = 0

    # Process each file, chunk, and prepare for insertion
    for file_path in files:
        print(f"Processing: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            relative_path = os.path.relpath(file_path, settings.data_dir)
            filename = os.path.basename(file_path)
            title = Path(file_path).stem.replace("-", " ").replace("_", " ").strip()
            document_fingerprint = hashlib.sha256(content.encode("utf-8")).hexdigest()
            existing_fingerprint = vector_store.get_document_fingerprint(relative_path)

            if existing_fingerprint == document_fingerprint:
                skipped_documents += 1
                print(f"Skipping unchanged file: {relative_path}")
                continue

            chunks = build_chunks(content, settings.chunk_size, settings.chunk_overlap)
            vector_store.delete_document_chunks(relative_path)

            point_chunks: List[IngestionChunk] = []

            for i, chunk in enumerate(chunks):
                chunk_text_value = chunk["content"]
                section_name = chunk["heading"]
                chunk_id_seed = f"{relative_path}:{section_name}:{i}:{chunk_text_value[:80]}"
                chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id_seed))

                point_chunks.append(
                    IngestionChunk(
                        chunk_id=chunk_id,
                        content=chunk_text_value,
                        metadata={
                            "source": filename,
                            "path": relative_path,
                            "title": title,
                            "section": section_name,
                            "chunk_index": i,
                            "section_chunk_index": chunk["section_chunk_index"],
                            "document_fingerprint": document_fingerprint,
                        },
                    )
                )
            upserted_chunks += vector_store.upsert_knowledge_chunks(point_chunks)
            indexed_documents += 1
        except Exception as e:
            failed_documents += 1
            print(f"Error processing {file_path}: {e}")

    print("Ingestion summary")
    print(f"- Indexed documents: {indexed_documents}")
    print(f"- Skipped unchanged documents: {skipped_documents}")
    print(f"- Failed documents: {failed_documents}")
    print(f"- Upserted chunks: {upserted_chunks}")

if __name__ == "__main__":
    main()
