"""
Ingestion script for the Mini-RAG System.
Reads files from the data directory, chunks them with light markdown awareness,
generates embeddings locally, and stores them in a local ChromaDB database.
"""

import hashlib
import os
from pathlib import Path
from typing import Dict, List

import chromadb
from chromadb.utils import embedding_functions

from config import (
    DATA_DIR,
    DB_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

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
    # Exclude instructional READMEs
    return sorted(
        str(file_path) for file_path in files if str(file_path) != str(root / "README.md")
    )


def reset_collection(chroma_client: chromadb.PersistentClient) -> chromadb.Collection:
    """Recreate the KB collection so re-ingestion does not duplicate chunks."""
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    return chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=sentence_transformer_ef,
        metadata={"description": "Knowledge base documents"},
    )

def main() -> None:
    """Main execution function for ingestion pipeline."""
    if not os.path.exists(DATA_DIR):
        print(f"Directory '{DATA_DIR}' not found. Please create it and add your documents.")
        return

    os.makedirs(DB_DIR, exist_ok=True)

    files = get_document_files(DATA_DIR)
    if not files:
        print(f"No documents found in '{DATA_DIR}' directory.")
        return

    print(f"Found {len(files)} files to ingest.")

    chroma_client = chromadb.PersistentClient(path=DB_DIR)
    collection = reset_collection(chroma_client)

    documents, metadatas, ids = [], [], []

    # Process each file, chunk, and prepare for insertion
    for file_path in files:
        print(f"Processing: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            relative_path = os.path.relpath(file_path, DATA_DIR)
            filename = os.path.basename(file_path)
            title = Path(file_path).stem.replace("-", " ").replace("_", " ").strip()
            chunks = build_chunks(content, CHUNK_SIZE, CHUNK_OVERLAP)

            for i, chunk in enumerate(chunks):
                chunk_text_value = chunk["content"]
                section_name = chunk["heading"]
                chunk_id_seed = f"{relative_path}:{section_name}:{i}:{chunk_text_value[:80]}"
                chunk_id = hashlib.sha256(chunk_id_seed.encode("utf-8")).hexdigest()[:16]

                documents.append(chunk_text_value)
                metadatas.append(
                    {
                        "source": filename,
                        "path": relative_path,
                        "title": title,
                        "section": section_name,
                        "chunk_index": i,
                        "section_chunk_index": chunk["section_chunk_index"],
                    }
                )
                ids.append(f"doc_{chunk_id}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Insert into ChromaDB
    if documents:
        print(f"Adding {len(documents)} chunks to the vector database...")
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print("Ingestion complete! Knowledge base is ready.")
    else:
        print("No valid content to ingest.")

if __name__ == "__main__":
    main()
