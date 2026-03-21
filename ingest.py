"""
Ingestion script for the Mini-RAG System.
Reads files from the data directory, chunks them, generates embeddings locally,
and stores them in a local ChromaDB (SQLite) vector database.
"""

import os
import glob
from typing import List
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

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Splits text into chunks of `chunk_size` characters with `overlap`.
    Attempts to break chunks at natural boundaries like newlines or spaces.
    
    Args:
        text (str): The document text to chunk.
        chunk_size (int): Target maximum characters per chunk.
        overlap (int): Number of overlapping characters between chunks.
        
    Returns:
        List[str]: List of text chunks.
    """
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        
        # Natural break detection logic
        if end < text_length:
            last_newline = text.rfind('\n', start, end)
            last_space = text.rfind(' ', start, end)
            
            # Prefer newline, otherwise space, if they exist in the latter half of the chunk
            if last_newline != -1 and last_newline > start + chunk_size // 2:
                end = last_newline + 1
            elif last_space != -1 and last_space > start + chunk_size // 2:
                end = last_space + 1
                
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
            
        start = end - overlap
        
    return chunks

def get_document_files(data_dir: str) -> List[str]:
    """Retrieves all markdown and text files from the data directory."""
    files = []
    files.extend(glob.glob(f"{data_dir}/**/*.txt", recursive=True))
    files.extend(glob.glob(f"{data_dir}/**/*.md", recursive=True))
    # Exclude instructional READMEs
    return [f for f in files if not f.endswith("data/README.md")]

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

    # Initialize Local ChromaDB (System Design: SQLite backend)
    chroma_client = chromadb.PersistentClient(path=DB_DIR)
    
    # Initialize Local Embedding Model (Efficiency: Small footprint)
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    
    # Create or get collection
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=sentence_transformer_ef,
        metadata={"description": "Knowledge base documents"}
    )

    documents, metadatas, ids = [], [], []
    global_chunk_id = 0

    # Process each file, chunk, and prepare for insertion
    for file_path in files:
        print(f"Processing: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            chunks = chunk_text(content, CHUNK_SIZE, CHUNK_OVERLAP)
            filename = os.path.basename(file_path)
            
            for i, chunk in enumerate(chunks):
                documents.append(chunk)
                metadatas.append({"source": filename, "chunk_index": i})
                ids.append(f"doc_{global_chunk_id}")
                global_chunk_id += 1
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
