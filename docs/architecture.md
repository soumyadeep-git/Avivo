# Architecture Notes

## Request flow

1. Telegram sends a webhook update to the FastAPI application.
2. FastAPI converts the payload to a Telegram `Update`.
3. The Telegram application routes the update to the correct handler.
4. The handler calls `RAGEngine`.
5. `RAGEngine` checks semantic cache first, then queries the knowledge collection.
6. Groq generates the final answer or image description.
7. The response is formatted and sent back to Telegram.

## Indexing flow

1. `ingest.py` walks the `data/` directory.
2. Files are split into sections and chunks.
3. A document fingerprint is computed.
4. Existing chunks for changed files are replaced in the vector store.
5. New embeddings are generated and upserted into Qdrant.

## Key production decisions

- Webhooks instead of polling for deployability
- Cloud-capable vector store instead of local-only persistence
- Structured config validation at startup
- Health endpoints for runtime visibility
- Lightweight test coverage plus a benchmark script for demos
