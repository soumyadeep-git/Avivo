# Avivo RAG Assistant

Avivo is a Telegram-based retrieval-augmented assistant built as a hiring assignment submission. It answers questions over a document collection, cites the retrieved evidence, keeps short conversational memory, and can also describe images with a vision model.

The current repository is optimized for a reviewer or evaluator who wants to run it locally with minimal friction. The retrieval stack uses local `sentence-transformers/all-MiniLM-L6-v2`, so there is no dependency on external embedding quotas during evaluation.

## Why this submission is strong

- Clean separation between ingestion, retrieval, bot interface, and runtime config
- Local semantic retrieval with source grounding and snippet previews
- Fast text generation with Groq and image understanding support
- Idempotent ingestion with document fingerprinting
- Lightweight test coverage for the most important paths
- Clear local setup and demo flow for evaluators

## What the bot does

- Answers document questions through `/ask`
- Shows whether the answer is grounded in retrieved material
- Returns source labels and short snippet evidence
- Keeps limited recent chat history for follow-up questions
- Supports `/summarize` for conversation summaries
- Accepts images and produces a caption plus tags

## Architecture at a glance

- `bot.py` handles Telegram commands and user interaction
- `rag_engine.py` handles retrieval, prompting, caching, and answer assembly
- `vector_store.py` stores document chunks and semantic cache entries in Qdrant
- `ingest.py` chunks documents, fingerprints them, and indexes them
- `config.py` centralizes runtime settings and validation
- `app.py` keeps webhook support available, although local polling is the simplest evaluation path

## Model choices

- Text generation: `llama-3.1-8b-instant` via Groq for low-latency responses
- Vision: `llama-3.2-11b-vision-preview` via Groq for image understanding
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` for strong local semantic search
- Vector store: Qdrant for simple local persistence and optional cloud portability

## Local setup

### 1. Clone the repo

```bash
git clone <your-repo-url>
cd avivo
```

### 2. Create the virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Create the environment file

Copy the example file and fill in your secrets:

```bash
cp .env.example .env
```

For local evaluation, the only required secrets are:

- `TELEGRAM_BOT_TOKEN`
- `GROQ_API_KEY`

You can leave `QDRANT_URL` and `QDRANT_API_KEY` empty if you want to use the local embedded Qdrant path.

### 4. Add or keep documents in `data/`

The project reads `.md` and `.txt` files from `data/`. A large FastAPI documentation corpus has already been used during development, but the system works with any similar text-based knowledge base.

### 5. Build the retrieval index

```bash
python ingest.py
```

This step is idempotent. Unchanged files are skipped on later runs.

### 6. Start the bot locally

```bash
python bot.py
```

This runs the bot in polling mode, which is the fastest way for an evaluator to try the system.

## Minimal local configuration

The default local flow is:

- `APP_ENV=development`
- `DEPLOYMENT_MODE=polling`
- local Qdrant path at `db/qdrant`
- local embeddings through MiniLM

That means no external deployment step is required to evaluate the assignment locally.

## Example demo flow

In Telegram:

1. Send `/start`
2. Ask `What is FastAPI?`
3. Ask `Explain FastAPI dependency injection with examples from the docs`
4. Ask `When should I use async def in FastAPI?`
5. Upload an image and ask for a description

## Example evaluator prompts

- `/ask What are path parameters in FastAPI?`
- `/ask How do I declare query parameters?`
- `/ask How should I structure a larger FastAPI app?`
- `/ask Explain FastAPI dependency injection with examples from the docs`
- `/ask Summarize how authentication works in FastAPI`

## Testing

Run the test suite with:

```bash
pytest
```

Run the lightweight retrieval evaluation script with:

```bash
python scripts/evaluate.py
```

## Repository structure

- `app.py` FastAPI app for webhook-style execution
- `bot.py` Telegram handlers and app factory
- `config.py` settings and runtime validation
- `ingest.py` document loading, chunking, and indexing
- `rag_engine.py` retrieval pipeline and answer generation
- `vector_store.py` Qdrant abstraction and embedding logic
- `logging_utils.py` structured logging helpers
- `tests/` focused regression coverage
- `docs/` supporting architecture and demo notes

## Engineering decisions

- I kept the retrieval pipeline modular so embedding, storage, and serving concerns are easy to reason about
- I used source-aware chunk metadata so answers can cite where they came from instead of returning generic text
- I added semantic caching to reduce repeated generation work for similar questions
- I kept conversation history intentionally short to reduce drift and keep responses grounded
- I preserved both polling and webhook execution paths, but local polling is the most practical review path

## Notes for reviewers

- The bot is intended to be run locally for evaluation
- The local path avoids deployment friction and hosted embedding limits
- If you re-run ingestion, the existing fingerprint logic prevents unnecessary re-indexing
- If you want to swap in your own documents, replace or extend the files under `data/` and run `python ingest.py` again
