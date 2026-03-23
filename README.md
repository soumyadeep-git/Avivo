# Avivo RAG Assistant

Avivo is a Telegram RAG assistant built for a hiring assignment. It answers questions over a document collection, shows grounded evidence, keeps short conversational memory, and can also describe images.

The repository is meant to be tested locally. The retrieval stack uses local `sentence-transformers/all-MiniLM-L6-v2`, the generation layer uses Groq, and the knowledge base is stored in Qdrant.

## Project overview

- Telegram interface with `/ask`, `/summarize`, and image input
- Local semantic retrieval with citation-style source output
- Groq-based answer generation and image understanding
- Idempotent ingestion with document fingerprinting
- Focused tests for config, ingestion, retrieval formatting, and answer flow

## Tech stack

| Area | Tooling |
| --- | --- |
| Interface | Telegram Bot API, `python-telegram-bot` |
| App runtime | Python, FastAPI |
| LLM | Groq `llama-3.1-8b-instant` |
| Vision model | Groq `llama-3.2-11b-vision-preview` |
| Retrieval embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | Qdrant |
| Testing | `pytest` |

## Dataset used

The main knowledge base used during development is the FastAPI documentation. The content was pulled from the official FastAPI GitHub repository and copied into `data/` as markdown files.

Because `data/` is ignored in git, someone cloning the repo will not automatically get the indexed documentation corpus. To reproduce the same dataset locally, run:

```bash
git clone --depth 1 https://github.com/fastapi/fastapi.git temp_fastapi
cp -r temp_fastapi/docs/en/docs/* data/
rm -rf temp_fastapi
```

After that, run ingestion again.

## High-level architecture

1. Telegram messages are received by the bot handlers in `bot.py`.
2. The query is passed to `rag_engine.py`.
3. The engine checks semantic cache first.
4. If needed, it retrieves the most relevant chunks from Qdrant using MiniLM embeddings.
5. Groq generates the final answer using the retrieved context.
6. The response is formatted with evidence, sources, snippets, and code blocks before being sent back to Telegram.

The indexing side works through `ingest.py`, which reads markdown/text files, splits them into sections and chunks, computes document fingerprints, and upserts the resulting vectors into Qdrant.

## How to set it up

1. Clone the repo

```bash
git clone <your-repo-url>
cd avivo
```

2. Create and activate the virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Create a local `.env`

Use your local `.env` with at least:

- `TELEGRAM_BOT_TOKEN`
- `GROQ_API_KEY`

Optional:

- `QDRANT_URL`
- `QDRANT_API_KEY`

If those Qdrant cloud values are left empty, the project can use the local embedded path at `db/qdrant`.

4. Prepare the dataset

Either place your own `.md` or `.txt` files into `data/`, or pull the FastAPI docs corpus using:

```bash
git clone --depth 1 https://github.com/fastapi/fastapi.git temp_fastapi
cp -r temp_fastapi/docs/en/docs/* data/
rm -rf temp_fastapi
```

5. Build the retrieval index

```bash
python ingest.py
```

6. Start the bot locally

```bash
python bot.py
```

## How to test it

Run the automated checks:

```bash
pytest
python scripts/evaluate.py
```

Then test the bot in Telegram with prompts like:

- `/start`
- `/ask What is FastAPI?`
- `/ask Explain FastAPI dependency injection with examples from the docs`
- `/ask When should I use async def in FastAPI?`
- upload one image and ask for a description

## Notes

- The intended review path is local execution, not hosted deployment
- The indexing step is idempotent, so re-running `python ingest.py` only reprocesses changed files
- The repository keeps both polling and webhook code paths, but local polling is the simplest way to evaluate the project
- If you want to use a different knowledge base, replace the files under `data/` and run ingestion again
