# RAGMill

[![PyPI](https://img.shields.io/pypi/v/ragmill.svg)](https://pypi.org/project/ragmill/)
[![CI](https://github.com/Abdullahbinaqeel/RAGMill/actions/workflows/ci.yml/badge.svg)](https://github.com/Abdullahbinaqeel/RAGMill/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/ragmill.svg)](https://pypi.org/project/ragmill/)

A lightweight, zero-config local pipeline engine for AI data ingestion,
semantic chunking, embeddings, vector search, and retrieval-augmented chat
— fully offline by default (no API keys, ever), with optional cloud
backends (Pinecone, Qdrant), a REST API, a standalone setup UI, and Docker
support.

## Install

```bash
pip install ragmill                          # core only (txt/md), zero dependencies
pip install ragmill[all]                     # everything (PDF, DOCX, embeddings, chat, server, cloud backends)
pip install ragmill[embeddings]              # + local ONNX embeddings
pip install ragmill[chat]                    # + local LLM for retrieval-augmented answers (no API key)
pip install ragmill[chat-gemini]             # + Gemini as the chat backend (needs GEMINI_API_KEY)
pip install ragmill[chat-openai]             # + ChatGPT as the chat backend (needs OPENAI_API_KEY)
pip install ragmill[pinecone]                # + Pinecone cloud backend
pip install ragmill[qdrant]                  # + Qdrant cloud backend
pip install ragmill[server]                  # + FastAPI REST API
pip install ragmill[config-ui]               # + standalone setup UI (writes .env)
```

## Quick start

### Ingest + chunk + embed + search (local SQLite)

> **Note:** This example requires the `embeddings` extra:
> `pip install ragmill[embeddings]`

```python
from ragmill import RAGEngine
from ragmill.embeddings import EmbeddingModel
from ragmill.vector_store import VectorStore

chunks = RAGEngine().execute_pipeline("./my_documents")

model = EmbeddingModel()
vectors = model.embed([c["content"] for c in chunks])

store = VectorStore("my_store.db")
store.add(chunks, vectors)

query = model.embed(["how does the overlap work?"])[0]
for r in store.search(query, top_k=3):
    print(r["score"], r["metadata"]["filename"], "->", r["content"][:80])
```

### Keep a store in sync with a folder

```python
from ragmill import RAGEngine
from ragmill.embeddings import EmbeddingModel
from ragmill.vector_store import VectorStore
from ragmill.sync import sync_directory

engine = RAGEngine()
model = EmbeddingModel()
store = VectorStore("my_store.db")

result = sync_directory("./my_documents", engine, model, store)
print(result)  # {"added": 2, "updated": 1, "skipped": 40, "deleted": 1}
```

## Ask questions, get grounded answers

Retrieval-augmented answer generation, with a choice of three backends —
selected via `RAGMILL_CHAT_BACKEND` (default `local`):

| Backend | Install | Needs a key? | Notes |
|---|---|---|---|
| `local` (default) | `ragmill[chat]` | No | Qwen2.5-1.5B-Instruct via `llama-cpp-python`. Downloads once (~1.1GB) to `~/.cache/ragmill/models`, then runs fully offline. |
| `gemini` | `ragmill[chat-gemini]` | `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) | Google's Gemini API. Best answer quality if you're online and have a key. |
| `openai` | `ragmill[chat-openai]` | `OPENAI_API_KEY` | OpenAI's Chat Completions API (ChatGPT). |

```bash
pip install ragmill[chat]           # local model (default, no key)
ragmill chat                        # interactive terminal Q&A over your ingested docs
```

```bash
# Switch to Gemini
pip install ragmill[chat-gemini]
export RAGMILL_CHAT_BACKEND=gemini
export GEMINI_API_KEY=xxxxxxxx
ragmill chat

# Or ChatGPT
pip install ragmill[chat-openai]
export RAGMILL_CHAT_BACKEND=openai
export OPENAI_API_KEY=xxxxxxxx
ragmill chat
```

```python
from ragmill.chat import generate_answer
answer = generate_answer("what does the overlap parameter do?", results)
```

All three backends share the same `generate_answer(query, chunks)` call — the
backend is picked at call time from `RAGMILL_CHAT_BACKEND`, so switching is
just an env var change, no code change. The [setup UI](#standalone-setup-ui)
below lets you pick a backend and enter its key without touching the
environment by hand.

Per-backend overrides:
- Local: `RAGMILL_CHAT_MODEL_REPO`, `RAGMILL_CHAT_MODEL_FILE`, `RAGMILL_CHAT_N_CTX`
- Gemini: `RAGMILL_GEMINI_MODEL` (default `gemini-flash-latest`)
- OpenAI: `RAGMILL_OPENAI_MODEL` (default `gpt-4o-mini`)

## Use a cloud vector store

Set environment variables and the store backend switches automatically:

```bash
export RAGMILL_STORE_TYPE=pinecone
export RAGMILL_PINECONE_API_KEY=xxxxxxxx
export RAGMILL_PINECONE_ENVIRONMENT=us-west1-gcp
export RAGMILL_PINECONE_INDEX_NAME=ragmill
```

```python
from ragmill.vector_store import store_from_config
from ragmill.config import RAGMillConfig

config = RAGMillConfig.from_env()
store = store_from_config(config)   # returns a PineconeVectorStore
```

Or for Qdrant (local via Docker, or a managed Qdrant Cloud cluster):

```bash
export RAGMILL_STORE_TYPE=qdrant
export RAGMILL_QDRANT_URL=http://localhost:6333          # or your cloud cluster URL
export RAGMILL_QDRANT_API_KEY=xxxxxxxx                    # required for Qdrant Cloud
export RAGMILL_QDRANT_COLLECTION_NAME=ragmill
```

Payload indexes (`filename`, `source_file`) are created automatically the
first time a collection is set up — Qdrant Cloud rejects filtered
search/delete/sync operations without them, while local/self-hosted Qdrant
is more lenient about it.

## Migrate between backends

Export your local SQLite store to JSONL, then import into a cloud store:

```bash
# 1. Export from SQLite
ragmill export ./backup.jsonl

# 2. Switch env to point at Pinecone
export RAGMILL_STORE_TYPE=pinecone
export RAGMILL_PINECONE_API_KEY=xxx

# 3. Import into Pinecone
ragmill import ./backup.jsonl
```

## REST API

```bash
pip install ragmill[server]
ragmill serve
# or: uvicorn ragmill.server:app --host 0.0.0.0 --port 8000
```

| Method | Path         | Description                             |
|--------|--------------|------------------------------------------|
| POST   | `/ingest`    | Ingest a directory                        |
| POST   | `/sync`      | Incremental sync                          |
| POST   | `/search`    | Search chunks                             |
| POST   | `/chat`      | Ask a question, get a grounded answer     |
| GET    | `/count`     | Number of stored chunks                   |
| POST   | `/export`    | Export store to JSONL                     |
| POST   | `/import`    | Import JSONL into store                   |
| GET    | `/health`    | Health check                              |
| GET    | `/`          | Minimal terminal-style chatbox (test `/chat` in a browser) |

## Standalone setup UI

A separate, minimal web UI for filling in optional config — cloud vector
store credentials, which chat backend to use (local/Gemini/ChatGPT) and its
key, or an override for the local chat model — without hand-editing
anything. It runs as its own server/process (a different port than
`ragmill serve`), so it's clearly a one-time setup tool independent of
wherever RAGMill actually runs as a dependency.

```bash
pip install ragmill[config-ui]
ragmill configure   # http://127.0.0.1:8090 by default — binds to localhost only
```

Fill in what you need and click "Save configuration" — it writes only the
fields you filled in to a local `.env` file (via `python-dotenv`, preserving
any unrelated lines already there). RAGMill loads that `.env` automatically
on the next run. Remember to add `.env` to `.gitignore` — nothing is ever
written into source code.

## Docker

```bash
# SQLite backend
docker compose --profile sqlite up

# Qdrant backend (spins up a Qdrant container too)
docker compose --profile qdrant up
```

## CLI

```bash
ragmill ingest ./docs       # Ingest + embed files
ragmill sync ./docs         # Incremental sync
ragmill search "query"      # Search
ragmill chat                # Interactive Q&A over stored chunks (local LLM)
ragmill count                # Chunk count
ragmill serve               # Start API
ragmill export ./out.jsonl  # Export
ragmill import ./in.jsonl   # Import
ragmill configure           # Standalone setup UI (writes .env)
```

## Configuration

All settings are controlled via environment variables (or a `.env` file —
see the setup UI above):

| Variable | Default | Description |
|---|---|---|
| `RAGMILL_STORE_TYPE` | `sqlite` | `sqlite`, `pinecone`, or `qdrant` |
| `RAGMILL_SQLITE_PATH` | `./ragmill.db` | Path to SQLite database file |
| `RAGMILL_EMBEDDING_MODEL` | `Xenova/all-MiniLM-L6-v2` | Hugging Face model for embeddings |
| `RAGMILL_EMBEDDING_DIM` | `384` | Embedding vector dimension |
| `RAGMILL_PINECONE_API_KEY` | — | Pinecone API key |
| `RAGMILL_PINECONE_ENVIRONMENT` | — | Pinecone environment |
| `RAGMILL_PINECONE_INDEX_NAME` | `ragmill` | Pinecone index name |
| `RAGMILL_QDRANT_URL` | — | Qdrant server/cluster URL |
| `RAGMILL_QDRANT_API_KEY` | — | Qdrant API key (required for Qdrant Cloud) |
| `RAGMILL_QDRANT_COLLECTION_NAME` | `ragmill` | Qdrant collection name |
| `RAGMILL_CHAT_BACKEND` | `local` | `local`, `gemini`, or `openai` |
| `RAGMILL_CHAT_MODEL_REPO` | `Qwen/Qwen2.5-1.5B-Instruct-GGUF` | Local chat model's Hugging Face repo |
| `RAGMILL_CHAT_MODEL_FILE` | `qwen2.5-1.5b-instruct-q4_k_m.gguf` | Local chat model's GGUF filename |
| `RAGMILL_CHAT_N_CTX` | `4096` | Local chat model's context window (tokens) |
| `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) | — | Required when `RAGMILL_CHAT_BACKEND=gemini` |
| `RAGMILL_GEMINI_MODEL` | `gemini-flash-latest` | Gemini model name |
| `OPENAI_API_KEY` | — | Required when `RAGMILL_CHAT_BACKEND=openai` |
| `RAGMILL_OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model name |
| `RAGMILL_CHUNK_SIZE` | `500` | Max chunk size in characters |
| `RAGMILL_OVERLAP` | `50` | Chunk overlap in characters |
| `RAGMILL_HOST` | `127.0.0.1` | Server bind address |
| `RAGMILL_PORT` | `8000` | Server port |

## Supported file types

`.txt`, `.md`, `.log`, `.rst`, `.pdf`, `.docx`

## Project structure

```
ragmill/
├── src/ragmill/
│   ├── __init__.py            # Public API exports
│   ├── engine.py              # RAGEngine: ingestion + chunking
│   ├── parsers.py             # PDF/DOCX text extractors
│   ├── embeddings.py          # Local ONNX embedding model
│   ├── chat.py                # Local LLM answer generation (llama-cpp-python)
│   ├── vector_store.py        # BaseVectorStore ABC + SQLiteVectorStore
│   ├── pinecone_store.py      # Pinecone backend (optional)
│   ├── qdrant_store.py        # Qdrant backend (optional)
│   ├── config.py              # RAGMillConfig: env/.env-based configuration
│   ├── sync.py                # Incremental directory sync
│   ├── export.py              # JSONL export/import for migration
│   ├── server.py              # FastAPI REST API + chat UI (optional)
│   ├── static/                # Terminal-style chatbox HTML for server.py's `/`
│   ├── config_ui.py           # Standalone setup UI, separate server (optional)
│   ├── config_ui_static/      # Setup UI's HTML form
│   └── __main__.py            # CLI entry point
├── tests/
├── Dockerfile
└── docker-compose.yml
```
