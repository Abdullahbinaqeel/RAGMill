# Configuration

Everything is configured through **environment variables** (or a `.env` file in
the directory you run from). There is no config file format to learn and no
settings baked into code. `RAGMillConfig.from_env()` reads them and auto-loads a
`.env` via `python-dotenv` if present.

## Full reference

### Pipeline

| Variable | Default | Description |
|---|---|---|
| `RAGMILL_CHUNK_SIZE` | `500` | Max chunk size in characters |
| `RAGMILL_OVERLAP` | `50` | Chunk overlap in characters |
| `RAGMILL_EMBEDDING_MODEL` | `Xenova/all-MiniLM-L6-v2` | Embedding model (see caveats below) |

### Vector store

| Variable | Default | Description |
|---|---|---|
| `RAGMILL_STORE_TYPE` | `sqlite` | `sqlite`, `pinecone`, or `qdrant` |
| `RAGMILL_SQLITE_PATH` | `:memory:` | Path to the SQLite DB file (set this to persist!) |
| `RAGMILL_PINECONE_API_KEY` | — | Pinecone API key |
| `RAGMILL_PINECONE_ENVIRONMENT` | — | Pinecone environment |
| `RAGMILL_PINECONE_INDEX_NAME` | `ragmill` | Pinecone index name |
| `RAGMILL_QDRANT_URL` | — | Qdrant server/cluster URL |
| `RAGMILL_QDRANT_API_KEY` | — | Qdrant API key (required for Qdrant Cloud) |
| `RAGMILL_QDRANT_COLLECTION_NAME` | `ragmill` | Qdrant collection name |
| `RAGMILL_QDRANT_PREFER_GRPC` | `false` | Use gRPC instead of HTTP |

### Chat backend

| Variable | Default | Description |
|---|---|---|
| `RAGMILL_CHAT_BACKEND` | `local` | `local`, `gemini`, or `openai` |
| `RAGMILL_CHAT_MODEL_REPO` | `Qwen/Qwen2.5-1.5B-Instruct-GGUF` | Local model's HF repo |
| `RAGMILL_CHAT_MODEL_FILE` | `qwen2.5-1.5b-instruct-q4_k_m.gguf` | Local model's GGUF filename |
| `RAGMILL_CHAT_N_CTX` | `4096` | Local model context window (tokens) |
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | — | Required when backend is `gemini` |
| `RAGMILL_GEMINI_MODEL` | `gemini-flash-latest` | Gemini model name |
| `OPENAI_API_KEY` | — | Required when backend is `openai` |
| `RAGMILL_OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model name |

### Server

| Variable | Default | Description |
|---|---|---|
| `RAGMILL_HOST` | `0.0.0.0` | Server bind address |
| `RAGMILL_PORT` | `8000` | Server port |

## Using a `.env` file

Create `.env` in your working directory:

```ini
RAGMILL_SQLITE_PATH=./ragmill.db
RAGMILL_CHUNK_SIZE=800
RAGMILL_CHAT_BACKEND=openai
OPENAI_API_KEY=sk-...
```

It's loaded automatically on the next run.

!!! danger "Never commit secrets"
    Add `.env` to your `.gitignore`. API keys belong in the environment, never
    in source control. RAGMill never writes keys into code.

The [setup UI](cli.md) (`ragmill configure`) writes this `.env` for you through
a local-only web form, so you don't have to hand-edit it.

## Changing the models

### Chat model — fully swappable

Any GGUF model on Hugging Face works. Point the two env vars at it:

```bash
export RAGMILL_CHAT_MODEL_REPO="TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
export RAGMILL_CHAT_MODEL_FILE="mistral-7b-instruct-v0.2.Q4_K_M.gguf"
```

Or skip local models entirely and use a hosted backend — see
[Chat & answer generation](chat.md).

### Embedding model — swappable with constraints

`RAGMILL_EMBEDDING_MODEL` can point at another model, but two constraints apply:

1. **It must be a "Xenova"-style ONNX repo** — the loader expects the files
   `onnx/model_quantized.onnx` and `tokenizer.json` at the repo root. Examples
   that work: `Xenova/bge-small-en-v1.5`, `Xenova/gte-small`.
2. **It must be 384-dimensional.** The pipeline currently assumes MiniLM's
   384-dim output. A model with a different dimension (e.g. a 768-dim `bge-base`)
   needs a one-line change to `EMBEDDING_DIM` in `embeddings.py`.

!!! warning "Re-index after changing the embedding model"
    Vectors from different embedding models aren't comparable. If you switch the
    embedding model, wipe the store and re-index — don't mix vectors.
