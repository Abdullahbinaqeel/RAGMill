# Python API reference

The public surface of RAGMill. Import stage classes from their submodules —
`ragmill/__init__.py` intentionally exports only what's zero-dependency so
`import ragmill` never forces `numpy`/`onnxruntime` on core-only users.

```python
from ragmill import RAGEngine, SQLiteVectorStore, store_from_config, RAGMillConfig
from ragmill.embeddings import EmbeddingModel
from ragmill.sync import sync_directory
from ragmill.chat import generate_answer
from ragmill.export import export_store, import_store
```

---

## `RAGEngine`

`ragmill.engine.RAGEngine(chunk_size=500, overlap=50)`

Ingestion + chunking. `overlap` must be `< chunk_size` (raises `ValueError`).

| Method | Returns | Description |
|---|---|---|
| `stream_directory(path)` | generator of file manifests | Walk a folder, yielding `{source_path, filename, raw_content, modified_at}` per supported file |
| `semantic_chunking(text)` | `list[str]` | Split one document's text into overlapping chunks |
| `execute_pipeline(path)` | `list[dict]` | Ingest + chunk a folder into `{metadata, content}` payloads |

---

## `EmbeddingModel`

`ragmill.embeddings.EmbeddingModel(model_name="Xenova/all-MiniLM-L6-v2", cache_dir=None)`

Local ONNX embedder. Downloads the model to `~/.cache/ragmill/models` on first
construction; requires the `embeddings` extra.

| Method | Returns | Description |
|---|---|---|
| `embed(texts, batch_size=16)` | `np.ndarray (N, 384)` | L2-normalized vectors. Runs in memory-bounded, length-sorted sub-batches; output stays in input order |

```python
model = EmbeddingModel()
vecs = model.embed(["hello", "world"])   # shape (2, 384), float32
```

---

## Vector stores

All backends implement `ragmill.vector_store.BaseVectorStore`. The built-in one:

`ragmill.vector_store.SQLiteVectorStore(db_path=":memory:")` — also exported as
`VectorStore`.

| Method | Description |
|---|---|
| `add(payloads, embeddings)` | Insert chunks + their vectors |
| `search(query_embedding, top_k=5, filename=None, source_file=None, modified_after=None, modified_before=None)` | Ranked `[{score, metadata, content}]` |
| `count()` | Number of stored chunks |
| `delete_by_source(source_file)` | Remove one file's chunks + state |
| `delete_missing_sources(known_sources)` | Remove chunks for files not in the set (returns count) |
| `get_file_state(source_file)` / `upsert_file_state(...)` | Per-file hash tracking (used by sync) |
| `scroll(cursor=None, limit=100)` | Page through all records *including embeddings* |
| `batch()` | Context manager: defer commits to one flush |
| `close()` | Close the connection |

```python
store = SQLiteVectorStore("kb.db")
store.add(payloads, vectors)
results = store.search(query_vec, top_k=5, filename="report.pdf")
```

### `store_from_config`

`ragmill.vector_store.store_from_config(config) -> BaseVectorStore`

Returns a `SQLiteVectorStore`, `PineconeVectorStore`, or `QdrantVectorStore`
depending on `config.store_type`.

---

## `sync_directory`

`ragmill.sync.sync_directory(directory_path, engine, model, store, batch_size=64, progress=None) -> dict`

Incremental index of a folder. Returns `{"added", "updated", "skipped", "deleted"}`
(file counts). Pass `progress=lambda seen, path: ...` for a per-file callback.

```python
sync_directory("./docs", engine, model, store,
               progress=lambda n, f: print(n, f))
```

---

## `generate_answer`

`ragmill.chat.generate_answer(query, chunks) -> str`

Grounded RAG answer over the chunks returned by `store.search()`. Backend is
selected from `RAGMILL_CHAT_BACKEND` (`local` / `gemini` / `openai`). Requires
the matching chat extra.

---

## `RAGMillConfig`

`ragmill.config.RAGMillConfig` — a dataclass of every setting.
`RAGMillConfig.from_env()` builds one from environment variables (auto-loading a
`.env` if `python-dotenv` is installed). See [Configuration](guide/configuration.md).

---

## Export / import

`ragmill.export.export_store(path, store) -> int` and
`ragmill.export.import_store(path, store) -> int` move data to/from JSONL
(vectors included). See [Migrating backends](guide/migration.md).
