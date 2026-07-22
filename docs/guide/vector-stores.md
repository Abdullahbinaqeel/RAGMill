# Vector stores

RAGMill talks to every backend through one interface, `BaseVectorStore`, so your
code doesn't change when you switch stores — only environment variables do. Pick
the store with `RAGMILL_STORE_TYPE` and `store_from_config()` returns the right
implementation.

| Backend | `RAGMILL_STORE_TYPE` | Best for |
|---|---|---|
| SQLite *(default)* | `sqlite` | local dev, single-folder knowledge bases, offline use |
| Qdrant | `qdrant` | self-hosted or managed cloud, larger corpora, a built-in vector dashboard |
| Pinecone | `pinecone` | fully managed cloud, scale |

```python
from ragmill.config import RAGMillConfig
from ragmill.vector_store import store_from_config

store = store_from_config(RAGMillConfig.from_env())   # type decided by env
```

## SQLite (local, default)

Zero setup. It's a single file (or in-memory) with brute-force dot-product
search — fast for thousands of chunks.

```bash
export RAGMILL_STORE_TYPE=sqlite
export RAGMILL_SQLITE_PATH=./ragmill.db     # omit → in-memory, not persisted
```

```python
from ragmill.vector_store import SQLiteVectorStore
store = SQLiteVectorStore("ragmill.db")
```

!!! note "Scaling limit"
    The SQLite search is O(n) per query — great to ~tens of thousands of chunks,
    not for millions. Past that, use Qdrant/Pinecone (approximate nearest
    neighbor) behind the same `search()` interface.

## Qdrant

```bash
pip install "ragmill[qdrant]"
export RAGMILL_STORE_TYPE=qdrant
export RAGMILL_QDRANT_URL=http://localhost:6333     # or your Qdrant Cloud URL
export RAGMILL_QDRANT_API_KEY=your-key              # required for Qdrant Cloud
export RAGMILL_QDRANT_COLLECTION_NAME=ragmill
```

Run Qdrant locally in one command:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

Payload indexes for `filename` and `source_file` are created automatically the
first time the collection is set up (Qdrant Cloud requires them for filtered
search/delete/sync).

!!! tip "Built-in visualization"
    Qdrant ships a web dashboard at `http://localhost:6333/dashboard` (or your
    cloud cluster's dashboard) that lists collections and visualizes your
    vectors — the easiest way to *see* your embedded data.

## Pinecone

```bash
pip install "ragmill[pinecone]"
export RAGMILL_STORE_TYPE=pinecone
export RAGMILL_PINECONE_API_KEY=your-key
export RAGMILL_PINECONE_ENVIRONMENT=us-west1-gcp
export RAGMILL_PINECONE_INDEX_NAME=ragmill
```

Manage and inspect indexes from the Pinecone console.

## Visualizing a local SQLite store

The SQLite store keeps text and metadata in a `chunks` table; embeddings are raw
`float32` BLOBs.

- **Browse text/metadata:** open `ragmill.db` in
  [DB Browser for SQLite](https://sqlitebrowser.org/). The embedding column is
  opaque there.
- **Visualize the vectors:** either migrate to Qdrant (below) and use its
  dashboard, or project the 384-dim vectors to 2-D with UMAP/t-SNE in a notebook
  and plot with Plotly. `store.scroll()` pages through every record *including*
  its embedding for exactly this.

To move a local store into a cloud one, see [Migrating backends](migration.md).
