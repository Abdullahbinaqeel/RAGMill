# Migrating backends

Every store exports to and imports from a portable JSONL format, so you can move
data between SQLite and the cloud without re-embedding. Each line is one chunk
with its metadata, content, and embedding.

## Local SQLite → cloud (Pinecone/Qdrant)

```bash
# 1. Export from your current (SQLite) store
export RAGMILL_STORE_TYPE=sqlite
export RAGMILL_SQLITE_PATH=./ragmill.db
ragmill export ./backup.jsonl

# 2. Point the environment at the cloud store
export RAGMILL_STORE_TYPE=qdrant
export RAGMILL_QDRANT_URL=https://your-cluster.qdrant.io
export RAGMILL_QDRANT_API_KEY=your-key

# 3. Import into it
ragmill import ./backup.jsonl
```

The same three steps work in reverse (cloud → local) and between any two
backends — export from one, switch env, import into the other.

## From Python

```python
from ragmill.export import export_store, import_store
from ragmill.vector_store import store_from_config
from ragmill.config import RAGMillConfig

src = store_from_config(RAGMillConfig.from_env())   # current store
written = export_store("./backup.jsonl", src)
print(f"exported {written} records")

# ... change RAGMILL_STORE_TYPE + creds, then:
dst = store_from_config(RAGMillConfig.from_env())
imported = import_store("./backup.jsonl", dst)
print(f"imported {imported} records")
```

!!! tip "Embeddings travel with the data"
    Export includes the vectors, so import does **not** re-run the embedding
    model — migration is fast and produces identical vectors. This only holds if
    both ends use the same embedding model; don't mix models across a migration.

!!! note "Back up before you wipe"
    `ragmill export` is also the simplest backup: one JSONL file you can re-import
    any time.
