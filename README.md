# RAGMill

[![PyPI](https://img.shields.io/pypi/v/ragmill.svg)](https://pypi.org/project/ragmill/)
[![CI](https://github.com/Abdullahbinaqeel/RAGMill/actions/workflows/ci.yml/badge.svg)](https://github.com/Abdullahbinaqeel/RAGMill/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/ragmill.svg)](https://pypi.org/project/ragmill/)

A lightweight, zero-config local pipeline engine for AI data ingestion, semantic chunking, embeddings, and vector search.

## Install

```bash
pip install ragmill[all]   # includes PDF + DOCX + embeddings support
# or
pip install ragmill        # core only (txt/md), zero dependencies
```

Developing locally instead? Clone the repo and use an editable install:

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Usage

### Ingest + chunk

```python
from ragmill import RAGEngine

engine = RAGEngine(chunk_size=500, overlap=50)
chunks = engine.execute_pipeline("./my_documents")

for chunk in chunks:
    print(chunk["metadata"]["filename"], chunk["content"][:80])
```

Supports `.txt`, `.md`, `.log`, `.rst`, `.pdf`, and `.docx` out of the box.

### Embed + search locally

Requires the `embeddings` extra (`pip install -e ".[embeddings]"`). The model
(a quantized MiniLM ONNX export, ~23MB) downloads once to
`~/.cache/ragmill/models` and runs fully offline after that.

```python
from ragmill import RAGEngine
from ragmill.embeddings import EmbeddingModel
from ragmill.vector_store import VectorStore

chunks = RAGEngine().execute_pipeline("./my_documents")

model = EmbeddingModel()
vectors = model.embed([c["content"] for c in chunks])

store = VectorStore("my_documents.db")   # or VectorStore() for in-memory
store.add(chunks, vectors)

query_vector = model.embed(["how does the overlap window work?"])[0]
for result in store.search(query_vector, top_k=3):
    print(round(result["score"], 3), result["metadata"]["filename"], "->", result["content"][:80])
```

Filter a search to a specific file or a time window:

```python
store.search(query_vector, top_k=3, filename="report.pdf")
store.search(query_vector, top_k=3, modified_after=1704067200.0)  # since 2024-01-01
```

### Keep a store in sync with a folder

Re-embedding every file on every run is wasteful once a folder is large.
`sync_directory` tracks a content hash per file and only touches what
actually changed:

```python
from ragmill.sync import sync_directory

result = sync_directory("./my_documents", engine, model, store)
print(result)  # {"added": 2, "updated": 1, "skipped": 40, "deleted": 1}
```

Unchanged files are skipped without re-embedding. A changed file has its old
chunks replaced with new ones. A file removed from disk has its chunks
removed from the store on the next sync.
