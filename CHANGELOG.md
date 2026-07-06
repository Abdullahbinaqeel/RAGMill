# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.2.0]

### Added
- `EmbeddingModel` (`ragmill.embeddings`) — local ONNX-based sentence embeddings via a quantized MiniLM model, downloaded once and cached offline.
- `VectorStore` (`ragmill.vector_store`) — SQLite-backed storage with brute-force cosine similarity search, plus filtering by `filename`, `source_file`, `modified_after`, `modified_before`.
- `sync_directory` (`ragmill.sync`) — incremental sync between a folder and a `VectorStore`: skips unchanged files (content-hash based), replaces chunks for changed files, removes chunks for deleted files.
- `modified_at` is now captured during ingestion and threaded through chunk metadata.

### Changed
- Project renamed to **RAGMill**. The original name, `nexus-flow`, was already taken on PyPI by an unrelated package; the next candidate, `nexusflow`, was rejected by PyPI for being confusingly similar to it. Package name is now `ragmill`, and the import path changed accordingly: `import ragmill` (previously `import nexus_flow`). The main class was renamed `NexusEngine` → `RAGEngine` to match.

## [0.1.0]

### Added
- `RAGEngine` — directory ingestion (`.txt`, `.md`, `.log`, `.rst`, `.pdf` via `pypdf`, `.docx` via `python-docx`) and semantic chunking with paragraph/sentence-boundary splitting and configurable overlap.
- Optional extras (`pdf`, `docx`, `all`, `dev`) so the core package has zero hard dependencies.
