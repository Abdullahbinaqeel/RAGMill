# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.4.2] - 2026-08-06

### Fixed
- Text files are now decoded by their actual encoding instead of being read as UTF-8 with `errors="ignore"`. The old behaviour corrupted files silently, which mattered most on Windows because it is what Notepad writes:
  - **"Unicode" (UTF-16)** decoded as UTF-8 produced NUL-interleaved mojibake (`T\x00h\x00e\x00…`). The file ingested "successfully", but its chunks embedded as noise and never matched a query — a `.txt` file that appeared not to work at all.
  - **"ANSI" (cp1252)** lost every non-ASCII byte, turning `costs £50` into `costs 50` — a silent change of meaning rather than a visible failure.
  - **UTF-8 with BOM** left a stray `﻿` at the head of the first chunk.

  Byte-order marks are now honoured (UTF-8/16/32, and the BOM is stripped rather than left in the text), BOM-less UTF-16 is detected via interior NULs, and non-UTF-8 files fall back to cp1252 then latin-1 **with a warning** naming the encoding used. This applies to every text reader — plain text (`.txt`, `.md`, `.log`, `.rst`), `.csv`/`.tsv`, `.html`, and `.rtf` — all four of which had the same flaw.

## [0.4.1] - 2026-08-05

### Fixed
- Errors for missing system binaries now name a command for the user's own OS. `tesseract` and poppler's `pdftoppm` cannot come from pip, and the messages previously suggested `brew install` on every platform — useless on Windows and Linux. The scanned-PDF error also mentions `enable_ocr=False` for opting out rather than only how to opt in.
- The "local chat backend unavailable" error no longer points solely at `pip install "ragmill[chat]"`, which is the command that fails on Windows. It now leads with the prebuilt-wheel index, explains why the package is not in `[all]`, and offers the hosted Gemini/OpenAI backends that need no local model.
- `pip install ragmill[all]` no longer fails on a clean machine. The `all` (and `dev`) extras pulled `llama-cpp-python`, which ships no PyPI wheels for recent versions, so pip fell back to a 70MB+ sdist that vendors llama.cpp — that needs a C++ toolchain, and on Windows the vendored tree exceeds the 260-char `MAX_PATH` limit, aborting the whole install with `OSError: [Errno 2] No such file or directory`. The local LLM is now opt-in: install a prebuilt wheel with `pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu` (no compiler needed), or use the hosted `chat-gemini` / `chat-openai` backends, which are included in `[all]`. `pip install "ragmill[chat]"` still works if you have CMake and a C++ toolchain — and, on Windows, long paths enabled.
- Source distributions are built from an explicit allowlist. The sdist previously included every file `.gitignore` did not exclude, so a locally built tarball could sweep in a maintainer's untracked working directories. Releases built by CI were unaffected.

## [0.4.0] - 2026-07-24

### Added
- Extended file-format support: `.csv`/`.tsv` (stdlib, no extra), plus `.html`/`.htm`, `.rtf`, `.xlsx`, and `.pptx` via the new `office` extra (`pip install ragmill[office]`).
- OCR support via the new `ocr` extra (`pip install ragmill[ocr]`): text extraction from images (`.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp`, `.gif`) and automatic fallback to OCR for scanned/image-only PDFs (requires the system `tesseract` binary, and `pdftoppm`/poppler for PDFs).
- DOCX extraction now also captures table cell text, not just paragraphs.

### Changed
- Chat answers now lead with a direct answer followed by a brief 2–4 sentence explanation, and no longer embed bracketed citation markers (`[1]`, `[report.pdf]`) — sources are listed separately by the caller.
- The ingestion engine now skips files that yield no extractable text (e.g. scanned PDFs with no OCR result) with a warning instead of storing an empty document.

## [0.3.2] - 2026-07-22

### Fixed
- CLI no longer requires numpy just to start: `ragmill --version` and `ragmill --help` now work on a core-only install (heavy imports are deferred into the commands that need them).
- Data commands that need numpy (`ingest`, `sync`, `search`, `count`, `export`, `import`) now fail with a clear "install `ragmill[embeddings]`" message instead of a raw `ModuleNotFoundError: numpy`.

## [0.3.1] - 2026-07-22

### Added
- `ragmill --version` prints the installed version, and `ragmill.__version__` exposes it programmatically.

## [0.3.0] - 2026-07-22

### Added
- CLI entry point (`ragmill` command) with subcommands: `ingest`, `sync`, `search`, `chat`, `count`, `serve`, `export`, `import`, `configure`.
- REST API server (`ragmill serve`) with FastAPI — endpoints for ingest, sync, search, chat, export, import, count, health.
- Retrieval-augmented chat via three backends: local GGUF model (`ragmill[chat]`), Gemini (`ragmill[chat-gemini]`), OpenAI (`ragmill[chat-openai]`).
- Standalone config UI (`ragmill configure`) for setting up cloud backends and chat keys without editing `.env` by hand.
- Pinecone cloud vector store backend (`ragmill[pinecone]`).
- Qdrant cloud vector store backend (`ragmill[qdrant]`).
- JSONL export/import for backup and cross-backend migration.
- `RAGMillConfig` centralizes all settings in a single dataclass, loaded from env vars / `.env`.
- Configurable embedding dimension (`RAGMILL_EMBEDDING_DIM`).
- Batched embedding and upsert operations for better throughput.
- Docker support with `docker-compose.yml` (SQLite and Qdrant profiles).

### Changed
- Default SQLite path is now `./ragmill.db` (was `:memory:`) — data persists across CLI invocations.
- Server binds to `127.0.0.1` by default (was `0.0.0.0`) — not exposed to the network unless explicitly configured.
- `config-ui` extra now includes FastAPI, uvicorn, pydantic, and numpy so `ragmill configure` works out of the box.
- `server` extra now includes `python-dotenv` so `.env` files are loaded automatically.
- Pinecone `RAGMILL_PINECONE_ENVIRONMENT` is now honored (parses region from formats like `us-west-2`, `us-west1-gcp`).

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
