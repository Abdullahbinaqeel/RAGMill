#!/usr/bin/env bash
# Turnkey runner for RAGMill: sets up the venv, installs every extra
# (embeddings, PDF/DOCX, REST server, local-LLM chat), indexes a documents
# folder, and starts the server + browser chatbox.
#
# Usage:
#   ./run.sh [path-to-your-documents]
#
# If no path is given, or the path doesn't exist, a small sample docs
# folder is created so there's something to search/chat about immediately.
set -euo pipefail

cd "$(dirname "$0")"

VENV_DIR=".venv"
DOCS_DIR="${1:-$HOME/documents}"

# ── 1. Virtual environment ──────────────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
  echo "==> Creating virtual environment at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# ── 2. Install RAGMill with every extra (server + chat + parsers + embeddings)
echo "==> Installing RAGMill (this can take a while the first time, mainly for llama-cpp-python)"
pip install -q --upgrade pip
pip install -q -e ".[all]"

# ── 3. Documents folder ─────────────────────────────────────────────────────
if [ ! -d "$DOCS_DIR" ]; then
  echo "==> No folder found at '$DOCS_DIR' — creating a sample one"
  mkdir -p "$DOCS_DIR"
  cat > "$DOCS_DIR/welcome.txt" <<'EOF'
RAGMill is a lightweight, zero-config local pipeline for document ingestion,
semantic chunking, embeddings, and vector search. It runs fully offline by
default, with no API keys required, and supports optional cloud backends
(Pinecone, Qdrant) plus a REST API and browser chatbox.
EOF
fi

# ── 4. Config: local SQLite backend, default host/port ──────────────────────
export RAGMILL_STORE_TYPE=sqlite
export RAGMILL_SQLITE_PATH="${RAGMILL_SQLITE_PATH:-./ragmill.db}"
export RAGMILL_HOST="${RAGMILL_HOST:-127.0.0.1}"
export RAGMILL_PORT="${RAGMILL_PORT:-8000}"

# ── 5. Index the documents folder before the server starts ─────────────────
echo "==> Syncing '$DOCS_DIR' into $RAGMILL_SQLITE_PATH"
ragmill sync "$DOCS_DIR"

# ── 6. Start the REST API + browser chatbox ─────────────────────────────────
echo ""
echo "==> Starting RAGMill on http://localhost:${RAGMILL_PORT}"
echo "    Open that URL in a browser: 'you>' asks questions (chat), 'docs>' re-syncs a folder."
echo ""
ragmill serve
