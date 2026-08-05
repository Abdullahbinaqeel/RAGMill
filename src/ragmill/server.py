"""
FastAPI server for deploying RAGMill as a REST API.

Start with:
    uv pip install ragmill[server]
    ragmill serve

Or directly:
    uvicorn ragmill.server:app --host 127.0.0.1 --port 8000

Endpoints:
    POST   /ingest       Ingest a directory (full pipeline)
    POST   /sync         Incremental sync directory → store
    POST   /search       Search for similar chunks
    GET    /count        Number of stored chunks
    POST   /export       Export store to JSONL
    POST   /import       Import JSONL into store
    POST   /chat         Ask a question, get a grounded natural-language answer
    GET    /             Minimal terminal-style chat UI for testing
    GET    /health       Health check
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, UploadFile, File, Security
from fastapi.responses import HTMLResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

from ragmill import RAGEngine
from ragmill.chat import generate_answer
from ragmill.config import RAGMillConfig
from ragmill.embeddings import EmbeddingModel
from ragmill.export import export_store, import_store
from ragmill.sync import sync_directory
from ragmill.vector_store import store_from_config

logger = logging.getLogger(__name__)

app = FastAPI(title="RAGMill", version="0.4.3")

STATIC_DIR = Path(__file__).parent / "static"

config: RAGMillConfig = RAGMillConfig.from_env()
engine = RAGEngine(chunk_size=config.chunk_size, overlap=config.overlap)
_model: Optional[EmbeddingModel] = None
_model_lock: Any = None
store = store_from_config(config)

# Allowed roots for ingest/sync — if set, paths outside these are rejected.
_allowed_roots: Optional[List[Path]] = None
if config.server_allowed_roots:
    _allowed_roots = [Path(r).resolve() for r in config.server_allowed_roots.split(":")]

# API key auth — if server_api_key is set, all endpoints (except /health)
# require the key via Authorization: Bearer <key> or X-API-Key header.
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _get_model() -> EmbeddingModel:
    global _model, _model_lock
    if _model_lock is None:
        import threading

        _model_lock = threading.Lock()
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = EmbeddingModel(model_name=config.embedding_model)
    return _model


def _verify_api_key(api_key: Optional[str] = Security(_api_key_header)):
    if config.server_api_key and api_key != config.server_api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def _validate_directory(path: str) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_dir():
        raise HTTPException(404, f"directory not found: {path}")
    if _allowed_roots:
        try:
            resolved.relative_to(*_allowed_roots)
        except ValueError:
            raise HTTPException(
                403,
                f"directory is outside the allowed roots ({config.server_allowed_roots}): {path}",
            )
    return resolved


# ── Request / response schemas ─────────────────────────────────────────────


class IngestRequest(BaseModel):
    directory: str


class IngestResponse(BaseModel):
    chunks: int


class SyncRequest(BaseModel):
    directory: str


class SyncResponse(BaseModel):
    added: int = 0
    updated: int = 0
    skipped: int = 0
    deleted: int = 0


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    filename: Optional[str] = None
    source_file: Optional[str] = None
    modified_after: Optional[float] = None
    modified_before: Optional[float] = None


class SearchResult(BaseModel):
    score: float
    metadata: Dict[str, Any]
    content: str


class SearchResponse(BaseModel):
    results: List[SearchResult]


class CountResponse(BaseModel):
    count: int


class ChatRequest(BaseModel):
    query: str
    top_k: int = 5
    filename: Optional[str] = None
    source_file: Optional[str] = None
    modified_after: Optional[float] = None
    modified_before: Optional[float] = None


class ChatSource(BaseModel):
    score: float
    filename: str
    chunk_index: int
    content: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[ChatSource]


class HealthResponse(BaseModel):
    status: str
    store_type: str
    chunk_count: int


# ── Endpoints ──────────────────────────────────────────────────────────────


@app.post("/ingest", response_model=IngestResponse, dependencies=[Depends(_verify_api_key)])
def ingest(request: IngestRequest):
    directory = _validate_directory(request.directory)
    chunks = engine.execute_pipeline(str(directory))
    if chunks:
        model = _get_model()
        from ragmill.embeddings import DEFAULT_EMBED_BATCH

        for i in range(0, len(chunks), DEFAULT_EMBED_BATCH):
            batch = chunks[i : i + DEFAULT_EMBED_BATCH]
            vectors = model.embed([c["content"] for c in batch])
            store.add(batch, vectors)
    return IngestResponse(chunks=len(chunks))


@app.post("/sync", response_model=SyncResponse, dependencies=[Depends(_verify_api_key)])
def sync(request: SyncRequest):
    directory = _validate_directory(request.directory)
    model = _get_model()
    result = sync_directory(str(directory), engine, model, store)
    return SyncResponse(**result)


@app.post("/search", response_model=SearchResponse, dependencies=[Depends(_verify_api_key)])
def search(request: SearchRequest):
    model = _get_model()
    query_vector = model.embed([request.query])[0]
    results = store.search(
        query_embedding=query_vector,
        top_k=request.top_k,
        filename=request.filename,
        source_file=request.source_file,
        modified_after=request.modified_after,
        modified_before=request.modified_before,
    )
    search_results = [SearchResult(**r) for r in results]
    return SearchResponse(results=search_results)


@app.get("/count", response_model=CountResponse)
def count():
    return CountResponse(count=store.count())


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(_verify_api_key)])
def chat(request: ChatRequest):
    try:
        model = _get_model()
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="Chat requires an embedding model. Install it with: pip install ragmill[embeddings]",
        )
    query_vector = model.embed([request.query])[0]
    results = store.search(
        query_embedding=query_vector,
        top_k=request.top_k,
        filename=request.filename,
        source_file=request.source_file,
        modified_after=request.modified_after,
        modified_before=request.modified_before,
    )
    try:
        answer = generate_answer(request.query, results, config)
    except ImportError as exc:
        raise HTTPException(status_code=501, detail=str(exc))
    sources = [
        ChatSource(
            score=r["score"],
            filename=r["metadata"].get("filename", "unknown"),
            chunk_index=r["metadata"].get("chunk_index", -1),
            content=r["content"],
        )
        for r in results
    ]
    return ChatResponse(answer=answer, sources=sources)


@app.get("/", response_class=HTMLResponse)
def chat_ui():
    return (STATIC_DIR / "index.html").read_text()


@app.post("/export", dependencies=[Depends(_verify_api_key)])
def export():
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    try:
        written = export_store(path, store)
        from fastapi.responses import FileResponse
        from starlette.background import BackgroundTask

        return FileResponse(
            path,
            media_type="application/x-ndjson",
            filename="ragmill_export.jsonl",
            background=BackgroundTask(os.unlink, path),
        )
    except Exception as exc:
        os.unlink(path)
        raise HTTPException(500, str(exc))


@app.post("/import", dependencies=[Depends(_verify_api_key)])
def import_jsonl(file: UploadFile = File(...)):
    import shutil

    tmp = Path(tempfile.mkstemp(suffix=".jsonl")[1])
    try:
        with open(tmp, "wb") as out_f:
            shutil.copyfileobj(file.file, out_f)
        imported = import_store(str(tmp), store)
        return {"imported": imported}
    finally:
        tmp.unlink(missing_ok=True)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        store_type=config.store_type,
        chunk_count=store.count(),
    )
