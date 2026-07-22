"""
Abstract vector store interface + SQLite-backed implementation.

BaseVectorStore defines the contract every store backend must satisfy.
SQLiteVectorStore is the built-in local implementation that ships
with ragmill — it uses brute-force dot-product search over all rows.

The plain name VectorStore is kept as an alias for SQLiteVectorStore
so existing code (from ragmill.vector_store import VectorStore) keeps working.

Usage:
    store = SQLiteVectorStore("my_store.db")
    store.add(payloads, embeddings)
    results = store.search(query_vector, top_k=5)

    # or from config:
    store = store_from_config(config)
"""

import sqlite3
import threading
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ragmill.config import RAGMillConfig

__all__ = [
    "BaseVectorStore",
    "SQLiteVectorStore",
    "VectorStore",
    "store_from_config",
]


# ── Abstract interface ──────────────────────────────────────────────────────


class BaseVectorStore(ABC):
    """Abstract interface for all vector store backends."""

    @abstractmethod
    def add(self, payloads: List[Dict[str, Any]], embeddings: np.ndarray) -> None: ...

    @abstractmethod
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filename: Optional[str] = None,
        source_file: Optional[str] = None,
        modified_after: Optional[float] = None,
        modified_before: Optional[float] = None,
    ) -> List[Dict[str, Any]]: ...

    @abstractmethod
    def count(self) -> int: ...

    @abstractmethod
    def delete_by_source(self, source_file: str) -> None: ...

    @abstractmethod
    def delete_missing_sources(self, known_sources: set) -> int: ...

    @abstractmethod
    def get_file_state(self, source_file: str) -> Optional[Dict[str, Any]]: ...

    @abstractmethod
    def upsert_file_state(
        self, source_file: str, content_hash: str, modified_at: Optional[float]
    ) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def scroll(
        self, cursor: Optional[Any] = None, limit: int = 100
    ) -> Tuple[List[Dict[str, Any]], Optional[Any]]:
        """
        Page through every chunk in the store, embeddings included.

        Pass the previously returned cursor back in to fetch the next page.
        Returns (records, next_cursor) — next_cursor is None once exhausted.
        Each record is {"metadata": {...}, "content": str, "embedding": np.ndarray}.
        """
        ...

    @contextmanager
    def batch(self):
        """
        Defer backend-specific per-write overhead (e.g. SQLite fsync'd commits)
        until this block exits, then flush once instead of after every call.
        Backends with no such per-write cost (Qdrant, Pinecone — their
        upsert() calls are already single network round-trips) just inherit
        this no-op default.
        """
        yield self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ── SQLite implementation ───────────────────────────────────────────────────


class SQLiteVectorStore(BaseVectorStore):
    def __init__(self, db_path: Union[str, Path] = ":memory:"):
        # check_same_thread=False + a lock: needed because server.py runs sync
        # endpoints in FastAPI's worker thread pool, not the thread that
        # created this store.
        self._lock = threading.RLock()
        self._local = threading.local()
        self.connection = sqlite3.connect(str(db_path), check_same_thread=False)
        # WAL + synchronous=NORMAL is the standard safe/fast combo recommended by
        # SQLite itself: still crash-consistent (no corruption), it just means an
        # OS crash/power loss (not an app crash) could lose the most recent commit.
        # No effect on :memory: databases (WAL is a no-op there).
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_file TEXT,
                filename TEXT,
                chunk_index INTEGER,
                content TEXT NOT NULL,
                modified_at REAL,
                embedding BLOB NOT NULL
            )
            """)
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS file_state (
                source_file TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL,
                modified_at REAL
            )
            """)
        # Without this, delete_by_source() / delete_missing_sources() do a full
        # table scan per file — fine at small scale, but scales badly once
        # `chunks` holds tens of thousands of rows (each stale-file cleanup
        # becomes an O(rows) scan). CREATE INDEX IF NOT EXISTS is safe to run
        # against an existing populated database too.
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_source_file ON chunks(source_file)"
        )
        self.connection.commit()

    def _get_batch_depth(self) -> int:
        return getattr(self._local, "batch_depth", 0)

    def _set_batch_depth(self, val: int) -> None:
        self._local.batch_depth = val

    def _maybe_commit(self) -> None:
        """Commit unless inside a `with store.batch():` block, which defers to its own commit."""
        if self._get_batch_depth() == 0:
            self.connection.commit()

    @contextmanager
    def batch(self):
        with self._lock:
            self._set_batch_depth(self._get_batch_depth() + 1)
        try:
            yield self
        except BaseException:
            with self._lock:
                self.connection.rollback()
                self._set_batch_depth(self._get_batch_depth() - 1)
            raise
        else:
            with self._lock:
                depth = self._get_batch_depth() - 1
                self._set_batch_depth(depth)
                if depth == 0:
                    self.connection.commit()

    def add(self, payloads: List[Dict[str, Any]], embeddings: np.ndarray) -> None:
        if len(payloads) != len(embeddings):
            raise ValueError("payloads and embeddings must be the same length")

        rows = [
            (
                payload["metadata"]["source_file"],
                payload["metadata"]["filename"],
                payload["metadata"]["chunk_index"],
                payload["content"],
                payload["metadata"].get("modified_at"),
                np.asarray(vector, dtype=np.float32).tobytes(),
            )
            for payload, vector in zip(payloads, embeddings)
        ]
        with self._lock:
            self.connection.executemany(
                """
                INSERT INTO chunks (source_file, filename, chunk_index, content, modified_at, embedding)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            self._maybe_commit()

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filename: Optional[str] = None,
        source_file: Optional[str] = None,
        modified_after: Optional[float] = None,
        modified_before: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        query_vector = np.asarray(query_embedding, dtype=np.float32)

        clauses = []
        params: List[Any] = []
        if filename is not None:
            clauses.append("filename = ?")
            params.append(filename)
        if source_file is not None:
            clauses.append("source_file = ?")
            params.append(source_file)
        if modified_after is not None:
            clauses.append("modified_at >= ?")
            params.append(modified_after)
        if modified_before is not None:
            clauses.append("modified_at <= ?")
            params.append(modified_before)

        query = (
            "SELECT source_file, filename, chunk_index, content, modified_at, embedding FROM chunks"
        )
        if clauses:
            query += " WHERE " + " AND ".join(clauses)

        with self._lock:
            rows = self.connection.execute(query, params).fetchall()
        if not rows:
            return []

        vectors = np.stack([np.frombuffer(row[5], dtype=np.float32) for row in rows])
        scores = vectors @ query_vector

        if len(scores) <= top_k:
            top_indices = np.argsort(-scores)
        else:
            top_indices = np.argpartition(-scores, top_k)[:top_k]
        return [
            {
                "score": float(scores[i]),
                "metadata": {
                    "source_file": rows[i][0],
                    "filename": rows[i][1],
                    "chunk_index": rows[i][2],
                    "modified_at": rows[i][4],
                },
                "content": rows[i][3],
            }
            for i in top_indices
        ]

    def count(self) -> int:
        with self._lock:
            return self.connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def delete_by_source(self, source_file: str) -> None:
        with self._lock:
            self.connection.execute("DELETE FROM chunks WHERE source_file = ?", (source_file,))
            self.connection.execute("DELETE FROM file_state WHERE source_file = ?", (source_file,))
            self._maybe_commit()

    def delete_missing_sources(self, known_sources: set) -> int:
        with self._lock:
            rows = self.connection.execute("SELECT source_file FROM file_state").fetchall()
        stale = [row[0] for row in rows if row[0] not in known_sources]

        # Bulk-delete in chunks (instead of one delete_by_source() call per stale
        # file) so cleaning up thousands of removed/renamed files stays a handful
        # of statements instead of thousands of individual round trips.
        CHUNK = 500
        with self._lock:
            for i in range(0, len(stale), CHUNK):
                batch = stale[i : i + CHUNK]
                placeholders = ",".join("?" * len(batch))
                self.connection.execute(
                    f"DELETE FROM chunks WHERE source_file IN ({placeholders})", batch
                )
                self.connection.execute(
                    f"DELETE FROM file_state WHERE source_file IN ({placeholders})", batch
                )
            if stale:
                self._maybe_commit()
        return len(stale)

    def get_file_state(self, source_file: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self.connection.execute(
                "SELECT content_hash, modified_at FROM file_state WHERE source_file = ?",
                (source_file,),
            ).fetchone()
        if row is None:
            return None
        return {"content_hash": row[0], "modified_at": row[1]}

    def upsert_file_state(
        self, source_file: str, content_hash: str, modified_at: Optional[float]
    ) -> None:
        with self._lock:
            self.connection.execute(
                """
                INSERT INTO file_state (source_file, content_hash, modified_at)
                VALUES (?, ?, ?)
                ON CONFLICT(source_file) DO UPDATE SET content_hash = excluded.content_hash, modified_at = excluded.modified_at
                """,
                (source_file, content_hash, modified_at),
            )
            self._maybe_commit()

    def scroll(
        self, cursor: Optional[int] = None, limit: int = 100
    ) -> Tuple[List[Dict[str, Any]], Optional[int]]:
        last_id = cursor or 0
        with self._lock:
            rows = self.connection.execute(
                """
                SELECT id, source_file, filename, chunk_index, content, modified_at, embedding
                FROM chunks WHERE id > ? ORDER BY id LIMIT ?
                """,
                (last_id, limit),
            ).fetchall()
        if not rows:
            return [], None

        records = [
            {
                "metadata": {
                    "source_file": row[1],
                    "filename": row[2],
                    "chunk_index": row[3],
                    "modified_at": row[5],
                },
                "content": row[4],
                "embedding": np.frombuffer(row[6], dtype=np.float32).copy(),
            }
            for row in rows
        ]
        next_cursor = rows[-1][0] if len(rows) == limit else None
        return records, next_cursor

    def close(self) -> None:
        with self._lock:
            self.connection.close()


# Backward-compatible alias
VectorStore = SQLiteVectorStore


# ── Factory ──────────────────────────────────────────────────────────────────


def store_from_config(config: RAGMillConfig) -> BaseVectorStore:
    if config.store_type == "sqlite":
        return SQLiteVectorStore(config.sqlite_path if config.sqlite_path else ":memory:")

    if config.store_type == "pinecone":
        from ragmill.pinecone_store import PineconeVectorStore

        if not config.pinecone_api_key:
            raise ValueError("pinecone_api_key is required for PineconeVectorStore")
        return PineconeVectorStore(
            api_key=config.pinecone_api_key,
            environment=config.pinecone_environment,
            index_name=config.pinecone_index_name,
            embedding_dim=config.embedding_dim,
        )

    if config.store_type == "qdrant":
        from ragmill.qdrant_store import QdrantVectorStore

        if not config.qdrant_url:
            raise ValueError("qdrant_url is required for QdrantVectorStore")
        return QdrantVectorStore(
            url=config.qdrant_url,
            api_key=config.qdrant_api_key,
            collection_name=config.qdrant_collection_name,
            embedding_dim=config.embedding_dim,
            prefer_grpc=config.qdrant_prefer_grpc,
        )

    raise ValueError(f"Unknown store_type: {config.store_type!r}")
