"""
A minimal local vector store backed by SQLite.

Stores chunk payloads alongside their embedding vectors and performs
brute-force cosine similarity search in-memory. No external vector index
needed at the scale this library targets (a local folder's worth of
documents) — every payload is loaded once per search and scored with a
single matrix multiply.

Also tracks per-file content hashes (the `file_state` table) so callers
can detect which files changed since the last run without re-embedding
everything — see sync.py for the orchestration that uses this.
"""

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


class VectorStore:
    def __init__(self, db_path: Union[str, Path] = ":memory:"):
        self.connection = sqlite3.connect(str(db_path))
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_file TEXT,
                filename TEXT,
                chunk_index INTEGER,
                content TEXT NOT NULL,
                modified_at REAL,
                embedding BLOB NOT NULL
            )
            """
        )
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS file_state (
                source_file TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL,
                modified_at REAL
            )
            """
        )
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
        self.connection.executemany(
            """
            INSERT INTO chunks (source_file, filename, chunk_index, content, modified_at, embedding)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self.connection.commit()

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

        query = "SELECT source_file, filename, chunk_index, content, embedding FROM chunks"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)

        rows = self.connection.execute(query, params).fetchall()
        if not rows:
            return []

        vectors = np.stack([np.frombuffer(row[4], dtype=np.float32) for row in rows])
        # Embeddings are pre-normalized (see EmbeddingModel.embed), so the dot
        # product against a normalized query is equivalent to cosine similarity.
        scores = vectors @ query_vector

        top_indices = np.argsort(-scores)[:top_k]
        return [
            {
                "score": float(scores[i]),
                "metadata": {
                    "source_file": rows[i][0],
                    "filename": rows[i][1],
                    "chunk_index": rows[i][2],
                },
                "content": rows[i][3],
            }
            for i in top_indices
        ]

    def count(self) -> int:
        return self.connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def delete_by_source(self, source_file: str) -> None:
        self.connection.execute("DELETE FROM chunks WHERE source_file = ?", (source_file,))
        self.connection.execute("DELETE FROM file_state WHERE source_file = ?", (source_file,))
        self.connection.commit()

    def delete_missing_sources(self, known_sources: set) -> int:
        """Removes chunks/file_state for any source_file not in known_sources. Returns count removed."""
        rows = self.connection.execute("SELECT source_file FROM file_state").fetchall()
        stale = [row[0] for row in rows if row[0] not in known_sources]
        for source_file in stale:
            self.delete_by_source(source_file)
        return len(stale)

    def get_file_state(self, source_file: str) -> Optional[Dict[str, Any]]:
        row = self.connection.execute(
            "SELECT content_hash, modified_at FROM file_state WHERE source_file = ?",
            (source_file,),
        ).fetchone()
        if row is None:
            return None
        return {"content_hash": row[0], "modified_at": row[1]}

    def upsert_file_state(self, source_file: str, content_hash: str, modified_at: Optional[float]) -> None:
        self.connection.execute(
            """
            INSERT INTO file_state (source_file, content_hash, modified_at)
            VALUES (?, ?, ?)
            ON CONFLICT(source_file) DO UPDATE SET content_hash = excluded.content_hash, modified_at = excluded.modified_at
            """,
            (source_file, content_hash, modified_at),
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()
