"""
Pinecone vector store backend.

Requires the 'pinecone' extra: pip install ragmill[pinecone]

Uses the Pinecone serverless API (the `pinecone` package, v5+ client class
based API — NOT the legacy `pinecone-client` module-level `pinecone.init()`
API, which was removed). Each chunk is stored as a vector with metadata
(source_file, filename, chunk_index, content, modified_at).

File-state tracking (for sync.py's skip-unchanged-files logic) is kept in
a dedicated namespace so it never shows up in count()/search() results.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import time

from ragmill.vector_store import BaseVectorStore

_STATE_NAMESPACE = "__ragmill_file_state__"
_DEFAULT_EMBEDDING_DIM = 384
_DEFAULT_RETRIES = 3


def _retry(fn: Callable[[], Any], retries: int = _DEFAULT_RETRIES) -> Any:
    """Call *fn* with exponential backoff on transient failures."""
    last_error: Optional[Exception] = None
    for attempt in range(retries):
        try:
            return fn()
        except (ConnectionError, TimeoutError) as exc:
            last_error = exc
            if attempt < retries - 1:
                time.sleep(2**attempt)
    raise last_error  # type: ignore[misc]


class PineconeVectorStore(BaseVectorStore):
    def __init__(
        self,
        api_key: str,
        environment: Optional[str] = None,
        index_name: str = "ragmill",
        embedding_dim: int = _DEFAULT_EMBEDDING_DIM,
    ):
        try:
            from pinecone import Pinecone, ServerlessSpec
        except ImportError as exc:
            raise ImportError(
                "Pinecone support requires the 'pinecone' extra. "
                "Install it with: pip install ragmill[pinecone]"
            ) from exc

        self._client = Pinecone(api_key=api_key)

        # Derive region from environment parameter if provided, else default.
        # Accepts formats like "us-west-2", "us-west1-gcp", "us-east-1-aws".
        region = "us-west-2"
        if environment:
            # Strip cloud-provider suffix (e.g. "-gcp", "-aws") and use the rest as region.
            parts = environment.split("-")
            # Find where cloud suffixes start (last 1-part tokens like "gcp", "aws")
            cloud_suffixes = {"gcp", "aws", "azure"}
            region_parts = parts
            for i in range(len(parts) - 1, 0, -1):
                if parts[i].lower() in cloud_suffixes:
                    region_parts = parts[:i]
                    break
            region = "-".join(region_parts)

        if not self._client.has_index(index_name):
            _retry(
                lambda: self._client.create_index(
                    name=index_name,
                    dimension=embedding_dim,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region=region),
                )
            )

        self.index = self._client.Index(index_name)
        self.index_name = index_name
        self._embedding_dim = embedding_dim

    def add(self, payloads: List[Dict[str, Any]], embeddings: np.ndarray) -> None:
        if len(payloads) != len(embeddings):
            raise ValueError("payloads and embeddings must be the same length")

        vectors = []
        for payload, vector in zip(payloads, embeddings):
            chunk_id = f"{payload['metadata']['source_file']}__{payload['metadata']['chunk_index']}"
            metadata: Dict[str, Any] = {
                "source_file": payload["metadata"]["source_file"],
                "filename": payload["metadata"]["filename"],
                "chunk_index": payload["metadata"]["chunk_index"],
                "content": payload["content"][:40000],
            }
            # Pinecone metadata cannot contain null values, so only include
            # modified_at when the file actually has one.
            raw_modified_at = payload["metadata"].get("modified_at")
            if raw_modified_at is not None:
                metadata["modified_at"] = float(raw_modified_at)
            vectors.append(
                {
                    "id": chunk_id,
                    "values": np.asarray(vector, dtype=np.float32).tolist(),
                    "metadata": metadata,
                }
            )

        # Pinecone has per-request vector limits; batch upserts of <=200.
        for i in range(0, len(vectors), 200):
            _retry(lambda i=i: self.index.upsert(vectors=vectors[i : i + 200]))  # type: ignore[misc]

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filename: Optional[str] = None,
        source_file: Optional[str] = None,
        modified_after: Optional[float] = None,
        modified_before: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        query_vector = np.asarray(query_embedding, dtype=np.float32).tolist()

        filter_dict: Dict[str, Any] = {}
        if filename is not None:
            filter_dict["filename"] = filename
        if source_file is not None:
            filter_dict["source_file"] = source_file
        if modified_after is not None:
            filter_dict["modified_at"] = {"$gte": modified_after}
        if modified_before is not None:
            if "modified_at" in filter_dict:
                filter_dict["modified_at"]["$lte"] = modified_before
            else:
                filter_dict["modified_at"] = {"$lte": modified_before}

        results = _retry(
            lambda: self.index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict or None,
            )
        )

        return [
            {
                "score": match.score,
                "metadata": {
                    "source_file": match.metadata.get("source_file", ""),
                    "filename": match.metadata.get("filename", ""),
                    "chunk_index": int(match.metadata.get("chunk_index", 0)),
                    "modified_at": match.metadata.get("modified_at"),
                },
                "content": match.metadata.get("content", ""),
            }
            for match in results.matches
        ]

    def count(self) -> int:
        stats = _retry(lambda: self.index.describe_index_stats())
        namespaces = stats.namespaces or {}
        total = stats.total_vector_count or 0
        # total_vector_count includes the file-state namespace; subtract it out
        # so count() reflects chunks only, matching SQLiteVectorStore semantics.
        state_ns = namespaces.get(_STATE_NAMESPACE)
        state_count = state_ns.vector_count if state_ns else 0
        return total - state_count

    def delete_by_source(self, source_file: str) -> None:
        # Pinecone serverless indexes do not support metadata-filtered deletes.
        # Instead, collect all chunk IDs for this source (IDs are deterministic:
        # "{source_file}__{chunk_index}") and delete by ID list.
        ids_to_delete: List[str] = []
        cursor = None
        while True:
            page = _retry(lambda c=cursor: self.index.list_paginated(limit=100, pagination_token=c))  # type: ignore[misc]
            for vector in page.vectors:
                if vector.id.startswith(source_file + "__"):
                    ids_to_delete.append(vector.id)
            cursor = page.pagination.next if page.pagination else None
            if cursor is None:
                break
        # Pinecone recommends ≤1000 IDs per delete call.
        for i in range(0, len(ids_to_delete), 1000):
            batch = ids_to_delete[i : i + 1000]
            _retry(lambda b=batch: self.index.delete(ids=b))  # type: ignore[misc]
        # Also clean up file_state for this source
        try:
            self.index.delete(ids=[source_file], namespace=_STATE_NAMESPACE)
        except Exception:
            pass

    def delete_missing_sources(self, known_sources: set) -> int:
        deleted = 0
        cursor = None
        seen_sources = set()
        while True:
            records, cursor = self.scroll(cursor=cursor, limit=100)
            for record in records:
                src = record["metadata"].get("source_file", "")
                if src:
                    seen_sources.add(src)
            if cursor is None:
                break
        for src in seen_sources:
            if src not in known_sources:
                self.delete_by_source(src)
                deleted += 1
        return deleted

    def _delete_file_state(self, source_file: str) -> None:
        """Delete a single file_state entry by ID."""
        try:
            self.index.delete(ids=[source_file], namespace=_STATE_NAMESPACE)
        except Exception:
            pass

    def get_file_state(self, source_file: str) -> Optional[Dict[str, Any]]:
        try:
            resp = _retry(lambda: self.index.fetch(ids=[source_file], namespace=_STATE_NAMESPACE))
            vec = resp.vectors.get(source_file)
            if vec is None:
                return None
            meta = vec.metadata or {}
            modified_at = meta.get("modified_at")
            return {
                "content_hash": meta.get("content_hash", ""),
                "modified_at": float(modified_at) if modified_at else None,
            }
        except (KeyError, ValueError, AttributeError):
            return None

    def upsert_file_state(
        self, source_file: str, content_hash: str, modified_at: Optional[float]
    ) -> None:
        _retry(
            lambda: self.index.upsert(
                vectors=[
                    {
                        "id": source_file,
                        "values": [0.0] * self._embedding_dim,
                        "metadata": {
                            "source_file": source_file,
                            "content_hash": content_hash,
                            "modified_at": modified_at if modified_at is not None else "",
                        },
                    }
                ],
                namespace=_STATE_NAMESPACE,
            )
        )

    def scroll(
        self, cursor: Optional[str] = None, limit: int = 100
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        page = _retry(lambda: self.index.list_paginated(limit=limit, pagination_token=cursor))
        ids = [v.id for v in page.vectors]
        if not ids:
            return [], None

        fetched = _retry(lambda: self.index.fetch(ids=ids))
        records = []
        for vector_id in ids:
            vec = fetched.vectors.get(vector_id)
            if vec is None:
                continue
            meta = vec.metadata or {}
            records.append(
                {
                    "metadata": {
                        "source_file": meta.get("source_file", ""),
                        "filename": meta.get("filename", ""),
                        "chunk_index": meta.get("chunk_index", 0),
                        "modified_at": meta.get("modified_at"),
                    },
                    "content": meta.get("content", ""),
                    "embedding": np.asarray(vec.values, dtype=np.float32),
                }
            )

        next_cursor = page.pagination.next if page.pagination else None
        return records, next_cursor

    def close(self) -> None:
        pass
