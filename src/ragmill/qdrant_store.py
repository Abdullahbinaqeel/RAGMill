"""
Qdrant vector store backend.

Requires the 'qdrant' extra: pip install ragmill[qdrant]

Connects to a running Qdrant instance (local or cloud) and stores
chunks as points with payload metadata.
"""

import uuid
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ragmill.vector_store import BaseVectorStore


def _retry(fn: Callable[[], Any], retries: int = 3) -> Any:
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


def _stable_point_id(key: str) -> str:
    """
    Deterministic point ID for a logical key, stable across processes.

    Qdrant requires unsigned-int or UUID point IDs; Python's hash() is
    randomized per-process (PYTHONHASHSEED), so it can't be used here.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


class QdrantVectorStore(BaseVectorStore):
    def __init__(
        self,
        url: str,
        api_key: Optional[str] = None,
        collection_name: str = "ragmill",
        embedding_dim: int = 384,
        prefer_grpc: bool = False,
    ):
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models
        except ImportError as exc:
            raise ImportError(
                "Qdrant support requires the 'qdrant' extra. "
                "Install it with: pip install ragmill[qdrant]"
            ) from exc

        self._models = models
        self.client = QdrantClient(
            url=url,
            api_key=api_key,
            prefer_grpc=prefer_grpc,
        )
        self.collection_name = collection_name
        self._state_collection = f"{collection_name}__file_state"

        collections = _retry(lambda: self.client.get_collections()).collections
        existing = {c.name for c in collections}

        if collection_name not in existing:
            _retry(
                lambda: self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=embedding_dim,
                        distance=models.Distance.COSINE,
                    ),
                )
            )
            # Managed Qdrant clusters reject filtering on a field with no
            # payload index ("Index required but not found"); local/self-hosted
            # Qdrant is more lenient, which is why this only surfaces against a
            # real cluster. filename/source_file are filtered by search() and
            # delete_by_source()/delete_missing_sources() (used by sync()), so
            # without these indexes those calls fail outright on Qdrant Cloud.
            for field in ("filename", "source_file"):
                _retry(
                    lambda f=field: self.client.create_payload_index(  # type: ignore[misc]
                        collection_name=collection_name,
                        field_name=f,
                        field_schema=models.PayloadSchemaType.KEYWORD,
                    )
                )

        if self._state_collection not in existing:
            _retry(
                lambda: self.client.create_collection(
                    collection_name=self._state_collection,
                    vectors_config=models.VectorParams(
                        size=1,
                        distance=models.Distance.COSINE,
                    ),
                )
            )
            # get_file_state() filters this collection by source_file — needs
            # the same payload index as above, for the same reason.
            _retry(
                lambda: self.client.create_payload_index(
                    collection_name=self._state_collection,
                    field_name="source_file",
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            )

    def add(self, payloads: List[Dict[str, Any]], embeddings: np.ndarray) -> None:
        if len(payloads) != len(embeddings):
            raise ValueError("payloads and embeddings must be the same length")

        from qdrant_client.http import models

        points = []
        for i, (payload, vector) in enumerate(zip(payloads, embeddings)):
            chunk_id = f"{payload['metadata']['source_file']}__{payload['metadata']['chunk_index']}"
            # A file may legitimately have no modified_at (explicit None). Qdrant
            # payloads allow null, so store None rather than coercing to 0.0.
            raw_modified_at = payload["metadata"].get("modified_at")
            points.append(
                models.PointStruct(
                    id=_stable_point_id(chunk_id),
                    vector=np.asarray(vector, dtype=np.float32).tolist(),
                    payload={
                        "_chunk_id": chunk_id,
                        "source_file": payload["metadata"]["source_file"],
                        "filename": payload["metadata"]["filename"],
                        "chunk_index": payload["metadata"]["chunk_index"],
                        "content": payload["content"][:40000],
                        "modified_at": (
                            float(raw_modified_at) if raw_modified_at is not None else None
                        ),
                    },
                )
            )

        _retry(lambda: self.client.upsert(collection_name=self.collection_name, points=points))

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

        filter_conditions: list = []
        if filename is not None:
            filter_conditions.append(
                self._models.FieldCondition(
                    key="filename",
                    match=self._models.MatchValue(value=filename),
                )
            )
        if source_file is not None:
            filter_conditions.append(
                self._models.FieldCondition(
                    key="source_file",
                    match=self._models.MatchValue(value=source_file),
                )
            )
        if modified_after is not None:
            filter_conditions.append(
                self._models.FieldCondition(
                    key="modified_at",
                    range=self._models.Range(gte=modified_after),
                )
            )
        if modified_before is not None:
            filter_conditions.append(
                self._models.FieldCondition(
                    key="modified_at",
                    range=self._models.Range(lte=modified_before),
                )
            )

        search_filter = self._models.Filter(must=filter_conditions) if filter_conditions else None

        # QdrantClient.search() was removed in qdrant-client >=1.10 in favor
        # of the unified query_points() endpoint.
        results = _retry(
            lambda: self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                query_filter=search_filter,
            )
        ).points

        return [
            {
                "score": float(result.score),
                "metadata": {
                    "source_file": result.payload.get("source_file", ""),
                    "filename": result.payload.get("filename", ""),
                    "chunk_index": int(result.payload.get("chunk_index", 0)),
                    "modified_at": result.payload.get("modified_at"),
                },
                "content": result.payload.get("content", ""),
            }
            for result in results
        ]

    def count(self) -> int:
        result = _retry(
            lambda: self.client.count(
                collection_name=self.collection_name,
                exact=True,
            )
        )
        return result.count

    def delete_by_source(self, source_file: str) -> None:
        _retry(
            lambda: self.client.delete(
                collection_name=self.collection_name,
                points_selector=self._models.FilterSelector(
                    filter=self._models.Filter(
                        must=[
                            self._models.FieldCondition(
                                key="source_file",
                                match=self._models.MatchValue(value=source_file),
                            )
                        ]
                    )
                ),
            )
        )
        # Also clean up file_state entry for this source
        _retry(
            lambda: self.client.delete(
                collection_name=self._state_collection,
                points_selector=self._models.FilterSelector(
                    filter=self._models.Filter(
                        must=[
                            self._models.FieldCondition(
                                key="source_file",
                                match=self._models.MatchValue(value=source_file),
                            )
                        ]
                    )
                ),
            )
        )

    def delete_missing_sources(self, known_sources: set) -> int:
        seen = set()
        next_offset = None
        while True:
            points, next_offset = _retry(
                lambda: self.client.scroll(
                    collection_name=self.collection_name,
                    limit=100,
                    offset=next_offset,
                    with_payload=True,
                    with_vectors=False,
                )
            )
            for point in points:
                src = point.payload.get("source_file", "")
                if src and src not in known_sources:
                    seen.add(src)
            if next_offset is None:
                break
        # Also scan state collection for orphaned entries
        state_next_offset = None
        while True:
            points, state_next_offset = _retry(
                lambda: self.client.scroll(
                    collection_name=self._state_collection,
                    limit=100,
                    offset=state_next_offset,
                    with_payload=True,
                    with_vectors=False,
                )
            )
            for point in points:
                src = point.payload.get("source_file", "")
                if src and src not in known_sources:
                    seen.add(src)
            if state_next_offset is None:
                break
        for src in seen:
            self.delete_by_source(src)
        return len(seen)

    def get_file_state(self, source_file: str) -> Optional[Dict[str, Any]]:
        try:
            points, _ = _retry(
                lambda: self.client.scroll(
                    collection_name=self._state_collection,
                    limit=1,
                    with_payload=True,
                    with_vectors=False,
                    scroll_filter=self._models.Filter(
                        must=[
                            self._models.FieldCondition(
                                key="source_file",
                                match=self._models.MatchValue(value=source_file),
                            )
                        ]
                    ),
                )
            )
            if not points:
                return None
            meta = points[0].payload
            modified_at = meta.get("modified_at")
            return {
                "content_hash": meta.get("content_hash", ""),
                "modified_at": float(modified_at) if modified_at is not None else None,
            }
        except (ValueError, KeyError, AttributeError):
            return None

    def upsert_file_state(
        self, source_file: str, content_hash: str, modified_at: Optional[float]
    ) -> None:
        _retry(
            lambda: self.client.upsert(
                collection_name=self._state_collection,
                points=[
                    self._models.PointStruct(
                        id=_stable_point_id(f"filestate::{source_file}"),
                        vector=[0.0],
                        payload={
                            "source_file": source_file,
                            "content_hash": content_hash,
                            "modified_at": modified_at,
                        },
                    )
                ],
            )
        )

    def scroll(
        self, cursor: Optional[Any] = None, limit: int = 100
    ) -> Tuple[List[Dict[str, Any]], Optional[Any]]:
        points, next_cursor = _retry(
            lambda: self.client.scroll(
                collection_name=self.collection_name,
                limit=limit,
                offset=cursor,
                with_payload=True,
                with_vectors=True,
            )
        )
        records = [
            {
                "metadata": {
                    "source_file": point.payload.get("source_file", ""),
                    "filename": point.payload.get("filename", ""),
                    "chunk_index": point.payload.get("chunk_index", 0),
                    "modified_at": point.payload.get("modified_at"),
                },
                "content": point.payload.get("content", ""),
                "embedding": np.asarray(point.vector, dtype=np.float32),
            }
            for point in points
        ]
        return records, next_cursor

    def close(self) -> None:
        self.client.close()
