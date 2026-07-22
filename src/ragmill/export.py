"""
Export/import utilities for migrating between vector store backends.

Writes chunks + embeddings as newline-delimited JSON (JSONL) so the data
is portable across environments. A typical workflow:

    # 1. Export from your local SQLite store
    export_store("embeddings.jsonl", sqlite_store)

    # 2. Import into a cloud Pinecone index
    import_store("embeddings.jsonl", pinecone_store)
"""

import base64
import json
from typing import Any, Dict, List

import numpy as np

from ragmill.vector_store import BaseVectorStore


def export_store(
    output_path: str,
    store: BaseVectorStore,
    batch_size: int = 100,
) -> int:
    """
    Export all chunks + embeddings from *store* to a JSONL file.

    Works against any BaseVectorStore backend via its scroll() method.
    Returns the number of records written.
    """
    cursor = None
    written = 0

    with open(output_path, "w") as f:
        while True:
            records, cursor = store.scroll(cursor=cursor, limit=batch_size)
            for record in _serialize_records(records):
                f.write(record + "\n")
                written += 1
            if not cursor:
                break

    return written


def import_store(
    input_path: str,
    store: BaseVectorStore,
    batch_size: int = 100,
) -> int:
    """
    Import chunks + embeddings from a JSONL file into *store*.

    Returns the number of records imported.
    """
    imported = 0
    batch_payloads: List[Dict[str, Any]] = []
    batch_vectors: List[np.ndarray] = []

    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            payload, vector = _deserialize_record(record)
            batch_payloads.append(payload)
            batch_vectors.append(vector)

            if len(batch_payloads) >= batch_size:
                store.add(
                    batch_payloads,
                    np.stack(batch_vectors).astype(np.float32),
                )
                imported += len(batch_payloads)
                batch_payloads = []
                batch_vectors = []

    if batch_payloads:
        store.add(
            batch_payloads,
            np.stack(batch_vectors).astype(np.float32),
        )
        imported += len(batch_payloads)

    return imported


# ── Internal helpers ────────────────────────────────────────────────────────

CHUNK_SCHEMA_VERSION = 1


def _serialize_records(records: List[Dict[str, Any]]) -> List[str]:
    lines = []
    for rec in records:
        vector = rec.get("embedding", None)
        if vector is None:
            continue
        vector = np.asarray(vector, dtype=np.float32)
        # Build a serialized copy to avoid mutating the input record.
        serialized = {
            "metadata": rec.get("metadata", {}),
            "content": rec.get("content", ""),
            "embedding_b64": base64.b64encode(vector.tobytes()).decode("ascii"),
            "schema_version": CHUNK_SCHEMA_VERSION,
        }
        lines.append(json.dumps(serialized, ensure_ascii=False))
    return lines


def _deserialize_record(record: Dict[str, Any]) -> tuple:
    b64 = record.pop("embedding_b64", None)
    if b64 is None:
        raise ValueError("record missing embedding_b64 field")
    vector = np.frombuffer(base64.b64decode(b64), dtype=np.float32)
    metadata = record.get("metadata", {})
    payload = {
        "metadata": {
            "source_file": metadata.get("source_file", ""),
            "filename": metadata.get("filename", ""),
            "chunk_index": metadata.get("chunk_index", 0),
            "character_length": len(record.get("content", "")),
            "modified_at": metadata.get("modified_at"),
        },
        "content": record.get("content", ""),
    }
    return payload, vector
