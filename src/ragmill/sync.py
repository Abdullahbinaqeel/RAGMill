"""
Incremental sync between a directory and a VectorStore.

Re-running execute_pipeline() on every call re-embeds and re-inserts every
file, every time. sync_directory() instead tracks a content hash per file
(in VectorStore.file_state) so unchanged files are skipped entirely, changed
files have their old chunks replaced, and files removed from disk have their
chunks removed from the store too.

This module has no hard dependencies of its own — it only calls methods on
the engine/model/store objects passed in, so it stays zero-dependency at
import time just like engine.py.
"""

import hashlib
from typing import Any, Dict


def _hash_content(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sync_directory(directory_path: str, engine, model, store) -> Dict[str, int]:
    """
    Walks directory_path with `engine`, embeds changed/new files with `model`,
    and reconciles the result into `store`.

    :return: counts of {"added", "updated", "skipped", "deleted"} files.
    """
    seen_sources = set()
    added = updated = skipped = 0

    for file_manifest in engine.stream_directory(directory_path):
        source_file = file_manifest["source_path"]
        seen_sources.add(source_file)

        content_hash = _hash_content(file_manifest["raw_content"])
        existing_state = store.get_file_state(source_file)

        if existing_state is not None and existing_state["content_hash"] == content_hash:
            skipped += 1
            continue

        text_chunks = engine.semantic_chunking(file_manifest["raw_content"])
        payloads = [
            {
                "metadata": {
                    "source_file": source_file,
                    "filename": file_manifest["filename"],
                    "chunk_index": index,
                    "character_length": len(chunk),
                    "modified_at": file_manifest["modified_at"],
                },
                "content": chunk,
            }
            for index, chunk in enumerate(text_chunks)
        ]

        # Replace any previous chunks for this file before inserting the new ones,
        # so an updated file doesn't leave stale chunks from its old content behind.
        store.delete_by_source(source_file)
        if payloads:
            vectors = model.embed([p["content"] for p in payloads])
            store.add(payloads, vectors)
        store.upsert_file_state(source_file, content_hash, file_manifest["modified_at"])

        if existing_state is None:
            added += 1
        else:
            updated += 1

    deleted = store.delete_missing_sources(seen_sources)

    return {"added": added, "updated": updated, "skipped": skipped, "deleted": deleted}
