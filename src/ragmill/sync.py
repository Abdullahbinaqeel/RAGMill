"""
Incremental sync between a directory and a VectorStore.

Re-running execute_pipeline() on every call re-embeds and re-inserts every
file, every time. sync_directory() instead tracks a content hash per file
(in VectorStore.file_state) so unchanged files are skipped entirely, changed
files have their old chunks replaced, and files removed from disk have their
chunks removed from the store too.

Changed/new files are grouped into batches of up to `batch_size` before
embedding, so one model call covers many files' chunks instead of paying
per-call tokenizer/inference overhead once per file, and the whole run is
wrapped in a single store.batch() — one SQLite commit for the entire sync
instead of one (or three) per file. Neither change alters which files get
embedded, what their vectors are, or the counts returned — same content in,
same content out, just far fewer round trips to get there.

This module has no hard dependencies of its own — it only calls methods on
the engine/model/store objects passed in, so it stays zero-dependency at
import time just like engine.py.
"""

import hashlib
from typing import Any, Callable, Dict, List, Optional, Tuple

DEFAULT_BATCH_SIZE = 64


def _hash_content(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _flush(pending: List[Dict[str, Any]], model, store) -> Tuple[int, int]:
    """Embed every pending file's chunks in one batch call, then write each
    file's chunks + file_state (deletes are per-file, so an update never
    leaves stale chunks behind, same as before)."""
    all_texts: List[str] = []
    spans: List[Tuple[int, int]] = []
    for item in pending:
        start = len(all_texts)
        all_texts.extend(item["text_chunks"])
        spans.append((start, len(all_texts)))

    vectors = model.embed(all_texts)

    added = updated = 0
    for item, (start, end) in zip(pending, spans):
        payloads = [
            {
                "metadata": {
                    "source_file": item["source_file"],
                    "filename": item["filename"],
                    "chunk_index": index,
                    "character_length": len(chunk),
                    "modified_at": item["modified_at"],
                },
                "content": chunk,
            }
            for index, chunk in enumerate(item["text_chunks"])
        ]

        # Replace any previous chunks for this file before inserting the new ones,
        # so an updated file doesn't leave stale chunks from its old content behind.
        store.delete_by_source(item["source_file"])
        if payloads:
            store.add(payloads, vectors[start:end])
        store.upsert_file_state(item["source_file"], item["content_hash"], item["modified_at"])

        if item["existing_state"] is None:
            added += 1
        else:
            updated += 1

    return added, updated


def sync_directory(
    directory_path: str,
    engine,
    model,
    store,
    batch_size: int = DEFAULT_BATCH_SIZE,
    progress: Optional[Callable[[int, str], None]] = None,
) -> Dict[str, int]:
    """
    Walks directory_path with `engine`, embeds changed/new files with `model`,
    and reconciles the result into `store`.

    :param progress: optional callback invoked once per file seen, as
        progress(files_seen_so_far, source_file), for surfacing live feedback
        during long syncs.
    :return: counts of {"added", "updated", "skipped", "deleted"} files.
    """
    seen_sources = set()
    added = updated = skipped = 0
    pending: List[Dict[str, Any]] = []

    with store.batch():
        for file_manifest in engine.stream_directory(directory_path):
            source_file = file_manifest["source_path"]
            seen_sources.add(source_file)
            if progress is not None:
                progress(len(seen_sources), source_file)

            content_hash = _hash_content(file_manifest["raw_content"])
            existing_state = store.get_file_state(source_file)

            if existing_state is not None and existing_state["content_hash"] == content_hash:
                skipped += 1
                continue

            pending.append(
                {
                    "source_file": source_file,
                    "filename": file_manifest["filename"],
                    "modified_at": file_manifest["modified_at"],
                    "content_hash": content_hash,
                    "existing_state": existing_state,
                    "text_chunks": engine.semantic_chunking(file_manifest["raw_content"]),
                }
            )

            if len(pending) >= batch_size:
                batch_added, batch_updated = _flush(pending, model, store)
                added += batch_added
                updated += batch_updated
                pending = []

        if pending:
            batch_added, batch_updated = _flush(pending, model, store)
            added += batch_added
            updated += batch_updated

        deleted = store.delete_missing_sources(seen_sources)

    return {"added": added, "updated": updated, "skipped": skipped, "deleted": deleted}
