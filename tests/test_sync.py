import numpy as np
import pytest

pytest.importorskip("onnxruntime")
pytest.importorskip("tokenizers")

from ragmill import RAGEngine
from ragmill.embeddings import EmbeddingModel
from ragmill.vector_store import VectorStore
from ragmill.sync import sync_directory


@pytest.fixture(scope="module")
def model():
    try:
        return EmbeddingModel()
    except Exception as exc:
        pytest.skip(f"embedding model unavailable (likely no network to download it): {exc}")


@pytest.fixture
def engine():
    return RAGEngine(chunk_size=200, overlap=20)


def test_first_sync_adds_every_file(tmp_path, engine, model):
    (tmp_path / "a.txt").write_text("Alpha file content.", encoding="utf-8")
    (tmp_path / "b.txt").write_text("Bravo file content.", encoding="utf-8")
    store = VectorStore()

    result = sync_directory(str(tmp_path), engine, model, store)

    assert result == {"added": 2, "updated": 0, "skipped": 0, "deleted": 0}
    assert store.count() == 2


def test_second_sync_with_no_changes_skips_everything(tmp_path, engine, model):
    (tmp_path / "a.txt").write_text("Alpha file content.", encoding="utf-8")
    store = VectorStore()
    sync_directory(str(tmp_path), engine, model, store)

    result = sync_directory(str(tmp_path), engine, model, store)

    assert result == {"added": 0, "updated": 0, "skipped": 1, "deleted": 0}
    assert store.count() == 1


def test_modifying_a_file_triggers_update_not_duplicate(tmp_path, engine, model):
    file_path = tmp_path / "a.txt"
    file_path.write_text("Original content here.", encoding="utf-8")
    store = VectorStore()
    sync_directory(str(tmp_path), engine, model, store)

    file_path.write_text("Completely different content now.", encoding="utf-8")
    result = sync_directory(str(tmp_path), engine, model, store)

    assert result == {"added": 0, "updated": 1, "skipped": 0, "deleted": 0}
    assert store.count() == 1

    query_vector = model.embed(["Completely different content now."])[0]
    top_match = store.search(query_vector, top_k=1)[0]
    assert "Completely different" in top_match["content"]


def test_removing_a_file_deletes_its_chunks(tmp_path, engine, model):
    keep_path = tmp_path / "keep.txt"
    remove_path = tmp_path / "remove.txt"
    keep_path.write_text("This one stays.", encoding="utf-8")
    remove_path.write_text("This one goes away.", encoding="utf-8")
    store = VectorStore()
    sync_directory(str(tmp_path), engine, model, store)
    assert store.count() == 2

    remove_path.unlink()
    result = sync_directory(str(tmp_path), engine, model, store)

    assert result == {"added": 0, "updated": 0, "skipped": 1, "deleted": 1}
    assert store.count() == 1
    remaining = store.search(model.embed(["stays"])[0], top_k=5)
    assert all("goes away" not in r["content"] for r in remaining)
