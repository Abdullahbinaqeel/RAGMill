import numpy as np
import pytest

pytest.importorskip("onnxruntime")
pytest.importorskip("tokenizers")

from ragmill.embeddings import EmbeddingModel, EMBEDDING_DIM


@pytest.fixture(scope="module")
def model():
    try:
        return EmbeddingModel()
    except Exception as exc:
        pytest.skip(f"embedding model unavailable (likely no network to download it): {exc}")


def test_embed_empty_list_returns_empty_array(model):
    result = model.embed([])
    assert result.shape == (0, EMBEDDING_DIM)


def test_embed_returns_correct_shape(model):
    vectors = model.embed(["one sentence", "another sentence here"])
    assert vectors.shape == (2, EMBEDDING_DIM)


def test_embeddings_are_l2_normalized(model):
    vectors = model.embed(["RAGMill ingests local documents and splits them into chunks."])
    norm = np.linalg.norm(vectors[0])
    assert abs(norm - 1.0) < 1e-4


def test_similar_sentences_score_higher_than_unrelated(model):
    vectors = model.embed([
        "Semantic chunking preserves sentence boundaries when splitting text.",
        "The recursive text splitter avoids cutting sentences in half.",
        "The weather today is sunny and warm.",
    ])
    sim_related = float(vectors[0] @ vectors[1])
    sim_unrelated = float(vectors[0] @ vectors[2])
    assert sim_related > sim_unrelated
