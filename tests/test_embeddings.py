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


def test_batched_embed_preserves_order_and_meaning(model):
    # Regression: embed() runs in length-sorted sub-batches instead of one giant
    # call (the old path OOM'd/thrashed on large inputs). Length-sorting + scatter
    # must NOT scramble the output order. Exact vector parity across batchings is
    # not achievable with this quantized ONNX model — padding perturbs values
    # slightly regardless of batch composition — so we assert what matters for
    # retrieval: each row maps back to its own text, and stays L2-normalized.
    texts = [
        "short",
        "a much longer sentence that has considerably more tokens than the others here",
        "medium length text about chunking",
        "tiny",
        "Semantic chunking preserves sentence boundaries when splitting text.",
    ]
    batched = model.embed(texts, batch_size=2)
    one_by_one = np.stack([model.embed([t])[0] for t in texts])
    assert batched.shape == (len(texts), EMBEDDING_DIM)

    # Order preserved: each batched row is most similar to its OWN single-text
    # embedding, proving the length-sort scatter maps rows back correctly.
    sims = batched @ one_by_one.T
    assert list(np.argmax(sims, axis=1)) == list(range(len(texts)))
    # And that self-similarity is high (semantics intact despite padding noise).
    for i in range(len(texts)):
        assert sims[i, i] > 0.95
        assert abs(np.linalg.norm(batched[i]) - 1.0) < 1e-4


def test_similar_sentences_score_higher_than_unrelated(model):
    vectors = model.embed(
        [
            "Semantic chunking preserves sentence boundaries when splitting text.",
            "The recursive text splitter avoids cutting sentences in half.",
            "The weather today is sunny and warm.",
        ]
    )
    sim_related = float(vectors[0] @ vectors[1])
    sim_unrelated = float(vectors[0] @ vectors[2])
    assert sim_related > sim_unrelated
