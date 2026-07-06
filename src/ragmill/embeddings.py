"""
Local embedding generation via ONNX Runtime.

Downloads a small quantized sentence-embedding model on first use and
caches it under ~/.cache/ragmill/models — every call after that runs
fully offline. Imports for onnxruntime/tokenizers are lazy so importing
this module doesn't force the 'embeddings' extra on core users.
"""

import urllib.request
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

DEFAULT_MODEL = "Xenova/all-MiniLM-L6-v2"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "ragmill" / "models"
EMBEDDING_DIM = 384

_MODEL_FILES = {
    "model.onnx": "onnx/model_quantized.onnx",
    "tokenizer.json": "tokenizer.json",
}


def _model_dir(model_name: str, cache_dir: Path) -> Path:
    return cache_dir / model_name.replace("/", "__")


def _download(model_name: str, cache_dir: Path) -> Path:
    target_dir = _model_dir(model_name, cache_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    for local_name, remote_path in _MODEL_FILES.items():
        local_path = target_dir / local_name
        if local_path.exists():
            continue
        url = f"https://huggingface.co/{model_name}/resolve/main/{remote_path}"
        urllib.request.urlretrieve(url, local_path)

    return target_dir


class EmbeddingModel:
    """Wraps an ONNX sentence-embedding model + tokenizer for local inference."""

    def __init__(self, model_name: str = DEFAULT_MODEL, cache_dir: Optional[Union[str, Path]] = None):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "Embeddings require the 'embeddings' extra. Install it with: pip install ragmill[embeddings]"
            ) from exc
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise ImportError(
                "Embeddings require the 'embeddings' extra. Install it with: pip install ragmill[embeddings]"
            ) from exc

        resolved_cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        model_dir = _download(model_name, resolved_cache_dir)

        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_padding()
        self.tokenizer.enable_truncation(max_length=256)
        self.session = ort.InferenceSession(str(model_dir / "model.onnx"))

    def embed(self, texts: List[str]) -> np.ndarray:
        """Encodes a batch of strings into L2-normalized dense vectors."""
        if not texts:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

        encodings = self.tokenizer.encode_batch(texts)
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        token_type_ids = np.zeros_like(input_ids)

        outputs = self.session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            },
        )
        token_embeddings = outputs[0]  # (batch, seq_len, dim)

        mask = attention_mask[:, :, np.newaxis].astype(np.float32)
        summed = (token_embeddings * mask).sum(axis=1)
        counts = np.clip(mask.sum(axis=1), 1e-9, None)
        pooled = summed / counts

        norms = np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-9, None)
        return (pooled / norms).astype(np.float32)
