"""
Local embedding generation via ONNX Runtime.

Downloads a small quantized sentence-embedding model on first use and
caches it under ~/.cache/ragmill/models — every call after that runs
fully offline. Imports for onnxruntime/tokenizers are lazy so importing
this module doesn't force the 'embeddings' extra on core users.
"""

import logging
import time
import urllib.request
from pathlib import Path
from typing import List, Optional, Union

try:
    import numpy as np
except ImportError as exc:  # numpy ships with the embeddings/cloud extras
    raise ImportError(
        "Embeddings require the 'embeddings' extra. Install it with: pip install ragmill[embeddings]"
    ) from exc

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "Xenova/all-MiniLM-L6-v2"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "ragmill" / "models"
EMBEDDING_DIM = 384

# Chunks are embedded in fixed-size sub-batches rather than one giant call.
# Padding is per-batch (to the longest sequence in that batch), so a single huge
# call pads every short chunk up to the longest one AND allocates a multi-GB
# activation tensor — on CPU that thrashes memory and gets *slower* as the batch
# grows (measured: 25 chunks/s at 8, 8 chunks/s at 128). 16 was the throughput
# sweet spot on an 8-core CPU.
DEFAULT_EMBED_BATCH = 16

_MODEL_FILES = {
    "model.onnx": "onnx/model_quantized.onnx",
    "tokenizer.json": "tokenizer.json",
}


def _model_dir(model_name: str, cache_dir: Path) -> Path:
    return cache_dir / model_name.replace("/", "__")


def _download(model_name: str, cache_dir: Path, retries: int = 5) -> Path:
    target_dir = _model_dir(model_name, cache_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    for local_name, remote_path in _MODEL_FILES.items():
        local_path = target_dir / local_name
        if local_path.exists():
            continue
        url = f"https://huggingface.co/{model_name}/resolve/main/{remote_path}"
        partial_path = local_path.with_suffix(local_path.suffix + ".part")
        last_error: Optional[Exception] = None
        for attempt in range(retries):
            try:
                urllib.request.urlretrieve(url, partial_path)
                partial_path.rename(local_path)
                break
            except OSError as exc:
                last_error = exc
                partial_path.unlink(missing_ok=True)
                if attempt < retries - 1:
                    time.sleep(2**attempt)
        else:
            raise ConnectionError(
                f"Failed to download {local_name} from {model_name} after {retries} "
                f"attempts (last error: {last_error}). Check your network connection."
            ) from last_error

    return target_dir


class EmbeddingModel:
    """Wraps an ONNX sentence-embedding model + tokenizer for local inference."""

    def __init__(
        self, model_name: str = DEFAULT_MODEL, cache_dir: Optional[Union[str, Path]] = None
    ):
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

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(model_dir / "model.onnx"), sess_options)

        # Derive embedding dimension from the model's actual output shape
        # instead of hardcoding, so non-384-dim models fail loudly at init.
        output_meta = self.session.get_outputs()[0]
        self.embedding_dim = output_meta.shape[2]

    def embed(self, texts: List[str], batch_size: int = DEFAULT_EMBED_BATCH) -> np.ndarray:
        """Encodes a list of strings into L2-normalized dense vectors.

        Runs inference in fixed-size sub-batches instead of one call, and groups
        similar-length texts together so each batch pads to a comparable length
        (padding is per-batch). This bounds peak memory and keeps CPU throughput
        high regardless of how many texts are passed in — the returned vectors
        are identical to a single-call embed, just in the original input order.
        """
        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        # Sort by length so each batch holds comparable-length texts (minimizing
        # padding waste), embed batch-by-batch, then scatter back to input order.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        result = np.empty((len(texts), self.embedding_dim), dtype=np.float32)
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            vectors = self._embed_batch([texts[i] for i in indices])
            for row, i in enumerate(indices):
                result[i] = vectors[row]
        return result

    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        """Runs a single ONNX inference over one batch of strings."""
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
