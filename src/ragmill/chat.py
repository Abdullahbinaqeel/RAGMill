"""
Retrieval-augmented answer generation.

Takes chunks returned by a vector store search and asks an LLM to
synthesize a grounded natural-language answer from them. Three
backends, selected via RAGMILL_CHAT_BACKEND (default "local"):

  - "local"  — a small GGUF model via llama-cpp-python, downloaded once
               and cached under ~/.cache/ragmill/models (same pattern as
               embeddings.py). Fully offline, no API key required.
  - "gemini" — Google's Gemini API (requires GEMINI_API_KEY/GOOGLE_API_KEY).
  - "openai" — OpenAI's Chat Completions API (requires OPENAI_API_KEY).

Each backend's SDK is imported lazily so importing this module never
forces any of the three extras on core users.
"""

import logging
import os
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ragmill.config import RAGMillConfig

logger = logging.getLogger(__name__)

DEFAULT_MODEL_REPO = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
DEFAULT_MODEL_FILE = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
DEFAULT_N_CTX = 4096
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "ragmill" / "models"

DEFAULT_GEMINI_MODEL = "gemini-flash-latest"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions using only the provided "
    "context chunks retrieved from a knowledge base. Cite the source filename "
    "for each claim in brackets, e.g. [report.pdf]. If the context doesn't "
    "contain enough information to answer, say so clearly instead of guessing."
)

_llm_cache: Dict[Tuple[str, str, int], Any] = {}


def _format_context(chunks: List[Dict[str, Any]]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        filename = chunk.get("metadata", {}).get("filename", "unknown")
        parts.append(f"[{i}] ({filename})\n{chunk['content']}")
    return "\n\n".join(parts)


# ── Local backend (llama-cpp-python) ────────────────────────────────────────


def _download_gguf(repo_id: str, filename: str, cache_dir: Path, retries: int = 5) -> Path:
    target_dir = cache_dir / repo_id.replace("/", "__")
    target_dir.mkdir(parents=True, exist_ok=True)

    local_path = target_dir / filename
    if local_path.exists():
        return local_path

    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    partial_path = local_path.with_suffix(local_path.suffix + ".part")

    last_error: Optional[Exception] = None
    for attempt in range(retries):
        try:
            urllib.request.urlretrieve(url, partial_path)
            partial_path.rename(local_path)
            return local_path
        except OSError as exc:
            last_error = exc
            partial_path.unlink(missing_ok=True)
            if attempt < retries - 1:
                time.sleep(2**attempt)

    raise ConnectionError(
        f"Failed to download {filename} from {repo_id} after {retries} attempts "
        f"(last error: {last_error}). This is usually a transient DNS/network issue — retry."
    ) from last_error


def _get_llm(repo_id: str, filename: str, n_ctx: int):
    """Loads (or reuses a cached) local GGUF model — loading is expensive, so
    the same instance is reused across calls within a process."""
    key = (repo_id, filename, n_ctx)
    if key not in _llm_cache:
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise ImportError(
                "The local chat backend requires the 'chat' extra. "
                "Install it with: pip install ragmill[chat]"
            ) from exc

        model_path = _download_gguf(repo_id, filename, DEFAULT_CACHE_DIR)
        _llm_cache[key] = Llama(model_path=str(model_path), n_ctx=n_ctx, verbose=False)

    return _llm_cache[key]


def _generate_local(
    query: str, chunks: List[Dict[str, Any]], context: str, config: Optional["RAGMillConfig"] = None
) -> str:
    repo_id = (config.chat_model_repo if config else None) or os.getenv(
        "RAGMILL_CHAT_MODEL_REPO", DEFAULT_MODEL_REPO
    )
    filename = (config.chat_model_file if config else None) or os.getenv(
        "RAGMILL_CHAT_MODEL_FILE", DEFAULT_MODEL_FILE
    )
    n_ctx = int(
        (str(config.chat_n_ctx) if config else None)
        or os.getenv("RAGMILL_CHAT_N_CTX", str(DEFAULT_N_CTX))
    )

    llm = _get_llm(repo_id, filename, n_ctx)
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
        temperature=0.2,
        max_tokens=512,
    )
    return response["choices"][0]["message"]["content"] or ""


# ── Gemini backend ───────────────────────────────────────────────────────────


def _generate_gemini(
    query: str, chunks: List[Dict[str, Any]], context: str, config: Optional["RAGMillConfig"] = None
) -> str:
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise ImportError(
            "The Gemini chat backend requires the 'chat-gemini' extra. "
            "Install it with: pip install ragmill[chat-gemini]"
        ) from exc

    api_key = (
        (config.gemini_api_key if config else None)
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=(config.gemini_model if config else None)
        or os.getenv("RAGMILL_GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
        contents=f"Context:\n{context}\n\nQuestion: {query}",
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
    )
    return response.text or ""


# ── OpenAI (ChatGPT) backend ─────────────────────────────────────────────────


def _generate_openai(
    query: str, chunks: List[Dict[str, Any]], context: str, config: Optional["RAGMillConfig"] = None
) -> str:
    try:
        import openai
    except ImportError as exc:
        raise ImportError(
            "The OpenAI chat backend requires the 'chat-openai' extra. "
            "Install it with: pip install ragmill[chat-openai]"
        ) from exc

    api_key = (config.openai_api_key if config else None) or os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI(api_key=api_key) if api_key else openai.OpenAI()
    response = client.chat.completions.create(
        model=(config.openai_model if config else None)
        or os.getenv("RAGMILL_OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
    )
    return response.choices[0].message.content or ""


# ── Dispatch ─────────────────────────────────────────────────────────────────

_BACKENDS = {
    "local": _generate_local,
    "gemini": _generate_gemini,
    "openai": _generate_openai,
}


def generate_answer(
    query: str, chunks: List[Dict[str, Any]], config: Optional["RAGMillConfig"] = None
) -> str:
    """Synthesizes a grounded answer to `query` from retrieved `chunks`.

    Backend is selected via RAGMILL_CHAT_BACKEND ("local" [default], "gemini",
    or "openai"). If `config` is provided, its chat-related fields are used
    (with env vars as fallback for backward compatibility)."""
    backend_name = (config.chat_backend if config else None) or os.getenv(
        "RAGMILL_CHAT_BACKEND", "local"
    )
    backend_name = backend_name.lower()
    backend = _BACKENDS.get(backend_name)
    if backend is None:
        raise ValueError(
            f"Unknown RAGMILL_CHAT_BACKEND: {backend_name!r} "
            f"(expected one of {sorted(_BACKENDS)})"
        )

    context = (
        _format_context(chunks)
        if chunks
        else "No relevant context was found in the knowledge base."
    )
    return backend(query, chunks, context, config)
