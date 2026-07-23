# FAQ

### Does RAGMill need an API key?

No. By default everything runs locally and offline — embeddings via ONNX, chat
via a local GGUF model. API keys are only needed if you *opt into* the Gemini or
OpenAI chat backends, or the Pinecone/Qdrant cloud stores.

### Are the models downloaded when I `pip install`?

No. `pip install` only installs Python code. Model weights download **lazily on
first use** and are cached in `~/.cache/ragmill/models/`:

- embedding model (~22 MB) on your first `embed()` / `sync` / `search`
- local chat model (~1 GB) on your first `chat`

That's why the first run after install is slow — it's the download, not the
pipeline.

### Why was my first sync so slow / did it hang?

Almost always the one-time model download on a fresh cache, or embedding a large
folder on CPU. Embedding is memory-bounded and batched, so it shouldn't thrash —
but a big corpus on CPU still takes time. The CLI prints per-file progress so you
can see it moving.

### Is the model fine-tuned on my data?

No. RAGMill does no training or fine-tuning. It uses off-the-shelf pretrained
models and improves answers purely through *retrieval* — feeding the LLM the
most relevant chunks of your documents.

### Can I change the models?

- **Chat model:** yes, freely — point `RAGMILL_CHAT_MODEL_REPO` /
  `RAGMILL_CHAT_MODEL_FILE` at any GGUF, or switch to a hosted backend.
- **Embedding model:** yes, but it must be a Xenova-style ONNX repo and
  384-dimensional (a different dimension needs a one-line code change). See
  [Configuration](guide/configuration.md).

### What file types are supported?

Core (no extra): `.txt`, `.md`, `.log`, `.rst`, `.csv`, `.tsv`.

With extras:

- `.pdf` — `pdf` extra
- `.docx` — `docx` extra
- `.html`, `.htm`, `.rtf`, `.xlsx`, `.pptx` — `office` extra
- `.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp`, `.gif`, and scanned/image-only PDFs (OCR) — `ocr` extra

OCR needs the system `tesseract` binary (and `pdftoppm`/poppler for scanned
PDFs), and is English-only by default. A scanned PDF with no text layer falls
back to OCR automatically when the `ocr` extra is installed.

### How large a corpus can the SQLite store handle?

The built-in store does a brute-force scan — great to tens of thousands of
chunks. For millions, use Qdrant or Pinecone behind the same interface. See
[Vector stores](guide/vector-stores.md).

### How do I persist data with the CLI?

Set `RAGMILL_SQLITE_PATH=./ragmill.db`. Without it, the store is in-memory and
disappears when the process exits.

### Where can I visualize my vectors?

Migrate to Qdrant and use its built-in dashboard, or project the vectors to 2-D
with UMAP/t-SNE in a notebook (`store.scroll()` gives you embeddings). Details in
[Vector stores](guide/vector-stores.md).

### It fails to install `onnxruntime` on my Python version.

You're likely on a very new Python (3.13/3.14) that `onnxruntime` doesn't have
wheels for yet. Use **Python 3.12**.

### How do I reset everything?

Delete the store (`rm ragmill.db*` for SQLite) and re-index. To also reclaim the
model cache: `rm -rf ~/.cache/ragmill/models`.
