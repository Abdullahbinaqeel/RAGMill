# Installation

## Requirements

- **Python 3.9 – 3.12.**

    !!! warning "Avoid the very newest Python"
        The embeddings stage depends on `onnxruntime`, which lags a few months
        behind brand-new Python releases. Installing on Python 3.13/3.14 can
        fail with `No matching distribution found for onnxruntime`. **Python
        3.12 is the recommended and tested version.**

- **~1 GB of disk** for the local models (downloaded on first use, not at install time).

## Install from PyPI

RAGMill ships as a single package with **opt-in extras**. The core install has
**zero dependencies** and only handles plain-text ingestion + chunking. You add
extras for the capabilities you actually want.

```bash
pip install ragmill
```

### Extras

| Command | Adds | Use when |
|---|---|---|
| `pip install ragmill` | *(nothing)* | `.txt` / `.md` / `.log` / `.rst` / `.csv` / `.tsv` ingestion + chunking only |
| `pip install "ragmill[pdf]"` | `pypdf` | reading `.pdf` files |
| `pip install "ragmill[docx]"` | `python-docx` | reading `.docx` files |
| `pip install "ragmill[office]"` | `beautifulsoup4`, `striprtf`, `openpyxl`, `python-pptx` | reading `.html` / `.htm` / `.rtf` / `.xlsx` / `.pptx` files |
| `pip install "ragmill[ocr]"` | `pytesseract`, `pillow` | OCR for images (`.png` / `.jpg` / …) and scanned/image-only PDFs |
| `pip install "ragmill[embeddings]"` | `onnxruntime`, `numpy`, `tokenizers` | local embeddings + vector search |
| `pip install "ragmill[server]"` | `fastapi`, `uvicorn`, `pydantic` | the REST API |
| `pip install "ragmill[chat]"` | `llama-cpp-python` | local LLM answers (no API key) — **not in `[all]`**, see below |
| `pip install "ragmill[chat-gemini]"` | `google-genai` | Gemini as the chat backend |
| `pip install "ragmill[chat-openai]"` | `openai` | OpenAI/ChatGPT as the chat backend |
| `pip install "ragmill[pinecone]"` | `pinecone` | Pinecone cloud vector store |
| `pip install "ragmill[qdrant]"` | `qdrant-client` | Qdrant vector store |
| `pip install "ragmill[config-ui]"` | `python-dotenv` | standalone setup UI that writes `.env` |
| `pip install "ragmill[all]"` | everything above **except `chat`** | you want everything that installs from wheels |

!!! warning "`[all]` excludes the local LLM — on purpose"
    `llama-cpp-python` publishes no PyPI wheels for recent versions, so pip falls
    back to a 70 MB+ source archive that vendors llama.cpp. Building it needs a
    C++ toolchain, and on Windows the vendored tree exceeds the 260-character
    `MAX_PATH` limit while unpacking, so the install dies before anything is
    installed:

    ```
    ERROR: Could not install packages due to an OSError: [Errno 2]
    No such file or directory: 'C:\\Users\\...\\vendor\\llama.cpp\\tools\\ui\\...'
    ```

    Because it sits in its own extra, `pip install "ragmill[all]"` resolves to
    wheels only, on every platform.

    **To get the local LLM anyway**, run `ragmill setup-chat`, which fetches a
    prebuilt wheel — no compiler and no extraction of the vendored tree:

    ```bash
    pip install "ragmill[all]"
    ragmill setup-chat
    ```

    It shows the package, the index, and the pip command, and asks before
    installing. The equivalent by hand:

    ```bash
    pip install llama-cpp-python \
      --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
      --only-binary llama-cpp-python
    ```

    `--only-binary` is required — without it pip picks PyPI's newer sdist-only
    release over the wheels on that index and compiles it.

    Prefer to build from source? You need CMake plus a C++ compiler (on Windows,
    the Visual Studio Build Tools "Desktop development with C++" workload), and
    long paths enabled:

    ```powershell
    # PowerShell as Administrator, then reboot
    New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
      -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force
    ```

    Or sidestep the local model entirely — `[chat-gemini]` and `[chat-openai]`
    are pure-Python and are both included in `[all]`.

!!! tip "Combine extras"
    Extras compose: `pip install "ragmill[embeddings,server,pdf]"` gets you
    exactly search + REST API + PDF support and nothing else.

!!! note "Quote the brackets"
    `zsh` (the default macOS shell) treats `[ ]` as a glob. Always quote the
    package spec: `pip install "ragmill[all]"`.

!!! warning "OCR needs system binaries"
    The `ocr` extra installs the Python side only. You also need the
    `tesseract` binary on `PATH` (`brew install tesseract` /
    `apt-get install tesseract-ocr`), plus `pdftoppm` from poppler
    (`brew install poppler` / `apt-get install poppler-utils`) to OCR scanned
    PDFs. OCR is English-only by default.

## What gets installed (and what doesn't)

`pip install` installs **only the Python package** (`ragmill/…`) plus the
dependencies for the extras you chose. It does **not**:

- download any model weights — those are fetched **lazily, on first use**, and
  cached in `~/.cache/ragmill/models/`:
    - the embedding model `Xenova/all-MiniLM-L6-v2` (~22 MB), on your first `embed()`
    - the local chat model `Qwen2.5-1.5B-Instruct` GGUF (~1 GB), on your first chat
- install the repo's `tests/`, `docs/`, `Dockerfile`, or shell scripts — those
  live in the source repository, not the distributed package.

So the first `ragmill sync` or `ragmill chat` after a fresh install is slower —
that's the one-time model download, not the pipeline itself.

## Install for development

Clone the repo and install in editable mode with the `dev` extra (adds
`pytest`, `black`, `mypy`, and test fixtures):

```bash
git clone https://github.com/Abdullahbinaqeel/RAGMill.git
cd RAGMill
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest            # runs the suite (integration tests are skipped by default)
```

## Verify the install

```bash
ragmill --help
python -c "import ragmill; print(ragmill.__name__, 'ok')"
```
