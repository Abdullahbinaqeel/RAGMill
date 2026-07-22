# Chat & answer generation

RAGMill's chat feature is **retrieval-augmented generation**: it searches your
store for the most relevant chunks, then asks an LLM to answer using *only*
those chunks, citing the source filename. This keeps answers grounded in your
documents instead of the model's memory.

The single entry point is `generate_answer(query, chunks)`. The backend is
chosen at call time from `RAGMILL_CHAT_BACKEND`, so switching providers is an
env-var change — no code change.

## Backends

| Backend | Extra | Needs a key? | Notes |
|---|---|---|---|
| `local` *(default)* | `ragmill[chat]` | **No** | `Qwen2.5-1.5B-Instruct` via `llama-cpp-python`. Downloads once (~1 GB) to `~/.cache/ragmill/models`, then fully offline. |
| `gemini` | `ragmill[chat-gemini]` | `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) | Google Gemini API. Best quality when online. |
| `openai` | `ragmill[chat-openai]` | `OPENAI_API_KEY` | OpenAI Chat Completions (ChatGPT). |

!!! info "Nothing is fine-tuned"
    Both local models are off-the-shelf pretrained models — RAGMill does no
    fine-tuning. Quality comes from retrieval (feeding the right chunks), not
    from a custom model.

## Local (default, offline)

```bash
pip install "ragmill[chat]"
ragmill chat
```

First call downloads the GGUF model; subsequent calls are offline. Override the
model with `RAGMILL_CHAT_MODEL_REPO` / `RAGMILL_CHAT_MODEL_FILE`, and the context
window with `RAGMILL_CHAT_N_CTX`.

## Gemini

```bash
pip install "ragmill[chat-gemini]"
export RAGMILL_CHAT_BACKEND=gemini
export GEMINI_API_KEY=your-key
ragmill chat
```

## OpenAI / ChatGPT

```bash
pip install "ragmill[chat-openai]"
export RAGMILL_CHAT_BACKEND=openai
export OPENAI_API_KEY=your-key
ragmill chat
```

## From Python

`generate_answer` takes a query and the list of chunks returned by
`store.search()`:

```python
from ragmill.embeddings import EmbeddingModel
from ragmill.vector_store import SQLiteVectorStore
from ragmill.chat import generate_answer

model = EmbeddingModel()
store = SQLiteVectorStore("kb.db")

question = "what is the refund window?"
qvec = model.embed([question])[0]
chunks = store.search(qvec, top_k=5)

answer = generate_answer(question, chunks)   # backend from RAGMILL_CHAT_BACKEND
print(answer)
```

## The system prompt

All backends share one instruction: answer *only* from the provided context,
cite the source filename per claim (e.g. `[report.pdf]`), and say so plainly
when the context doesn't contain the answer rather than guessing. This is what
makes answers auditable — every claim points back to a file.
