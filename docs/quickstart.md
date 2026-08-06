# Quickstart

This walks you from an empty folder to semantic search and grounded Q&A in a
few minutes. It assumes `pip install "ragmill[all]"`.

## 1. Point it at a folder

Any directory of `.txt`, `.md`, `.log`, `.rst`, `.csv`, `.tsv`, `.pdf`,
`.docx`, `.html`, `.rtf`, `.xlsx`, `.pptx`, or image files (`.png`, `.jpg`,
… via OCR) works. For a first run, make one:

```bash
mkdir my_docs
echo "RAGMill turns a folder of documents into a searchable knowledge base." > my_docs/about.txt
```

## 2. Index it

=== "CLI"

    ```bash
    ragmill sync ./my_docs
    # Synced ./my_docs: {'added': 1, 'updated': 0, 'skipped': 0, 'deleted': 0}
    ```

=== "Python"

    ```python
    from ragmill import RAGEngine, SQLiteVectorStore
    from ragmill.embeddings import EmbeddingModel
    from ragmill.sync import sync_directory

    engine = RAGEngine(chunk_size=500, overlap=50)
    model  = EmbeddingModel()             # downloads the model once, then offline
    store  = SQLiteVectorStore("kb.db")   # a local file; use ":memory:" to not persist

    print(sync_directory("./my_docs", engine, model, store))
    ```

`sync` is incremental: run it again and unchanged files are skipped, changed
files are re-embedded, and files deleted from disk have their chunks removed.

!!! info "Where does the CLI store data?"
    By default the CLI uses an in-memory store, so set a path to persist:
    ```bash
    export RAGMILL_SQLITE_PATH=./ragmill.db
    ```
    See [Configuration](guide/configuration.md).

## 3. Search

=== "CLI"

    ```bash
    ragmill search "what is ragmill?" --top-k 3
    ```

=== "Python"

    ```python
    qvec = model.embed(["what is ragmill?"])[0]
    for hit in store.search(qvec, top_k=3):
        print(round(hit["score"], 3), hit["metadata"]["filename"], "->", hit["content"][:80])
    ```

Results are ranked by **meaning**, not keywords — a query never has to share
words with the text it matches.

## 4. Ask questions (RAG)

`chat` retrieves the most relevant chunks and asks an LLM to answer using only
those chunks, citing the source file.

=== "CLI (local model, no key)"

    The local model is a one-time install — it is not part of `[all]`, because
    it has no PyPI wheels ([why](installation.md)):

    ```bash
    ragmill setup-chat
    ```

    Then:

    ```bash
    ragmill chat
    # you> what does ragmill do?
    # ragmill> RAGMill turns a folder of documents into a searchable knowledge base [about.txt].
    ```

=== "Python"

    ```python
    from ragmill.chat import generate_answer

    qvec = model.embed(["what does ragmill do?"])[0]
    results = store.search(qvec, top_k=5)
    print(generate_answer("what does ragmill do?", results))
    ```

The first chat call downloads the local model (~1 GB). To use a hosted model
instead, see [Chat & answer generation](guide/chat.md).

## 5. Serve it (optional)

Expose the whole thing as a REST API plus a tiny browser chat box:

```bash
ragmill serve
# → http://localhost:8000  (chat UI at /, OpenAPI docs at /docs)
```

See the [REST API guide](guide/rest-api.md).

## The end-to-end script

```python
from ragmill import RAGEngine, SQLiteVectorStore
from ragmill.embeddings import EmbeddingModel
from ragmill.sync import sync_directory
from ragmill.chat import generate_answer

engine = RAGEngine(chunk_size=500, overlap=50)
model  = EmbeddingModel()
store  = SQLiteVectorStore("kb.db")

sync_directory("./my_docs", engine, model, store)      # index (incremental)

question = "what does ragmill do?"
qvec = model.embed([question])[0]
hits = store.search(qvec, top_k=5)                      # retrieve
print(generate_answer(question, hits))                  # answer
```

Next: **[How it works](concepts.md)** to understand each stage, or **[Use in
your project](integration.md)** to embed RAGMill in your own app.
