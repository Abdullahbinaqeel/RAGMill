# Contributing

Thanks for your interest in improving RAGMill!

## Development setup

```bash
git clone https://github.com/Abdullahbinaqeel/RAGMill.git
cd RAGMill
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Use **Python 3.12** — `onnxruntime` doesn't publish wheels for the very newest
Python releases right away.

## Running tests

```bash
pytest                    # unit tests (integration tests skipped by default)
pytest -m integration     # tests needing a live Qdrant/Pinecone
pytest tests/test_sync.py -v
```

Tests that need to download the embedding model skip themselves automatically
when there's no network.

## Code style

```bash
black src tests           # format
mypy src                  # type-check
```

- Keep new dependencies **optional** — add them to the right extra in
  `pyproject.toml`, and import them lazily (inside functions), so
  `import ragmill` stays zero-dependency.
- Add or update tests for any behavior change; bug fixes get a regression test.

## Adding a vector-store backend

Implement `BaseVectorStore` (see `vector_store.py`) and register it in
`store_from_config()`. The existing `pinecone_store.py` / `qdrant_store.py` are
good templates.

## Building the docs

```bash
pip install "ragmill[docs]"
mkdocs serve              # preview at http://127.0.0.1:8000
mkdocs build --strict     # verify no broken links before pushing
```

See [Publishing these docs](publishing-docs.md) for the deploy pipeline.

## Pull requests

- Branch off `main`, keep changes focused.
- Use [Conventional Commits](https://www.conventionalcommits.org/) for messages
  (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`).
- Make sure `pytest`, `black`, and `mypy` pass, and `mkdocs build --strict` is
  clean if you touched docs.
