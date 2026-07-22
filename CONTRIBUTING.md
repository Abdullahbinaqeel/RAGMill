# Contributing to RAGMill

Thanks for your interest in improving RAGMill! This guide covers everything you
need to get set up, make a change, and open a pull request. For the same
information rendered on the docs site, see
[the online contributing guide](https://abdullahbinaqeel.github.io/RAGMill/contributing/).

By participating in this project you agree to abide by our
[Code of Conduct](CODE_OF_CONDUCT.md).

---

## Development setup

```bash
git clone https://github.com/Abdullahbinaqeel/RAGMill.git
cd RAGMill
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Use **Python 3.12**. `onnxruntime` (needed by the embeddings extra) does not
publish wheels for the very newest Python releases immediately, so 3.12 is the
most reliable version for the full development install.

---

## Running the checks

Every pull request must pass the same gates CI runs:

```bash
pytest                                   # unit tests (integration skipped by default)
pytest -m integration                    # requires a live Qdrant/Pinecone (see below)
black --check src/ tests/                # formatting
mypy src/ragmill/ --ignore-missing-imports --no-strict-optional   # types
```

- Unit tests must stay green and coverage must remain **≥ 75 %**
  (`--cov-fail-under=75`, enforced in CI).
- Tests that download the embedding model self-skip when there is no network.
- Integration tests need `RAGMILL_QDRANT_URL` (and optionally
  `RAGMILL_QDRANT_API_KEY`) or `PINECONE_API_KEY`. Locally, start Qdrant with
  `docker compose up -d qdrant`.

Format your code before committing:

```bash
black src/ tests/
```

---

## Coding guidelines

- **Keep new dependencies optional.** Add them to the correct extra in
  `pyproject.toml` and import them lazily (inside the function that needs them),
  so `import ragmill` stays zero-dependency and each extra fails with a clear,
  copy-pasteable install hint.
- **Test every behavior change.** Bug fixes get a regression test — no
  exceptions.
- **Match the surrounding style.** Comments explain *why*, not *what*.
- **No hardcoded secrets.** Read credentials from env vars / `.env` only.

### Adding a vector-store backend

Implement the `BaseVectorStore` interface (`src/ragmill/vector_store.py`) and
register it in `store_from_config()`. `pinecone_store.py` and `qdrant_store.py`
are good templates. Add both mocked unit tests and integration tests marked
`@pytest.mark.integration`.

---

## Commit & pull-request process

1. Branch off `main` and keep each change focused.
2. Write commit messages in
   [Conventional Commits](https://www.conventionalcommits.org/) style:
   `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`, `perf:`.
3. Ensure `pytest`, `black --check`, and `mypy` all pass. If you touched docs,
   `mkdocs build --strict` must also be clean.
4. Fill out the pull-request template and link any related issue.
5. A maintainer will review; please be responsive to feedback.

---

## Reporting bugs & requesting features

Open an issue using the appropriate
[issue template](https://github.com/Abdullahbinaqeel/RAGMill/issues/new/choose).
For **security vulnerabilities**, do **not** open a public issue — follow
[SECURITY.md](SECURITY.md) instead.
