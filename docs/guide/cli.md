# CLI reference

Installing RAGMill puts a `ragmill` command on your PATH (via the
`project.scripts` entry point). Every command reads its configuration from
environment variables or a `.env` file — see [Configuration](configuration.md).

```bash
ragmill <command> [args]
```

| Command | Description |
|---|---|
| `ragmill ingest <dir>` | Full ingest + embed + store (no change detection — re-runs duplicate data) |
| `ragmill sync <dir>` | Incremental sync: add new, update changed, delete removed |
| `ragmill search <query> [--top-k N]` | Semantic search over stored chunks |
| `ragmill chat [--top-k N]` | Interactive terminal Q&A over stored chunks |
| `ragmill count` | Print the number of stored chunks |
| `ragmill serve [--reload]` | Start the FastAPI REST server |
| `ragmill export <path.jsonl>` | Export the whole store to JSONL |
| `ragmill import <path.jsonl>` | Import JSONL into the store |
| `ragmill configure` | Launch the standalone setup UI that writes `.env` |
| `ragmill --version` | Print the installed RAGMill version and exit |

## Checking the installed version

```bash
ragmill --version          # e.g. "ragmill 0.4.1"
```

Or from Python:

```python
import ragmill
print(ragmill.__version__)
```

## Common flows

### Index and persist a folder

```bash
export RAGMILL_SQLITE_PATH=./ragmill.db     # otherwise the store is in-memory
ragmill sync ./my_docs
ragmill count
```

### Search

```bash
ragmill search "termination clause" --top-k 5
```

### Chat in the terminal

```bash
ragmill chat --top-k 5
# you> summarize the refund policy
# ragmill> Refunds are issued within 14 days of purchase. ...
#          Sources: policy.pdf
# (Ctrl+D or type 'exit' to quit)
```

### Serve the API

```bash
ragmill serve                 # binds RAGMILL_HOST:RAGMILL_PORT (default 0.0.0.0:8000)
ragmill serve --reload        # dev mode: hot-reload on code changes
```

## `ingest` vs `sync`

- **`ingest`** always processes and inserts everything. Running it twice on the
  same folder stores every chunk twice.
- **`sync`** tracks per-file hashes and only touches what changed. **Prefer
  `sync` for anything you re-run.** `ingest` exists for one-shot loads.

## Notes

- `search` and `chat` embed your query with the same model used at index time,
  so they trigger the model download on first use if it isn't cached yet.
- `chat`'s backend (local / Gemini / OpenAI) is chosen by `RAGMILL_CHAT_BACKEND`
  — see [Chat & answer generation](chat.md).
