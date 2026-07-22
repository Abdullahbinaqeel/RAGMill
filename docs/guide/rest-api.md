# REST API

RAGMill ships a FastAPI server so any language or frontend can use it over HTTP.

```bash
pip install "ragmill[server]"
ragmill serve
# or: uvicorn ragmill.server:app --host 0.0.0.0 --port 8000
```

Once running:

- **Chat UI** — `http://localhost:8000/` (a minimal terminal-style box for testing)
- **Interactive OpenAPI docs** — `http://localhost:8000/docs` (Swagger UI, auto-generated)
- **Health** — `http://localhost:8000/health`

The server reads the same [configuration](configuration.md) env vars, so it uses
whatever store and chat backend you've set.

## Endpoints

| Method | Path | Body | Description |
|---|---|---|---|
| `POST` | `/ingest` | `{directory}` | Full ingest + embed a directory |
| `POST` | `/sync` | `{directory}` | Incremental sync a directory |
| `POST` | `/search` | `{query, top_k, ...filters}` | Semantic search |
| `POST` | `/chat` | `{query, top_k, ...filters}` | Grounded answer + sources |
| `GET` | `/count` | — | Number of stored chunks |
| `POST` | `/export` | — | Export store to a JSONL file on the server |
| `POST` | `/import` | multipart file | Import a JSONL file |
| `GET` | `/health` | — | `{status, store_type, chunk_count}` |
| `GET` | `/` | — | Browser chat UI |

Filters accepted by `/search` and `/chat`: `filename`, `source_file`,
`modified_after`, `modified_before`.

## Examples

### Sync a folder

```bash
curl -X POST http://localhost:8000/sync \
  -H 'Content-Type: application/json' \
  -d '{"directory": "/data/docs"}'
# {"added": 12, "updated": 0, "skipped": 88, "deleted": 1}
```

### Search

```bash
curl -X POST http://localhost:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "refund policy", "top_k": 3}'
```

### Chat (grounded answer)

```bash
curl -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"query": "what is the refund window?", "top_k": 5}'
# {"answer": "...", "sources": [{"filename": "policy.pdf", "score": 0.71, ...}]}
```

## Calling it from a frontend

Because it's plain JSON over HTTP, any client works. A TypeScript example:

```ts
async function ask(query: string) {
  const res = await fetch("http://localhost:8000/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, top_k: 5 }),
  });
  if (!res.ok) throw new Error(`RAGMill ${res.status}: ${await res.text()}`);
  return (await res.json()) as { answer: string; sources: unknown[] };
}
```

!!! warning "Deploying beyond localhost"
    `/ingest` and `/sync` take a **server-side** directory path and there's no
    auth built in. If you expose the API beyond localhost, put it behind your
    own auth/gateway and don't accept untrusted directory paths. See
    [Docker](../deployment.md).
