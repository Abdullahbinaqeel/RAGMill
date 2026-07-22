# Docker

RAGMill ships a `Dockerfile` and `docker-compose.yml` for running the REST API
as a container, with either the local SQLite backend or a Qdrant sidecar.

## Compose profiles

```bash
# SQLite backend (self-contained)
docker compose --profile sqlite up

# Qdrant backend (also starts a Qdrant container)
docker compose --profile qdrant up
```

The API is then available on `http://localhost:8000` (chat UI at `/`, OpenAPI
docs at `/docs`).

## Persisting data

The SQLite database and the model cache should live on mounted volumes so they
survive container restarts:

- **Store:** set `RAGMILL_SQLITE_PATH` to a path on a mounted volume (e.g.
  `/data/ragmill.db`).
- **Models:** mount a volume at `~/.cache/ragmill/models` (or set a cache dir) so
  the ~1 GB of models download only once, not on every rebuild.

Configure everything through the [environment variables](guide/configuration.md)
— pass them via compose `environment:` or an `env_file:`.

## Production notes

!!! warning "Add your own auth"
    The API has no built-in authentication, and `/ingest` / `/sync` accept a
    server-side directory path. Behind a public endpoint you should:

    - put it behind a reverse proxy / API gateway that handles auth and TLS
    - restrict or disable `/ingest` and `/sync`, or validate the directory
      against an allowlist
    - bind to an internal network rather than `0.0.0.0` where possible

- **First request is slow** — it triggers the one-time model download. Warm the
  container by hitting `/health` and doing one `/search` after startup, or bake
  the models into the image / a pre-populated volume.
- **Scale reads, not the store** — the SQLite store is single-file. For
  concurrent write-heavy workloads or large corpora, use the Qdrant/Pinecone
  backend instead.
