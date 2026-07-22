# ── Build stage ────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

COPY pyproject.toml README.md .
COPY src/ src/

RUN pip install --upgrade pip \
    && pip install build \
    && python -m build --wheel


# ── Runtime stage ──────────────────────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

COPY --from=builder /build/dist/*.whl /tmp/

RUN pip install --upgrade pip \
    && WHEEL="$(find /tmp -maxdepth 1 -name 'ragmill-*.whl')" \
    && pip install "${WHEEL}[all,server]" \
    && rm "${WHEEL}"

EXPOSE 8000

ENV RAGMILL_STORE_TYPE=sqlite
ENV RAGMILL_SQLITE_PATH=/data/ragmill.db

COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["serve"]
