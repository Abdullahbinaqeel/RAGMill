#!/bin/sh
set -e

# If the first argument is "serve", sync /docs first then serve.
if [ "$1" = "serve" ]; then
  echo "==> Syncing /docs into $RAGMILL_SQLITE_PATH ..."
  ragmill sync /docs 2>/dev/null || echo "==> No /docs to sync (empty or missing)"
  echo "==> Starting RAGMill server ..."
fi

exec "$@"
