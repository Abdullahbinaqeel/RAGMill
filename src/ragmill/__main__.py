"""
Command-line interface for RAGMill.

Usage:
    ragmill ingest <directory>          Ingest + embed + store files
    ragmill sync <directory>            Incremental sync
    ragmill search <query> [--top-k 5]  Search stored chunks
    ragmill chat [--top-k 5]            Interactive Q&A over stored chunks
    ragmill setup-chat [--yes]          Install the local chat model runtime
    ragmill count                       Show chunk count
    ragmill serve                       Start the REST API server
    ragmill export <path.jsonl>         Export store to JSONL
    ragmill import <path.jsonl>         Import JSONL into store
    ragmill configure                  Start the standalone setup UI (writes .env)
    ragmill --version                   Print the installed RAGMill version
"""

import argparse
import logging
import os
import sys

# NOTE: only import modules here that are dependency-free (no numpy/onnxruntime),
# so `ragmill --version` / `ragmill --help` work on a core-only install. Heavy
# modules (embeddings, vector_store, export, sync, chat) are imported lazily
# inside the commands that need them, and raise a clear "install ragmill[...]"
# message when the corresponding extra is missing.
from ragmill import RAGEngine, __version__
from ragmill.config import RAGMillConfig

logger = logging.getLogger(__name__)


def _get_config() -> RAGMillConfig:
    return RAGMillConfig.from_env()


def _get_store(cfg: RAGMillConfig):
    from ragmill.vector_store import store_from_config

    return store_from_config(cfg)


def cmd_ingest(args):
    from ragmill.embeddings import EmbeddingModel

    cfg = _get_config()
    store = _get_store(cfg)
    engine = RAGEngine(chunk_size=cfg.chunk_size, overlap=cfg.overlap)
    model = EmbeddingModel(model_name=cfg.embedding_model)

    chunks = engine.execute_pipeline(args.directory)
    if chunks:
        from ragmill.embeddings import DEFAULT_EMBED_BATCH

        for i in range(0, len(chunks), DEFAULT_EMBED_BATCH):
            batch = chunks[i : i + DEFAULT_EMBED_BATCH]
            vectors = model.embed([c["content"] for c in batch])
            store.add(batch, vectors)
    logger.info("Ingested %d chunks from %s", len(chunks), args.directory)


def cmd_sync(args):
    from ragmill.embeddings import EmbeddingModel
    from ragmill.sync import sync_directory

    cfg = _get_config()
    store = _get_store(cfg)
    engine = RAGEngine(chunk_size=cfg.chunk_size, overlap=cfg.overlap)
    model = EmbeddingModel(model_name=cfg.embedding_model)

    def _progress(seen: int, source_file: str) -> None:
        logger.info("  [%d] %s", seen, os.path.basename(source_file)[:60])

    result = sync_directory(args.directory, engine, model, store, progress=_progress)
    logger.info("Synced %s: %s", args.directory, result)


def cmd_search(args):
    from ragmill.embeddings import EmbeddingModel

    cfg = _get_config()
    store = _get_store(cfg)
    model = EmbeddingModel(model_name=cfg.embedding_model)

    query_vector = model.embed([args.query])[0]
    results = store.search(query_vector, top_k=args.top_k)

    if not results:
        logger.info("No results found.")
        return

    logger.info("Top %d results:\n", len(results))
    for r in results:
        logger.info(
            "  [%.4f] %s (chunk %d)",
            r["score"],
            r["metadata"]["filename"],
            r["metadata"]["chunk_index"],
        )
        logger.info("       %s...\n", r["content"][:120].replace(chr(10), " "))


def cmd_setup_chat(args):
    """Install the local chat model runtime, with consent.

    Python packaging has no post-install hook — a wheel is unpacked, never
    executed — and that is deliberate, so this cannot happen during
    `pip install`. Doing it as an explicit command also keeps the third-party
    index visible: the user is told which package comes from where and agrees
    before anything is fetched. Same shape as `playwright install` or
    `python -m spacy download`.
    """
    import subprocess
    import sys

    from ragmill.chat import LLAMA_INSTALL_ARGS, LLAMA_INSTALL_COMMAND, LLAMA_WHEEL_INDEX

    if _llama_cpp_present():
        logger.info("The local chat model is already installed. Run `ragmill chat` to use it.")
        return

    logger.info("This will install the local chat model runtime:\n")
    logger.info("  package:  llama-cpp-python (prebuilt wheel, no compiler needed)")
    logger.info("  from:     %s", LLAMA_WHEEL_INDEX)
    logger.info("  command:  %s\n", LLAMA_INSTALL_COMMAND)
    logger.info(
        "That index is maintained by the llama-cpp-python author and is not PyPI.\n"
        "It is needed because the package publishes no wheels to PyPI itself.\n"
    )

    if not args.yes:
        if not sys.stdin.isatty():
            raise SystemExit(
                "Not running interactively — re-run as `ragmill setup-chat --yes` to confirm."
            )
        try:
            reply = input("Continue? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            raise SystemExit("\nCancelled.")
        if reply not in ("y", "yes"):
            raise SystemExit("Cancelled — nothing was installed.")

    # sys.executable, not a bare `pip`: installs into the interpreter running
    # ragmill rather than whichever pip happens to be first on PATH.
    logger.info("\nInstalling…\n")
    result = subprocess.run([sys.executable, "-m", "pip", "install", *LLAMA_INSTALL_ARGS])

    if result.returncode != 0:
        raise SystemExit(
            f"\nInstall failed (pip exited {result.returncode}).\n\n"
            f"If pip reported no matching distribution, this Python "
            f"({sys.version_info.major}.{sys.version_info.minor}) has no prebuilt wheel on that "
            "index — it currently covers 3.8 to 3.13. Either use a Python in that range, or set "
            "a hosted backend instead:\n"
            "  RAGMILL_CHAT_BACKEND=gemini  (with GEMINI_API_KEY)\n"
            "  RAGMILL_CHAT_BACKEND=openai  (with OPENAI_API_KEY)"
        )

    if not _llama_cpp_present():
        raise SystemExit(
            "\npip reported success but llama_cpp still cannot be imported. "
            "If you are using a virtualenv, check that it is the active one."
        )

    logger.info("\nDone. Run `ragmill chat` — the first question downloads the model (~1.1GB).")


def _llama_cpp_present() -> bool:
    import importlib
    import importlib.util
    import sys

    # Invalidate caches: pip just wrote into site-packages inside this process's
    # lifetime, so a negative result may be stale from an earlier lookup.
    importlib.invalidate_caches()
    if "llama_cpp" in sys.modules:
        return True
    try:
        return importlib.util.find_spec("llama_cpp") is not None
    except (ImportError, ValueError):
        return False


def cmd_chat(args):
    from ragmill.chat import check_backend_available, generate_answer
    from ragmill.embeddings import EmbeddingModel

    cfg = _get_config()

    # Check before the REPL starts. Failing inside the loop would throw away a
    # question the user already typed and print a traceback for what is just a
    # missing optional install.
    try:
        check_backend_available(cfg)
    except ImportError as exc:
        raise SystemExit(f"\n{exc}\n")

    store = _get_store(cfg)
    model = EmbeddingModel(model_name=cfg.embedding_model)

    logger.info("RAGMill chat — ask a question (Ctrl+D or 'exit' to quit)\n")
    while True:
        try:
            query = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            logger.info("")
            break
        if not query or query.lower() in ("exit", "quit"):
            break

        query_vector = model.embed([query])[0]
        results = store.search(query_vector, top_k=args.top_k)
        answer = generate_answer(query, results, cfg)

        logger.info("\nragmill> %s\n", answer)
        if results:
            logger.info("sources:")
            for r in results:
                logger.info(
                    "  - %s (chunk %d)", r["metadata"]["filename"], r["metadata"]["chunk_index"]
                )
        logger.info("")


def cmd_count(args):
    cfg = _get_config()
    store = _get_store(cfg)
    logger.info("%d", store.count())


def cmd_serve(args):
    cfg = _get_config()
    os.environ.setdefault("RAGMILL_STORE_TYPE", cfg.store_type)

    logger.info("Starting RAGMill server on %s:%d", cfg.server_host, cfg.server_port)
    logger.info("  store: %s", cfg.store_type)

    import uvicorn

    uvicorn.run(
        "ragmill.server:app",
        host=cfg.server_host,
        port=cfg.server_port,
        reload=args.reload,
    )


def cmd_configure(args):
    os.environ["RAGMILL_ENV_PATH"] = args.env_path

    logger.info("Starting RAGMill setup UI on http://%s:%d", args.host, args.port)
    logger.info("  writing config to: %s", os.path.abspath(args.env_path))
    logger.info(
        "  binds to 127.0.0.1 (local only) by default — this page renders a credential form"
    )

    import uvicorn

    uvicorn.run(
        "ragmill.config_ui:config_app",
        host=args.host,
        port=args.port,
    )


def cmd_export(args):
    from ragmill.export import export_store

    cfg = _get_config()
    store = _get_store(cfg)
    written = export_store(args.path, store)
    logger.info("Exported %d records to %s", written, args.path)


def cmd_import(args):
    from ragmill.export import import_store

    cfg = _get_config()
    store = _get_store(cfg)
    imported = import_store(args.path, store)
    logger.info("Imported %d records from %s", imported, args.path)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )
    parser = argparse.ArgumentParser(prog="ragmill", description="RAGMill CLI")
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Print the installed RAGMill version and exit",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Ingest a directory")
    p_ingest.add_argument("directory")
    p_ingest.set_defaults(func=cmd_ingest)

    p_sync = sub.add_parser("sync", help="Incremental sync")
    p_sync.add_argument("directory")
    p_sync.set_defaults(func=cmd_sync)

    p_search = sub.add_parser("search", help="Search chunks")
    p_search.add_argument("query")
    p_search.add_argument("--top-k", type=int, default=5)
    p_search.set_defaults(func=cmd_search)

    p_chat = sub.add_parser("chat", help="Interactive Q&A over stored chunks")
    p_chat.add_argument("--top-k", type=int, default=5)
    p_chat.set_defaults(func=cmd_chat)

    p_setup_chat = sub.add_parser(
        "setup-chat", help="Install the local chat model runtime (prebuilt wheel)"
    )
    p_setup_chat.add_argument("--yes", action="store_true", help="Skip the confirmation prompt")
    p_setup_chat.set_defaults(func=cmd_setup_chat)

    p_count = sub.add_parser("count", help="Count chunks")
    p_count.set_defaults(func=cmd_count)

    p_serve = sub.add_parser("serve", help="Start the REST API")
    p_serve.add_argument("--reload", action="store_true", help="Hot-reload on code changes")
    p_serve.set_defaults(func=cmd_serve)

    p_export = sub.add_parser("export", help="Export store to JSONL")
    p_export.add_argument("path")
    p_export.set_defaults(func=cmd_export)

    p_import = sub.add_parser("import", help="Import JSONL into store")
    p_import.add_argument("path")
    p_import.set_defaults(func=cmd_import)

    p_configure = sub.add_parser(
        "configure", help="Start the standalone setup UI (separate server, writes .env)"
    )
    p_configure.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    p_configure.add_argument("--port", type=int, default=8090, help="Bind port (default: 8090)")
    p_configure.add_argument(
        "--env-path", default="./.env", help="Where to write config (default: ./.env)"
    )
    p_configure.set_defaults(func=cmd_configure)

    parsed = parser.parse_args()
    parsed.func(parsed)


if __name__ == "__main__":
    main()
