import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class RAGMillConfig:
    chunk_size: int = 500
    overlap: int = 50
    embedding_model: str = "Xenova/all-MiniLM-L6-v2"
    embedding_dim: int = 384

    store_type: str = "sqlite"

    sqlite_path: Optional[str] = "./ragmill.db"

    pinecone_api_key: Optional[str] = None
    pinecone_environment: Optional[str] = None
    pinecone_index_name: str = "ragmill"

    qdrant_url: Optional[str] = None
    qdrant_api_key: Optional[str] = None
    qdrant_collection_name: str = "ragmill"
    qdrant_prefer_grpc: bool = False

    chat_backend: str = "local"

    chat_model_repo: str = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
    chat_model_file: str = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
    chat_n_ctx: int = 4096

    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-flash-latest"

    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o-mini"

    server_host: str = "127.0.0.1"
    server_port: int = 8000
    server_api_key: Optional[str] = None
    server_allowed_roots: Optional[str] = None

    @classmethod
    def from_env(cls) -> "RAGMillConfig":
        try:
            from dotenv import find_dotenv, load_dotenv

            # usecwd=True: search from the caller's current working directory
            # (e.g. wherever `ragmill serve` is run), not from inside this
            # installed package — find_dotenv()'s default without usecwd.
            load_dotenv(find_dotenv(usecwd=True))
        except ImportError:
            pass

        return cls(
            chunk_size=int(os.getenv("RAGMILL_CHUNK_SIZE", "500")),
            overlap=int(os.getenv("RAGMILL_OVERLAP", "50")),
            embedding_model=os.getenv("RAGMILL_EMBEDDING_MODEL", "Xenova/all-MiniLM-L6-v2"),
            embedding_dim=int(os.getenv("RAGMILL_EMBEDDING_DIM", "384")),
            store_type=os.getenv("RAGMILL_STORE_TYPE", "sqlite"),
            sqlite_path=os.getenv("RAGMILL_SQLITE_PATH") or "./ragmill.db",
            pinecone_api_key=os.getenv("RAGMILL_PINECONE_API_KEY"),
            pinecone_environment=os.getenv("RAGMILL_PINECONE_ENVIRONMENT"),
            pinecone_index_name=os.getenv("RAGMILL_PINECONE_INDEX_NAME", "ragmill"),
            qdrant_url=os.getenv("RAGMILL_QDRANT_URL"),
            qdrant_api_key=os.getenv("RAGMILL_QDRANT_API_KEY"),
            qdrant_collection_name=os.getenv("RAGMILL_QDRANT_COLLECTION_NAME", "ragmill"),
            qdrant_prefer_grpc=os.getenv("RAGMILL_QDRANT_PREFER_GRPC", "").lower()
            in ("1", "true", "yes"),
            chat_backend=os.getenv("RAGMILL_CHAT_BACKEND", "local"),
            chat_model_repo=os.getenv("RAGMILL_CHAT_MODEL_REPO", "Qwen/Qwen2.5-1.5B-Instruct-GGUF"),
            chat_model_file=os.getenv(
                "RAGMILL_CHAT_MODEL_FILE", "qwen2.5-1.5b-instruct-q4_k_m.gguf"
            ),
            chat_n_ctx=int(os.getenv("RAGMILL_CHAT_N_CTX", "4096")),
            gemini_api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
            gemini_model=os.getenv("RAGMILL_GEMINI_MODEL", "gemini-flash-latest"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_model=os.getenv("RAGMILL_OPENAI_MODEL", "gpt-4o-mini"),
            server_host=os.getenv("RAGMILL_HOST", "127.0.0.1"),
            server_port=int(os.getenv("RAGMILL_PORT", "8000")),
            server_api_key=os.getenv("RAGMILL_API_KEY"),
            server_allowed_roots=os.getenv("RAGMILL_ALLOWED_ROOTS"),
        )
