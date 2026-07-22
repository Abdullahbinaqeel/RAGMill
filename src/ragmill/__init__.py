from importlib.metadata import PackageNotFoundError, version as _version

from ragmill.engine import RAGEngine

try:
    __version__ = _version("ragmill")
except PackageNotFoundError:  # running from a source checkout that isn't installed
    __version__ = "0.0.0+unknown"

__all__ = ["RAGEngine"]
