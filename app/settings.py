"""Shared paths and model settings for the local RAG application."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "Documents"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma"
CHAT_MODEL = "llama3.2:3b"
EMBEDDING_MODEL = "nomic-embed-text"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
DEFAULT_RETRIEVAL_K = 4
COLLECTION_NAME = "documents"
