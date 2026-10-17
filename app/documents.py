"""Document discovery and chunking used by the ingestion command."""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from settings import CHUNK_OVERLAP, CHUNK_SIZE, DOCUMENTS_DIR


def load_documents(
    directory: Path = DOCUMENTS_DIR, extensions: set[str] | None = None
) -> list[Document]:
    """Load supported files and attach stable source metadata to every page."""
    if not directory.exists():
        raise FileNotFoundError(f"Document directory does not exist: {directory}")

    supported_extensions = extensions or {".txt", ".pdf"}
    documents: list[Document] = []
    for path in sorted(directory.rglob("*")):
        if path.suffix.lower() not in supported_extensions:
            continue
        if path.suffix.lower() == ".txt":
            loaded = TextLoader(str(path), encoding="utf-8").load()
        elif path.suffix.lower() == ".pdf":
            loaded = PyPDFLoader(str(path)).load()
        for document in loaded:
            document.metadata["source"] = path.name
            document.metadata["source_path"] = str(path.relative_to(directory.parent))
        documents.extend(loaded)

    if not documents:
        raise ValueError(f"No .txt or .pdf documents found in {directory}")
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """Split documents and give each chunk a stable per-source identifier."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(documents)
    counters: dict[str, int] = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        counters[source] = counters.get(source, 0) + 1
        chunk.metadata["chunk"] = counters[source]
    return chunks
