"""Ingest every supported document into the local Chroma vector database."""

from langchain_chroma import Chroma

from documents import load_documents, split_documents
from rag import get_embeddings
from settings import CHROMA_DIR, COLLECTION_NAME


def main() -> None:
    documents = load_documents()
    chunks = split_documents(documents)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=get_embeddings(),
    )
    try:
        store.delete_collection()
    except ValueError:
        # The collection does not exist on the first ingestion.
        pass
    Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )
    print(f"Indexed {len(chunks)} chunks from {len(documents)} document pages.")


if __name__ == "__main__":
    main()
