"""Reusable RAG operations with grounded answers and readable citations."""

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings

from settings import CHAT_MODEL, CHROMA_DIR, COLLECTION_NAME, DEFAULT_RETRIEVAL_K, EMBEDDING_MODEL

NO_CONTEXT_ANSWER = "I don't have enough information in the provided documents."
PROMPT = ChatPromptTemplate.from_template(
    """You are an Azure technical assistant. Answer using only the context below.
If the context does not answer the question, reply exactly with:
{no_context_answer}

Context:
{context}

Question: {question}
"""
)


def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=EMBEDDING_MODEL)


def get_vector_store() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=get_embeddings(),
    )


def format_citation(document: Document) -> str:
    """Return a compact citation from ingestion metadata."""
    parts = [document.metadata.get("source", "unknown source")]
    if document.metadata.get("page") is not None:
        parts.append(f"page {int(document.metadata['page']) + 1}")
    if document.metadata.get("chunk") is not None:
        parts.append(f"chunk {document.metadata['chunk']}")
    return ", ".join(parts)


def retrieve(question: str, k: int = DEFAULT_RETRIEVAL_K) -> list[Document]:
    return get_vector_store().similarity_search(question, k=k)


def answer_question(question: str, k: int = DEFAULT_RETRIEVAL_K) -> tuple[str, list[str]]:
    """Answer a question and return the sources used to build its context."""
    documents = retrieve(question, k)
    if not documents:
        return NO_CONTEXT_ANSWER, []
    context = "\n\n".join(document.page_content for document in documents)
    response = ChatOllama(model=CHAT_MODEL).invoke(
        PROMPT.invoke({"context": context, "question": question, "no_context_answer": NO_CONTEXT_ANSWER})
    )
    citations = list(dict.fromkeys(format_citation(document) for document in documents))
    return response.content, citations
