"""Search and retrieval helpers for official Microsoft Learn documentation."""

from functools import lru_cache
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from settings import CHUNK_OVERLAP, CHUNK_SIZE

LEARN_SEARCH_URL = "https://learn.microsoft.com/api/search"


def _load_source(url: str, timeout: int) -> Document:
    response = requests.get(
        url,
        headers={"User-Agent": "azure-learn-assistant/1.0"},
        timeout=timeout,
    )
    response.raise_for_status()
    if urlparse(response.url).hostname != "learn.microsoft.com":
        raise ValueError("redirected outside learn.microsoft.com")

    soup = BeautifulSoup(response.text, "html.parser")
    article = soup.select_one("main article") or soup.select_one("article")
    if article is None:
        main = soup.select_one("main")
        content_sections = main.select(".content") if main else []
        article = (
            max(content_sections, key=lambda section: len(section.get_text(strip=True)))
            if content_sections
            else main
        )
    if article is None:
        raise ValueError("Microsoft Learn article content was not found")

    for element in article.select("script, style, nav, aside, footer, .feedback-section"):
        element.decompose()
    content = article.get_text("\n", strip=True)
    if not content:
        raise ValueError("Microsoft Learn article was empty")

    title_element = soup.find("h1")
    title = title_element.get_text(" ", strip=True) if title_element else url.rsplit("/", 1)[-1]
    return Document(
        page_content=content[:60_000],
        metadata={"source": title, "url": response.url},
    )


def load_official_source(url: str, timeout: int = 20) -> Document:
    """Fetch a page only from the official Microsoft Learn host."""
    parsed_url = urlparse(url)
    if parsed_url.scheme != "https" or parsed_url.hostname != "learn.microsoft.com":
        raise ValueError("URL must use the official learn.microsoft.com host.")
    return _load_source(url, timeout)


@lru_cache(maxsize=128)
def search_official_sources(
    question: str,
    limit: int = 5,
    timeout: int = 20,
) -> tuple[tuple[str, str, str], ...]:
    """Search Microsoft Learn for documentation matching an Azure question."""
    query = question.strip()
    if not query:
        raise ValueError("Search question cannot be empty.")
    if "azure" not in query.lower():
        query = f"Azure {query}"

    response = requests.get(
        LEARN_SEARCH_URL,
        params={"search": query, "locale": "en-us", "$top": max(limit * 3, 10)},
        headers={"User-Agent": "azure-learn-assistant/1.0"},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError("Microsoft Learn search returned an unexpected response.")

    sources: list[tuple[str, str, str]] = []
    seen_urls: set[str] = set()
    for result in results:
        if not isinstance(result, dict) or result.get("category") != "Documentation":
            continue
        url = result.get("url")
        if not isinstance(url, str):
            continue
        parsed_url = urlparse(url)
        if parsed_url.scheme != "https" or parsed_url.hostname != "learn.microsoft.com":
            continue
        canonical_url = parsed_url._replace(query="", fragment="").geturl()
        if canonical_url in seen_urls:
            continue
        seen_urls.add(canonical_url)
        title = result.get("title")
        description = result.get("description")
        sources.append(
            (
                title.strip() if isinstance(title, str) and title.strip() else canonical_url,
                canonical_url,
                description.strip() if isinstance(description, str) else "",
            )
        )
        if len(sources) >= limit:
            break
    return tuple(sources)


def clear_source_search_cache() -> None:
    """Clear cached Microsoft Learn search results."""
    search_official_sources.cache_clear()


def split_official_sources(documents: list[Document]) -> list[Document]:
    """Chunk Microsoft Learn pages while preserving their citation metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    counters: dict[str, int] = {}
    for chunk in chunks:
        source = str(chunk.metadata.get("source", "unknown"))
        counters[source] = counters.get(source, 0) + 1
        chunk.metadata["chunk"] = counters[source]
    return chunks
