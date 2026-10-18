"""Search and retrieval helpers for official Microsoft Learn documentation."""

import re
from functools import lru_cache
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from settings import CHUNK_OVERLAP, CHUNK_SIZE

LEARN_SEARCH_URL = "https://learn.microsoft.com/api/search"
SEARCH_STOP_WORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "azure",
    "can",
    "could",
    "describe",
    "does",
    "explain",
    "for",
    "from",
    "give",
    "how",
    "i",
    "in",
    "introduction",
    "is",
    "it",
    "me",
    "of",
    "on",
    "or",
    "overview",
    "please",
    "tell",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "why",
    "with",
}
INTRO_INTENT_TERMS = {"introduce", "introduction", "intro", "overview", "basics"}
SOURCE_VARIANT_TERMS = {"cassandra", "gremlin", "mongodb"}
GENERAL_QUESTION_TERMS = {"describe", "explain", "how", "what"}
TASK_INTENT_TERMS = {"configure", "create", "deploy", "enable", "install", "set"}


def _search_queries(question: str, query: str) -> tuple[str, ...]:
    """Use focused Learn searches for the assistant's broad, common Azure questions."""
    terms = _search_terms(question)
    if (
        {"compare", "versus", "vs"} & terms
        and {"sql", "cosmos"} <= terms
    ):
        return ("Azure SQL Database overview", "Azure Cosmos DB overview")
    if {"private", "endpoint"} <= terms:
        return ("Azure Private Link private endpoint overview",)
    if {"monitor", "application"} <= terms:
        return ("Azure Monitor application monitoring overview",)
    if {"kubernetes", "workload"} <= terms and {"secure", "security"} & terms:
        return ("Azure Kubernetes workload security overview",)
    return (query,)


def normalize_search_term(term: str) -> str:
    if term in {"cosmos", "kubernetes", "series", "business", "analysis", "access"}:
        return term
    if len(term) > 4 and term.endswith("ies"):
        return f"{term[:-3]}y"
    if len(term) > 3 and term.endswith("s") and not term.endswith("ss"):
        return term[:-1]
    return term


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


def _raw_terms(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _search_terms(text: str) -> set[str]:
    terms = {normalize_search_term(term) for term in _raw_terms(text)}
    terms.difference_update(SEARCH_STOP_WORDS)
    return terms


def _is_overview_path(path: str) -> bool:
    last_segment = path.rstrip("/").rsplit("/", 1)[-1]
    return last_segment.endswith(("overview", "introduction"))


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

    query_terms = _search_terms(query)
    search_queries = _search_queries(question, query)
    intro_intent = bool(_raw_terms(query) & INTRO_INTENT_TERMS)
    broad_intent = bool(_raw_terms(question) & GENERAL_QUESTION_TERMS)
    task_intent = bool(_raw_terms(question) & TASK_INTENT_TERMS)
    requested_result_count = min(max(limit * 6, 20), 30)
    sources: list[tuple[str, str, str, int]] = []
    seen_urls: set[str] = set()
    comparison_services = ("sql", "cosmos") if (
        {"compare", "versus", "vs"} & query_terms
        and {"sql", "cosmos"} <= query_terms
    ) else ()
    for search_query in search_queries:
        search_terms = query_terms | _search_terms(search_query)
        response = requests.get(
            LEARN_SEARCH_URL,
            params={
                "search": search_query,
                "locale": "en-us",
                "$top": requested_result_count,
            },
            headers={"User-Agent": "azure-learn-assistant/1.0"},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        results = payload.get("results")
        if not isinstance(results, list):
            raise ValueError("Microsoft Learn search returned an unexpected response.")

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
            title = result.get("title")
            description = result.get("description")
            title = title.strip() if isinstance(title, str) and title.strip() else canonical_url
            description = description.strip() if isinstance(description, str) else ""
            title_terms = _search_terms(title)
            description_terms = _search_terms(description)
            path_terms = _search_terms(parsed_url.path.replace("/", " ").replace("-", " "))
            path = parsed_url.path.lower()
            relevance = (
                5 * len(search_terms & title_terms)
                + 3 * len(search_terms & path_terms)
                + len(search_terms & description_terms)
            )
            if intro_intent and _is_overview_path(path):
                relevance += 20
            if comparison_services and "overview" in path:
                relevance += 20
            if broad_intent and not task_intent:
                if _is_overview_path(path):
                    relevance += 12
                if any(segment in path for segment in ("/how-to-", "/quickstart", "/tutorial")):
                    relevance -= 10
            if {"private", "endpoint"} <= query_terms and "/azure/private-link/" in path:
                relevance += 30
            if not task_intent and any(
                segment in path for segment in ("/deploy-", "/create-", "/enable-")
            ):
                relevance -= 8
            is_hub = parsed_url.path.endswith("/") or title.lower().endswith("documentation")
            if is_hub:
                relevance -= 8
            if intro_intent:
                unrequested_variants = (
                    _search_terms(path) & (SOURCE_VARIANT_TERMS - query_terms)
                )
                relevance -= 20 * len(unrequested_variants)
            seen_urls.add(canonical_url)
            sources.append((title, canonical_url, description, relevance))

    relevant_sources = [
        source
        for source in sources
        if source[3] > 0
        and (
            not query_terms
            or query_terms
            & (
                _search_terms(source[0])
                | _search_terms(urlparse(source[1]).path)
            )
        )
    ]
    if intro_intent:
        requested_variants = SOURCE_VARIANT_TERMS & query_terms
        general_sources = [
            source
            for source in relevant_sources
            if not (
                _search_terms(urlparse(source[1]).path)
                & (SOURCE_VARIANT_TERMS - requested_variants)
            )
        ]
        if general_sources:
            relevant_sources = general_sources
    substantive_sources = [
        source
        for source in relevant_sources
        if not urlparse(source[1]).path.endswith("/")
    ]
    ranked_sources = substantive_sources or relevant_sources or sources
    ranked_sources.sort(key=lambda source: source[3], reverse=True)
    if comparison_services and len(ranked_sources) > 1:
        balanced_sources = []
        for service in comparison_services:
            matching = next(
                (
                    source
                    for source in ranked_sources
                    if service in _search_terms(urlparse(source[1]).path)
                ),
                None,
            )
            if matching is not None:
                balanced_sources.append(matching)
        ranked_sources = balanced_sources + [
            source for source in ranked_sources if source not in balanced_sources
        ]
    return tuple(
        (title, url, description)
        for title, url, description, _ in ranked_sources[:limit]
    )


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
