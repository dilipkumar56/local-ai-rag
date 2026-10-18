"""Official Microsoft Learn pages used by the AZ-104 study assistant."""

import logging
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from settings import CHUNK_OVERLAP, CHUNK_SIZE

logger = logging.getLogger(__name__)

MICROSOFT_LEARN_SOURCES = [
    "https://learn.microsoft.com/en-us/credentials/certifications/resources/study-guides/az-104",
    "https://learn.microsoft.com/en-us/entra/fundamentals/what-is-entra",
    "https://learn.microsoft.com/en-us/entra/fundamentals/how-to-create-delete-users",
    "https://learn.microsoft.com/en-us/entra/fundamentals/how-to-manage-groups",
    "https://learn.microsoft.com/en-us/entra/identity/users/groups-create-rule",
    "https://learn.microsoft.com/en-us/entra/identity/users/licensing-groups-assign",
    "https://learn.microsoft.com/en-us/azure/role-based-access-control/overview",
    "https://learn.microsoft.com/en-us/azure/role-based-access-control/role-assignments-portal",
    "https://learn.microsoft.com/en-us/entra/identity/role-based-access-control/permissions-reference",
    "https://learn.microsoft.com/en-us/azure/governance/policy/overview",
    "https://learn.microsoft.com/en-us/azure/azure-resource-manager/management/lock-resources",
    "https://learn.microsoft.com/en-us/azure/cost-management-billing/costs/tutorial-acm-create-budgets",
    "https://learn.microsoft.com/en-us/azure/storage/common/storage-account-overview",
    "https://learn.microsoft.com/en-us/azure/storage/common/storage-redundancy",
    "https://learn.microsoft.com/en-us/azure/storage/common/storage-network-security",
    "https://learn.microsoft.com/en-us/azure/storage/common/storage-sas-overview",
    "https://learn.microsoft.com/en-us/azure/storage/files/storage-files-introduction",
    "https://learn.microsoft.com/en-us/azure/storage/blobs/storage-blobs-overview",
    "https://learn.microsoft.com/en-us/azure/storage/blobs/soft-delete-blob-overview",
    "https://learn.microsoft.com/en-us/azure/storage/blobs/lifecycle-management-overview",
    "https://learn.microsoft.com/en-us/azure/storage/blobs/versioning-overview",
    "https://learn.microsoft.com/en-us/azure/azure-resource-manager/templates/overview",
    "https://learn.microsoft.com/en-us/azure/virtual-machines/overview",
    "https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/overview",
    "https://learn.microsoft.com/en-us/azure/virtual-machines/availability",
    "https://learn.microsoft.com/en-us/azure/virtual-machine-scale-sets/overview",
    "https://learn.microsoft.com/en-us/azure/container-instances/container-instances-overview",
    "https://learn.microsoft.com/en-us/azure/container-apps/overview",
    "https://learn.microsoft.com/en-us/azure/app-service/overview",
    "https://learn.microsoft.com/en-us/azure/virtual-network/virtual-networks-overview",
    "https://learn.microsoft.com/en-us/azure/virtual-network/network-security-groups-overview",
    "https://learn.microsoft.com/en-us/azure/virtual-network/virtual-network-peering-overview",
    "https://learn.microsoft.com/en-us/azure/virtual-network/virtual-networks-udr-overview",
    "https://learn.microsoft.com/en-us/azure/private-link/private-endpoint-overview",
    "https://learn.microsoft.com/en-us/azure/virtual-network/virtual-network-service-endpoints-overview",
    "https://learn.microsoft.com/en-us/azure/bastion/bastion-overview",
    "https://learn.microsoft.com/en-us/azure/load-balancer/load-balancer-overview",
    "https://learn.microsoft.com/en-us/azure/dns/dns-overview",
    "https://learn.microsoft.com/en-us/azure/azure-monitor/fundamentals/overview",
    "https://learn.microsoft.com/en-us/azure/azure-monitor/logs/log-query-overview",
    "https://learn.microsoft.com/en-us/azure/azure-monitor/alerts/alerts-overview",
    "https://learn.microsoft.com/en-us/azure/network-watcher/network-watcher-monitoring-overview",
    "https://learn.microsoft.com/en-us/azure/backup/backup-overview",
    "https://learn.microsoft.com/en-us/azure/site-recovery/site-recovery-overview",
    "https://learn.microsoft.com/en-us/azure/azure-resource-manager/management/overview",
]


def _load_source(url: str, timeout: int) -> Document:
    response = requests.get(
        url,
        headers={"User-Agent": "local-ai-rag-az104-study-assistant/1.0"},
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
    """Fetch and extract one curated Microsoft Learn page."""
    if url not in MICROSOFT_LEARN_SOURCES:
        raise ValueError("URL is not in the curated Microsoft Learn source list.")
    return _load_source(url, timeout)


def load_official_sources(timeout: int = 20) -> list[Document]:
    """Fetch and extract the current text of curated Microsoft Learn pages."""
    documents: list[Document] = []
    failures: list[str] = []

    def load_result(url: str) -> tuple[str, Document | None, str | None]:
        try:
            return url, _load_source(url, timeout), None
        except (requests.RequestException, ValueError) as error:
            return url, None, str(error)

    with ThreadPoolExecutor(max_workers=8) as executor:
        for url, document, error in executor.map(load_result, MICROSOFT_LEARN_SOURCES):
            if document is not None:
                documents.append(document)
            elif error is not None:
                failures.append(f"{url}: {error}")

    if not documents:
        details = "\n".join(failures)
        raise RuntimeError(f"Could not load any official Microsoft Learn sources.\n{details}")

    if failures:
        failed_urls = "\n".join(failures)
        logger.warning("Some Microsoft Learn pages could not be loaded:\n%s", failed_urls)

    return documents


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
