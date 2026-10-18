import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from learn_sources import (
    load_official_source,
    search_official_sources,
    split_official_sources,
)


class OfficialSourceTests(unittest.TestCase):
    def tearDown(self) -> None:
        search_official_sources.cache_clear()

    @patch("learn_sources.requests.get")
    def test_loads_official_learn_page_with_citation_metadata(self, get: Mock) -> None:
        url = "https://learn.microsoft.com/en-us/azure/role-based-access-control/overview"
        get.return_value = Mock(
            url=url,
            text="<html><main><article><h1>Azure RBAC</h1>"
            "<nav>Navigation that should be removed</nav><p>Role assignments</p>"
            "</article></main></html>",
        )
        get.return_value.raise_for_status.return_value = None

        document = load_official_source(url)

        self.assertEqual(document.metadata["source"], "Azure RBAC")
        get.assert_called_once()
        self.assertEqual(get.call_args.args[0], url)
        self.assertNotIn("Navigation that should be removed", document.page_content)

    def test_refuses_to_fetch_a_url_outside_microsoft_learn(self) -> None:
        with patch("learn_sources.requests.get") as get:
            with self.assertRaisesRegex(ValueError, "learn.microsoft.com"):
                load_official_source("https://example.com/not-a-learn-page")
        get.assert_not_called()

    @patch("learn_sources.requests.get")
    def test_searches_learn_for_azure_documentation_and_filters_untrusted_results(
        self,
        get: Mock,
    ) -> None:
        get.return_value = Mock(
            json=Mock(
                return_value={
                    "results": [
                        {
                            "title": "AKS Workload Identity",
                            "url": "https://learn.microsoft.com/en-us/azure/aks/workload-identity-overview",
                            "description": "Use workload identity with AKS.",
                            "category": "Documentation",
                        },
                        {
                            "title": "Training result",
                            "url": "https://learn.microsoft.com/training/azure",
                            "category": "Training",
                        },
                        {
                            "title": "Untrusted result",
                            "url": "https://example.com/fake",
                            "category": "Documentation",
                        },
                    ]
                }
            )
        )
        get.return_value.raise_for_status.return_value = None

        results = search_official_sources("How does workload identity work?")

        self.assertEqual(
            results,
            (
                (
                    "AKS Workload Identity",
                    "https://learn.microsoft.com/en-us/azure/aks/workload-identity-overview",
                    "Use workload identity with AKS.",
                ),
            ),
        )
        self.assertEqual(
            get.call_args.kwargs["params"]["search"],
            "Azure How does workload identity work?",
        )

    @patch("learn_sources.requests.get")
    def test_ranks_cosmos_db_overview_above_hub_and_unrelated_results(
        self,
        get: Mock,
    ) -> None:
        get.return_value = Mock(
            json=Mock(
                return_value={
                    "results": [
                        {
                            "title": "Azure Cosmos DB documentation - Azure Cosmos DB",
                            "url": "https://learn.microsoft.com/en-us/azure/cosmos-db/",
                            "description": (
                                "Overview Introduction to Azure Cosmos DB Architecture "
                                "Vector Database in Azure Cosmos DB"
                            ),
                            "category": "Documentation",
                        },
                        {
                            "title": "Entity Framework documentation hub",
                            "url": "https://learn.microsoft.com/en-us/ef/",
                            "description": "A modern object relational mapper for .NET.",
                            "category": "Documentation",
                        },
                        {
                            "title": "Managed identities for Azure resources",
                            "url": (
                                "https://learn.microsoft.com/en-us/entra/identity/"
                                "managed-identities-azure-resources/overview"
                            ),
                            "description": (
                                "Use managed identities to securely access Azure Cosmos DB."
                            ),
                            "category": "Documentation",
                        },
                        {
                            "title": "Introduction/Overview - Azure Cosmos DB for MongoDB",
                            "url": "https://learn.microsoft.com/en-us/azure/cosmos-db/mongodb/overview",
                            "description": (
                                "Azure Cosmos DB for MongoDB is a fully managed NoSQL database "
                                "with MongoDB wire protocol compatibility."
                            ),
                            "category": "Documentation",
                        },
                        {
                            "title": "Unified AI Database - Azure Cosmos DB",
                            "url": "https://learn.microsoft.com/en-us/azure/cosmos-db/overview",
                            "description": (
                                "Azure Cosmos DB is a NoSQL and vector database with "
                                "reliability and flexible operational data."
                            ),
                            "category": "Documentation",
                        },
                    ]
                }
            )
        )
        get.return_value.raise_for_status.return_value = None

        results = search_official_sources("Give me an introduction to Azure Cosmos DB")

        self.assertEqual(
            results[0][1],
            "https://learn.microsoft.com/en-us/azure/cosmos-db/overview",
        )
        self.assertFalse(any(url.endswith("/ef/") for _, url, _ in results))
        self.assertFalse(any(url.endswith("/cosmos-db/") for _, url, _ in results))
        self.assertFalse(any("/managed-identities-" in url for _, url, _ in results))
        self.assertFalse(any("/mongodb/" in url for _, url, _ in results[:2]))

    def test_split_preserves_source_link_and_adds_chunk_numbers(self) -> None:
        from langchain_core.documents import Document

        documents = [
            Document(
                page_content="RBAC policy " * 150,
                metadata={"source": "Azure RBAC", "url": "https://learn.microsoft.com/example"},
            )
        ]

        chunks = split_official_sources(documents)

        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0].metadata["source"], "Azure RBAC")
        self.assertEqual(chunks[0].metadata["url"], "https://learn.microsoft.com/example")
        self.assertEqual(chunks[0].metadata["chunk"], 1)
        self.assertEqual(chunks[1].metadata["chunk"], 2)

    @patch("learn_sources.requests.get")
    def test_learn_search_results_are_cached(self, get: Mock) -> None:
        get.return_value = Mock(
            json=Mock(return_value={"results": []}),
        )
        get.return_value.raise_for_status.return_value = None

        search_official_sources("Azure storage")
        search_official_sources("Azure storage")

        get.assert_called_once()


if __name__ == "__main__":
    unittest.main()
