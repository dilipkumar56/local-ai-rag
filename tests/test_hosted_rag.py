import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from hosted_rag import (
    DEFAULT_CHAT_MODEL,
    _load_source_chunks,
    answer_azure_question,
    clear_learn_source_cache,
    gemini_api_help,
    retrieve_relevant_documents,
)


class HostedRagTests(unittest.TestCase):
    def tearDown(self) -> None:
        clear_learn_source_cache()

    def test_default_chat_model_is_current_available_model(self) -> None:
        self.assertEqual(DEFAULT_CHAT_MODEL, "gemini-3.8-flash")

    @patch("hosted_rag._load_source_chunks")
    @patch("hosted_rag.search_official_sources")
    def test_retrieves_only_selected_chunks_and_ranks_by_question_terms(
        self,
        search_sources: Mock,
        load_chunks: Mock,
    ) -> None:
        from langchain_core.documents import Document

        search_sources.return_value = (("RBAC docs", "selected-url", "Azure role assignments"),)
        relevant = Document(
            page_content="Azure role assignments use RBAC roles to control access.",
            metadata={"source": "RBAC", "url": "selected-url"},
        )
        unrelated = Document(
            page_content="Storage account lifecycle and blob versioning.",
            metadata={"source": "Storage", "url": "selected-url"},
        )
        load_chunks.return_value = (unrelated, relevant)

        documents = retrieve_relevant_documents("How does RBAC control access?", k=1)

        self.assertEqual(documents, [relevant])
        search_sources.assert_called_once_with("How does RBAC control access?", 5)
        load_chunks.assert_called_once_with("selected-url")

    @patch("hosted_rag._load_source_chunks")
    @patch("hosted_rag.search_official_sources")
    def test_introduction_retrieval_prefers_early_overview_chunks(
        self,
        search_sources: Mock,
        load_chunks: Mock,
    ) -> None:
        from langchain_core.documents import Document

        url = "https://learn.microsoft.com/en-us/azure/cosmos-db/overview"
        search_sources.return_value = (("Azure Cosmos DB overview", url, "Overview"),)
        first_chunk = Document(
            page_content="Azure Cosmos DB is a fully managed globally distributed database.",
            metadata={"source": "Azure Cosmos DB overview", "url": url, "chunk": 1},
        )
        later_chunk = Document(
            page_content="Azure Cosmos DB emulator supports local testing.",
            metadata={"source": "Azure Cosmos DB overview", "url": url, "chunk": 8},
        )
        load_chunks.return_value = (later_chunk, first_chunk)

        documents = retrieve_relevant_documents(
            "Give me an introduction to Azure Cosmos DB",
            k=1,
        )

        self.assertEqual(documents, [first_chunk])

    @patch("hosted_rag._load_source_chunks")
    @patch("hosted_rag.search_official_sources")
    def test_broad_workload_question_prefers_explanatory_overview_chunk(
        self,
        search_sources: Mock,
        load_chunks: Mock,
    ) -> None:
        from langchain_core.documents import Document

        url = "https://learn.microsoft.com/en-us/azure/aks/workload-identity-overview"
        search_sources.return_value = (("AKS Workload Identity", url, "Overview"),)
        overview_chunk = Document(
            page_content=(
                "Azure Kubernetes Service workloads can use Microsoft Entra Workload ID "
                "to access Azure resources without stored credentials."
            ),
            metadata={"source": "AKS Workload Identity", "url": url, "chunk": 1},
        )
        implementation_chunk = Document(
            page_content=(
                "AZURE_FEDERATED_TOKEN_FILE identifies the projected token file for "
                "a Kubernetes workload identity."
            ),
            metadata={"source": "AKS Workload Identity", "url": url, "chunk": 15},
        )
        load_chunks.return_value = (implementation_chunk, overview_chunk)

        documents = retrieve_relevant_documents(
            "How do I secure an Azure Kubernetes workload?",
            k=1,
        )

        self.assertEqual(documents, [overview_chunk])

    @patch("hosted_rag._load_source_chunks")
    @patch("hosted_rag.search_official_sources")
    def test_comparison_retrieval_includes_both_requested_services(
        self,
        search_sources: Mock,
        load_chunks: Mock,
    ) -> None:
        from langchain_core.documents import Document

        sql_url = (
            "https://learn.microsoft.com/en-us/azure/azure-sql/database/"
            "sql-database-paas-overview"
        )
        cosmos_url = "https://learn.microsoft.com/en-us/azure/cosmos-db/overview"
        search_sources.return_value = (
            ("Azure SQL Database overview", sql_url, "Managed relational database."),
            ("Azure Cosmos DB overview", cosmos_url, "Globally distributed NoSQL database."),
        )
        sql_chunks = tuple(
            Document(
                page_content=f"Azure SQL Database relational overview section {index}.",
                metadata={"source": "Azure SQL Database", "url": sql_url, "chunk": index},
            )
            for index in range(1, 5)
        )
        cosmos_chunks = tuple(
            Document(
                page_content=f"Azure Cosmos DB NoSQL overview section {index}.",
                metadata={"source": "Azure Cosmos DB", "url": cosmos_url, "chunk": index},
            )
            for index in range(1, 5)
        )
        load_chunks.side_effect = (sql_chunks, cosmos_chunks)

        documents = retrieve_relevant_documents(
            "Compare Azure SQL Database and Cosmos DB.",
            k=5,
        )

        urls = {document.metadata["url"] for document in documents}
        self.assertIn(sql_url, urls)
        self.assertIn(cosmos_url, urls)
        self.assertEqual(len(documents), 5)
        self.assertEqual(
            {
                url: min(
                    document.metadata["chunk"]
                    for document in documents
                    if document.metadata["url"] == url
                )
                for url in (sql_url, cosmos_url)
            },
            {sql_url: 1, cosmos_url: 1},
        )

    @patch("hosted_rag.split_official_sources")
    @patch("hosted_rag.load_official_source")
    def test_selected_learn_pages_are_cached_until_refresh(
        self,
        load_source: Mock,
        split_sources: Mock,
    ) -> None:
        document = Mock()
        chunks = [Mock()]
        load_source.return_value = document
        split_sources.return_value = chunks
        url = "https://learn.microsoft.com/en-us/azure/storage/common/storage-redundancy"

        self.assertEqual(_load_source_chunks(url), tuple(chunks))
        self.assertEqual(_load_source_chunks(url), tuple(chunks))
        load_source.assert_called_once_with(url)

        clear_learn_source_cache()
        _load_source_chunks(url)

        self.assertEqual(load_source.call_count, 2)

    @patch("hosted_rag.retrieve_relevant_documents")
    @patch("hosted_rag.genai.Client")
    def test_follow_up_uses_gemini_and_preserves_citations(
        self,
        client_factory: Mock,
        retrieve: Mock,
    ) -> None:
        from langchain_core.documents import Document

        document = Document(
            metadata={
                "source": "Azure RBAC overview",
                "url": "https://learn.microsoft.com/example",
            },
            page_content="RBAC assigns roles to control access.",
        )
        retrieve.return_value = [document]
        client = client_factory.return_value.__enter__.return_value
        client.models.generate_content.return_value.text = "RBAC assigns permissions."

        answer, sources = answer_azure_question(
            "Why?",
            "test-key",
            "gemini-test-model",
            [("user", "How does Azure RBAC control access?")],
        )

        self.assertEqual(answer, "RBAC assigns permissions.")
        self.assertEqual(sources, [document])
        retrieve.assert_called_once_with("How does Azure RBAC control access?\nWhy?", k=5)
        client_factory.assert_called_once_with(api_key="test-key")
        client_factory.return_value.__exit__.assert_called_once_with(None, None, None)
        call = client.models.generate_content.call_args
        self.assertEqual(call.kwargs["model"], "gemini-test-model")
        self.assertIn("Microsoft Learn", call.kwargs["contents"])
        self.assertIn(
            "retrieval-augmented generation (RAG)",
            call.kwargs["config"].system_instruction,
        )
        client.models.embed_content.assert_not_called()

    def test_rate_limit_help_identifies_chat_quota(self) -> None:
        message = gemini_api_help(Exception("429 RESOURCE_EXHAUSTED"))

        self.assertIn("chat API quota or rate limit", message)
        self.assertIn("no longer calls Gemini's embedding API", message)

    def test_other_api_errors_suggest_key_and_chat_model_checks(self) -> None:
        message = gemini_api_help(Exception("permission denied"))

        self.assertIn("GEMINI_API_KEY", message)
        self.assertIn("chat model", message)


if __name__ == "__main__":
    unittest.main()
