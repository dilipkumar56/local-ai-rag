import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from hosted_rag import (
    EMBEDDING_BATCH_SIZE,
    GeminiEmbeddings,
    answer_az104_question,
    build_az104_store,
    gemini_api_help,
)


class HostedRagTests(unittest.TestCase):
    @patch("hosted_rag.Chroma.from_documents")
    @patch("hosted_rag.Chroma")
    @patch("hosted_rag.genai.Client")
    @patch("hosted_rag.split_official_sources")
    @patch("hosted_rag.load_official_sources")
    def test_builds_index_in_small_batches(
        self,
        load_sources: Mock,
        split_sources: Mock,
        client_factory: Mock,
        chroma_factory: Mock,
        from_documents: Mock,
    ) -> None:
        load_sources.return_value = [Mock()]
        split_sources.return_value = [Mock() for _ in range(5)]
        chroma_factory.return_value._collection.count.return_value = 0
        from_documents.return_value = Mock()

        with patch("hosted_rag.EMBEDDING_BATCH_SIZE", 2):
            store, source_count = build_az104_store("test-key", "test-embedding")

        self.assertEqual(source_count, 1)
        self.assertEqual(store, from_documents.return_value)
        self.assertEqual(from_documents.call_count, 3)
        self.assertEqual(
            [len(call.kwargs["documents"]) for call in from_documents.call_args_list],
            [2, 2, 1],
        )
        chroma_factory.return_value.delete_collection.assert_called_once()
        client_factory.assert_called_once_with(api_key="test-key")

    @patch("hosted_rag.Chroma.from_documents")
    @patch("hosted_rag.Chroma")
    @patch("hosted_rag.genai.Client")
    @patch("hosted_rag.load_official_sources")
    def test_reuses_existing_index_without_refetching_sources(
        self,
        load_sources: Mock,
        client_factory: Mock,
        chroma_factory: Mock,
        from_documents: Mock,
    ) -> None:
        chroma_factory.return_value._collection.count.return_value = 12

        store, source_count = build_az104_store("test-key", "test-embedding")

        self.assertEqual(store, chroma_factory.return_value)
        self.assertEqual(source_count, 45)
        load_sources.assert_not_called()
        from_documents.assert_not_called()

    def test_gemini_embedding_adapter_uses_document_and_query_tasks(self) -> None:
        client = Mock()
        client.models.embed_content.side_effect = [
            SimpleNamespace(
                embeddings=[
                    SimpleNamespace(values=[1.0, 2.0]),
                    SimpleNamespace(values=[3.0, 4.0]),
                ]
            ),
            SimpleNamespace(embeddings=[SimpleNamespace(values=[5.0, 6.0])]),
        ]
        embeddings = GeminiEmbeddings(client, "test-embedding")

        self.assertEqual(embeddings.embed_documents(["first", "second"]), [[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(embeddings.embed_query("question"), [5.0, 6.0])
        document_config = client.models.embed_content.call_args_list[0].kwargs["config"]
        query_config = client.models.embed_content.call_args_list[1].kwargs["config"]
        self.assertEqual(document_config.task_type, "RETRIEVAL_DOCUMENT")
        self.assertEqual(query_config.task_type, "RETRIEVAL_QUERY")

    @patch("hosted_rag.genai.Client")
    def test_follow_up_uses_gemini_and_preserves_citations(self, client_factory: Mock) -> None:
        document = Mock(
            metadata={
                "source": "Azure RBAC overview",
                "url": "https://learn.microsoft.com/example",
            },
            page_content="RBAC assigns roles to control access.",
        )
        store = Mock()
        store.similarity_search.return_value = [document]
        client_factory.return_value.models.generate_content.return_value.text = (
            "Because RBAC assigns permissions."
        )

        answer, sources = answer_az104_question(
            "Why?",
            store,
            "test-key",
            "gemini-test-model",
            [("user", "How does Azure RBAC control access?")],
        )

        self.assertEqual(answer, "Because RBAC assigns permissions.")
        self.assertEqual(sources, [document])
        store.similarity_search.assert_called_once_with(
            "How does Azure RBAC control access?\nWhy?",
            k=5,
        )
        client_factory.assert_called_once_with(api_key="test-key")
        call = client_factory.return_value.models.generate_content.call_args
        self.assertEqual(call.kwargs["model"], "gemini-test-model")
        self.assertIn("Microsoft Learn", call.kwargs["contents"])
        self.assertIn("only the Microsoft Learn", call.kwargs["config"].system_instruction)

    def test_rate_limit_help_suggests_checking_gemini_quota(self) -> None:
        error = Exception("429 RESOURCE_EXHAUSTED")

        message = gemini_api_help(error)

        self.assertIn("Gemini's API quota or rate limit", message)
        self.assertIn("Google AI Studio", message)

    def test_other_api_errors_suggest_key_and_model_checks(self) -> None:
        error = Exception("permission denied")

        message = gemini_api_help(error)

        self.assertIn("GEMINI_API_KEY", message)
        self.assertIn("embedding models", message)


if __name__ == "__main__":
    unittest.main()
