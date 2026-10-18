import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from hosted_rag import (
    answer_az104_question,
    build_az104_store,
    rate_limit_help,
)


class HostedRagTests(unittest.TestCase):
    @patch("hosted_rag.Chroma.from_documents")
    @patch("hosted_rag.Chroma")
    @patch("hosted_rag.OpenAIEmbeddings")
    @patch("hosted_rag.split_official_sources")
    @patch("hosted_rag.load_official_sources")
    def test_builds_index_in_small_batches(
        self,
        load_sources: Mock,
        split_sources: Mock,
        embeddings_factory: Mock,
        chroma_factory: Mock,
        from_documents: Mock,
    ) -> None:
        source = Mock()
        chunks = [Mock() for _ in range(5)]
        load_sources.return_value = [source]
        split_sources.return_value = chunks
        from_documents.return_value = Mock()

        with patch("hosted_rag.EMBEDDING_BATCH_SIZE", 2):
            store, source_count = build_az104_store("test-key", None, "test-embedding")

        self.assertEqual(source_count, 1)
        self.assertEqual(store, from_documents.return_value)
        self.assertEqual(from_documents.call_count, 3)
        self.assertEqual(
            [len(call.kwargs["documents"]) for call in from_documents.call_args_list],
            [2, 2, 1],
        )
        chroma_factory.return_value.delete_collection.assert_called_once()

    def test_quota_error_points_to_api_billing(self) -> None:
        error = Mock(body={"error": {"code": "insufficient_quota"}})

        message = rate_limit_help(error)

        self.assertIn("API billing", message)
        self.assertIn("ChatGPT subscription", message)

    def test_transient_rate_limit_suggests_waiting(self) -> None:
        error = Mock(body={"error": {"code": "rate_limit_exceeded"}})

        message = rate_limit_help(error)

        self.assertIn("Wait a few minutes", message)
        self.assertIn("smaller batches", message)

    @patch("hosted_rag.ChatOpenAI")
    def test_follow_up_retrieves_with_previous_question_and_keeps_sources(
        self, chat_model: Mock
    ) -> None:
        document = Mock(
            metadata={
                "source": "Azure RBAC overview",
                "url": "https://learn.microsoft.com/example",
            },
            page_content="RBAC assigns roles to control access.",
        )
        store = Mock()
        store.similarity_search.return_value = [document]
        chat_model.return_value.invoke.return_value.content = "Because RBAC assigns permissions."

        answer, sources = answer_az104_question(
            "Why?",
            store,
            "test-key",
            None,
            "test-model",
            [("user", "How does Azure RBAC control access?")],
        )

        self.assertEqual(answer, "Because RBAC assigns permissions.")
        self.assertEqual(sources, [document])
        store.similarity_search.assert_called_once_with(
            "How does Azure RBAC control access?\nWhy?",
            k=5,
        )


if __name__ == "__main__":
    unittest.main()
