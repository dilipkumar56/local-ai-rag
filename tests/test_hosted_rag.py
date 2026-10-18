import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from hosted_rag import (
    EMBEDDING_INPUT_INTERVAL_SECONDS,
    GeminiEmbeddings,
    _chunk_id,
    answer_az104_question,
    build_az104_store,
    gemini_api_help,
)


class HostedRagTests(unittest.TestCase):
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
    ) -> None:
        load_sources.return_value = [Mock()]
        split_sources.return_value = [
            Mock(page_content=f"chunk-{index}", metadata={"url": "url", "chunk": index})
            for index in range(5)
        ]
        chroma_factory.return_value._collection.count.return_value = 0
        chroma_factory.return_value._collection.metadata = None
        chroma_factory.return_value._collection.get.return_value = {"ids": []}
        chroma_factory.return_value.add_documents.return_value = None

        with (
            patch("hosted_rag.EMBEDDING_BATCH_SIZE", 2),
            patch("hosted_rag.EMBEDDING_INPUT_INTERVAL_SECONDS", 0),
        ):
            store, source_count = build_az104_store("test-key", "test-embedding")

        self.assertEqual(source_count, 1)
        self.assertEqual(store, chroma_factory.return_value)
        self.assertEqual(store.add_documents.call_count, 3)
        self.assertEqual(
            [len(call.kwargs["documents"]) for call in store.add_documents.call_args_list],
            [2, 2, 1],
        )
        self.assertTrue(all(call.kwargs["ids"] for call in store.add_documents.call_args_list))
        store._collection.modify.assert_called_with(
            metadata={
                "embedding_provider": "gemini",
                "embedding_model": "test-embedding",
                "index_complete": True,
            }
        )
        chroma_factory.return_value.delete_collection.assert_called_once()
        client_factory.assert_called_once_with(api_key="test-key")

    @patch("hosted_rag.Chroma")
    @patch("hosted_rag.genai.Client")
    @patch("hosted_rag.load_official_sources")
    def test_reuses_existing_index_without_refetching_sources(
        self,
        load_sources: Mock,
        client_factory: Mock,
        chroma_factory: Mock,
    ) -> None:
        chroma_factory.return_value._collection.metadata = {
            "embedding_provider": "gemini",
            "embedding_model": "test-embedding",
            "index_complete": True,
        }

        store, source_count = build_az104_store("test-key", "test-embedding")

        self.assertEqual(store, chroma_factory.return_value)
        self.assertEqual(source_count, 45)
        load_sources.assert_not_called()

    @patch("hosted_rag.Chroma")
    @patch("hosted_rag.genai.Client")
    @patch("hosted_rag.split_official_sources")
    @patch("hosted_rag.load_official_sources")
    def test_resumes_incomplete_index_without_deleting_existing_chunks(
        self,
        load_sources: Mock,
        split_sources: Mock,
        client_factory: Mock,
        chroma_factory: Mock,
    ) -> None:
        chunks = [
            Mock(page_content=f"chunk-{index}", metadata={"url": "url", "chunk": index})
            for index in range(3)
        ]
        load_sources.return_value = [Mock()]
        split_sources.return_value = chunks
        chroma_factory.return_value._collection.metadata = {
            "embedding_provider": "gemini",
            "embedding_model": "test-embedding",
            "index_complete": False,
        }
        chroma_factory.return_value._collection.get.return_value = {
            "ids": [_chunk_id(chunks[0])]
        }

        with patch("hosted_rag.EMBEDDING_INPUT_INTERVAL_SECONDS", 0):
            store, _ = build_az104_store("test-key", "test-embedding")

        self.assertEqual(store, chroma_factory.return_value)
        store.delete_collection.assert_not_called()
        store.add_documents.assert_called_once()
        self.assertEqual(store.add_documents.call_args.kwargs["documents"], chunks[1:])
        self.assertEqual(
            store.add_documents.call_args.kwargs["ids"],
            [_chunk_id(chunks[1]), _chunk_id(chunks[2])],
        )

    def test_gemini_embedding_adapter_uses_document_and_query_tasks(self) -> None:
        client = Mock()
        client.models.embed_content.side_effect = [
            SimpleNamespace(
                embeddings=[
                    SimpleNamespace(values=[1.0, 2.0]),
                    SimpleNamespace(values=[3.0, 4.0]),
                ]
            ),
            SimpleNamespace(embeddings=[SimpleNamespace(values=[7.0, 8.0])]),
            SimpleNamespace(embeddings=[SimpleNamespace(values=[5.0, 6.0])]),
        ]
        embeddings = GeminiEmbeddings(client, "test-embedding")

        with (
            patch("hosted_rag.EMBEDDING_BATCH_SIZE", 2),
            patch("hosted_rag.EMBEDDING_INPUT_INTERVAL_SECONDS", 0),
        ):
            self.assertEqual(
                embeddings.embed_documents(["first", "second", "third"]),
                [[1.0, 2.0], [3.0, 4.0], [7.0, 8.0]],
            )
        self.assertEqual(embeddings.embed_query("question"), [5.0, 6.0])
        document_config = client.models.embed_content.call_args_list[0].kwargs["config"]
        query_config = client.models.embed_content.call_args_list[2].kwargs["config"]
        self.assertEqual(document_config.task_type, "RETRIEVAL_DOCUMENT")
        self.assertEqual(query_config.task_type, "RETRIEVAL_QUERY")
        self.assertEqual(client.models.embed_content.call_count, 3)
        self.assertEqual(EMBEDDING_INPUT_INTERVAL_SECONDS, 0.75)

    def test_embedding_requests_are_spaced_apart(self) -> None:
        client = Mock()
        client.models.embed_content.side_effect = [
            SimpleNamespace(embeddings=[SimpleNamespace(values=[1.0])]),
            SimpleNamespace(embeddings=[SimpleNamespace(values=[2.0])]),
        ]
        embeddings = GeminiEmbeddings(client, "test-embedding")

        with (
            patch("hosted_rag.EMBEDDING_INPUT_INTERVAL_SECONDS", 1.0),
            patch("hosted_rag._LAST_EMBEDDING_REQUEST", -10.0),
            patch("hosted_rag.time.monotonic", side_effect=[0.0, 0.0, 0.2, 1.2]),
            patch("hosted_rag.time.sleep") as sleep,
        ):
            embeddings.embed_query("first")
            embeddings.embed_query("second")

        sleep.assert_called_once_with(0.8)

    def test_embedding_batches_are_paced_by_number_of_inputs(self) -> None:
        client = Mock()
        client.models.embed_content.return_value = SimpleNamespace(
            embeddings=[
                SimpleNamespace(values=[1.0]),
                SimpleNamespace(values=[2.0]),
            ]
        )
        embeddings = GeminiEmbeddings(client, "test-embedding")

        with (
            patch("hosted_rag.EMBEDDING_BATCH_SIZE", 2),
            patch("hosted_rag.EMBEDDING_INPUT_INTERVAL_SECONDS", 1.0),
            patch("hosted_rag._LAST_EMBEDDING_REQUEST", -10.0),
            patch("hosted_rag.time.monotonic", side_effect=[0.0, 0.0, 0.0, 0.0]),
            patch("hosted_rag.time.sleep") as sleep,
        ):
            embeddings.embed_documents(["first", "second", "third", "fourth"])

        sleep.assert_called_once_with(2.0)

    def test_chunk_id_is_stable_and_varies_with_content(self) -> None:
        first = Mock(page_content="same content", metadata={"url": "source", "chunk": 1})
        equivalent = Mock(page_content="same content", metadata={"url": "source", "chunk": 1})
        changed = Mock(page_content="changed content", metadata={"url": "source", "chunk": 1})

        self.assertEqual(_chunk_id(first), _chunk_id(equivalent))
        self.assertNotEqual(_chunk_id(first), _chunk_id(changed))

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
