import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from hosted_rag import (
    _load_source_chunks,
    answer_az104_question,
    clear_learn_source_cache,
    gemini_api_help,
    rank_source_urls,
    retrieve_relevant_documents,
)


class HostedRagTests(unittest.TestCase):
    def tearDown(self) -> None:
        clear_learn_source_cache()

    def test_ranker_selects_sources_for_acronyms_and_comparison_topics(self) -> None:
        urls = rank_source_urls("How does Azure RBAC differ from Azure Policy?", limit=5)

        self.assertIn(
            "https://learn.microsoft.com/en-us/azure/role-based-access-control/overview",
            urls,
        )
        self.assertIn(
            "https://learn.microsoft.com/en-us/azure/governance/policy/overview",
            urls,
        )
        self.assertLessEqual(len(urls), 5)

    def test_ranker_falls_back_to_az104_study_guide_for_general_questions(self) -> None:
        urls = rank_source_urls("Can you help me?")

        self.assertEqual(
            urls,
            [
                "https://learn.microsoft.com/en-us/credentials/certifications/"
                "resources/study-guides/az-104"
            ],
        )

    @patch("hosted_rag._load_source_chunks")
    @patch("hosted_rag.rank_source_urls")
    def test_retrieves_only_selected_chunks_and_ranks_by_question_terms(
        self,
        rank_urls: Mock,
        load_chunks: Mock,
    ) -> None:
        from langchain_core.documents import Document

        rank_urls.return_value = ["selected-url"]
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
        rank_urls.assert_called_once_with("How does RBAC control access?", 5)
        load_chunks.assert_called_once_with("selected-url")

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
        url = rank_source_urls("storage redundancy")[0]

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
        client = client_factory.return_value
        client.models.generate_content.return_value.text = "RBAC assigns permissions."

        answer, sources = answer_az104_question(
            "Why?",
            "test-key",
            "gemini-test-model",
            [("user", "How does Azure RBAC control access?")],
        )

        self.assertEqual(answer, "RBAC assigns permissions.")
        self.assertEqual(sources, [document])
        retrieve.assert_called_once_with("How does Azure RBAC control access?\nWhy?", k=5)
        client_factory.assert_called_once_with(api_key="test-key")
        call = client.models.generate_content.call_args
        self.assertEqual(call.kwargs["model"], "gemini-test-model")
        self.assertIn("Microsoft Learn", call.kwargs["contents"])
        self.assertIn("only the Microsoft Learn", call.kwargs["config"].system_instruction)
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
