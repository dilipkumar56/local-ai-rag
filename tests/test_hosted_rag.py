import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from hosted_rag import answer_az104_question


class HostedRagTests(unittest.TestCase):
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
