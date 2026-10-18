import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from az104_sources import MICROSOFT_LEARN_SOURCES, load_official_sources, split_official_sources


class OfficialSourceTests(unittest.TestCase):
    @patch("az104_sources.requests.get")
    def test_loads_official_learn_article_with_citation_metadata(self, get: Mock) -> None:
        get.return_value = Mock(
            url="https://learn.microsoft.com/en-us/azure/role-based-access-control/overview",
            text=(
                "<html><main><article><h1>Azure RBAC</h1>"
                "<nav>Navigation that should be removed</nav>"
                "<p>Use role assignments to control access.</p>"
                "</article></main></html>"
            ),
        )
        get.return_value.raise_for_status.return_value = None

        documents = load_official_sources()

        self.assertEqual(len(documents), len(MICROSOFT_LEARN_SOURCES))
        self.assertEqual(documents[0].metadata["source"], "Azure RBAC")
        self.assertEqual(
            documents[0].metadata["url"],
            "https://learn.microsoft.com/en-us/azure/role-based-access-control/overview",
        )
        self.assertNotIn("Navigation that should be removed", documents[0].page_content)
        self.assertIn("Use role assignments", documents[0].page_content)

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

    @patch("az104_sources.requests.get")
    def test_rejects_redirects_outside_microsoft_learn(self, get: Mock) -> None:
        get.return_value = Mock(
            url="https://example.com/redirect",
            text="<html><main><article><h1>Not Microsoft Learn</h1></article></main></html>",
        )
        get.return_value.raise_for_status.return_value = None

        with self.assertRaisesRegex(RuntimeError, "Could not load any official"):
            load_official_sources()


if __name__ == "__main__":
    unittest.main()
