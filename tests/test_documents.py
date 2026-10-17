import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from langchain_core.documents import Document

from documents import load_documents, split_documents
from rag import format_citation
from settings import DOCUMENTS_DIR


class DocumentTests(unittest.TestCase):
    def test_load_documents_adds_source_metadata(self) -> None:
        documents = load_documents(DOCUMENTS_DIR, extensions={".txt"})

        self.assertGreater(len(documents), 0)
        self.assertTrue(all("source" in document.metadata for document in documents))


    def test_split_documents_adds_chunk_number(self) -> None:
        chunks = split_documents([Document(page_content="network " * 200, metadata={"source": "network.txt"})])

        self.assertEqual(chunks[0].metadata["chunk"], 1)


    def test_format_citation_includes_pdf_page_and_chunk(self) -> None:
        document = Document(page_content="text", metadata={"source": "guide.pdf", "page": 2, "chunk": 4})

        self.assertEqual(format_citation(document), "guide.pdf, page 3, chunk 4")


if __name__ == "__main__":
    unittest.main()
