from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# 1. Load PDF
# ============================================================

pdf_path = "data/documents/azure_fundamentals.pdf"

loader = PyPDFLoader(pdf_path)

documents = loader.load()

print(f"Number of pages loaded: {len(documents)}")


# ============================================================
# 2. Split PDF into chunks
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print(f"Number of chunks created: {len(chunks)}")


# ============================================================
# 3. Show first few chunks
# ============================================================

for i, chunk in enumerate(chunks[:5]):

    print("\n===================================")
    print(f"CHUNK {i + 1}")
    print("===================================")

    print(chunk.page_content[:1000])

    print("\nMetadata:")
    print(chunk.metadata)