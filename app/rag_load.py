from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Load document
loader = TextLoader("data/documents/azure_vnet.txt")
documents = loader.load()

# 2. Create text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)

# 3. Split document
chunks = text_splitter.split_documents(documents)

# 4. Display chunks
print("Number of chunks:", len(chunks))

for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i + 1} ---")
    print(chunk.page_content)