from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# 1. Load document
loader = TextLoader("data/documents/azure_vnet.txt")
documents = loader.load()

# 2. Split document into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks")

# 3. Connect to local Ollama embedding model
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

# 4. Create Chroma vector database
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="data/chroma"
)

print("Vector database created successfully!")