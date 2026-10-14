from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# Connect to the same embedding model
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

# Connect to our existing Chroma database
vector_store = Chroma(
    persist_directory="data/chroma",
    embedding_function=embeddings
)

# Ask a question
question = "How can two Azure VNets communicate?"

# Search for the 2 most relevant chunks
results = vector_store.similarity_search(
    question,
    k=2
)

print(f"Question: {question}")

for i, document in enumerate(results):
    print(f"\n--- Result {i + 1} ---")
    print(document.page_content)
    print("Metadata:", document.metadata)