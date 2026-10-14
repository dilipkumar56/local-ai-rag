from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

# 1. Embedding model
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

# 2. Connect to existing vector database
vector_store = Chroma(
    persist_directory="data/chroma",
    embedding_function=embeddings
)

# 3. Local Llama model
llm = ChatOllama(
    model="llama3.2:3b"
)

# 4. Prompt
prompt = ChatPromptTemplate.from_template("""
You are an Azure technical assistant.

Answer the question using ONLY the context provided below.

If the answer is not present in the context, say:
"I don't have enough information in the provided documents."

Context:
{context}

Question:
{question}
""")

while True:

    question = input("\nYou: ")

    if question.lower() == "exit":
        break

    # 5. Retrieve relevant documents
    documents = vector_store.similarity_search(
        question,
        k=2
    )

    # 6. Combine retrieved chunks
    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    # 7. Create prompt
    messages = prompt.invoke({
        "context": context,
        "question": question
    })

    # 8. Send context + question to Llama
    response = llm.invoke(messages)

    print("\nAI:", response.content)