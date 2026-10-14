from typing import TypedDict

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

from langgraph.graph import StateGraph, START, END


# ============================================================
# 1. GRAPH STATE
# ============================================================

class GraphState(TypedDict):
    question: str
    context: str
    answer: str
    score: float


# ============================================================
# 2. EMBEDDING MODEL
# ============================================================

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


# ============================================================
# 3. CHROMA VECTOR DATABASE
# ============================================================

vector_store = Chroma(
    persist_directory="data/chroma",
    embedding_function=embeddings
)


# ============================================================
# 4. LLM - LLAMA
# ============================================================

llm = ChatOllama(
    model="llama3.2:3b"
)


# ============================================================
# 5. RETRIEVE NODE
# ============================================================

def retrieve(state: GraphState):

    question = state["question"]

    print("\n[Graph] Retrieving relevant documents...")

    # Get documents + similarity scores
    results = vector_store.similarity_search_with_score(
        question,
        k=2
    )

    # Nothing found
    if not results:

        print("[Graph] No documents found.")

        return {
            "context": "",
            "score": 999
        }

    documents = []

    best_score = 999

    # Process results
    for document, score in results:

        print(
            f"[Graph] Retrieved document score: {score:.4f}"
        )

        documents.append(document)

        if score < best_score:
            best_score = score

    # Combine document chunks
    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    print(
        f"[Graph] Best similarity score: {best_score:.4f}"
    )

    return {
        "context": context,
        "score": best_score
    }


# ============================================================
# 6. CHECK CONTEXT NODE
# ============================================================

def check_context(state: GraphState):

    score = state["score"]

    print(
        f"[Graph] Checking relevance. Score: {score:.4f}"
    )

    # IMPORTANT:
    # Lower Chroma distance generally means more similar.
    #
    # This is only a starting threshold.
    # We will tune it after seeing your actual scores.

    if score < 1.0:

        print("[Graph] Relevant context found.")

        return "has_context"

    print("[Graph] Context is not relevant.")

    return "no_context"


# ============================================================
# 7. GENERATE NODE
# ============================================================

def generate(state: GraphState):

    print("[Graph] Generating answer with Llama...")

    prompt = ChatPromptTemplate.from_template(
        """
You are an Azure technical assistant.

Answer the question using ONLY the provided context.

Do NOT use outside knowledge.

If the context does not contain enough information,
say:

"I don't have enough information in the provided documents."

Context:
{context}

Question:
{question}
"""
    )

    # Create prompt messages
    messages = prompt.invoke(
        {
            "context": state["context"],
            "question": state["question"]
        }
    )

    # Send prompt to Llama
    response = llm.invoke(messages)

    return {
        "answer": response.content
    }


# ============================================================
# 8. BUILD LANGGRAPH
# ============================================================

graph_builder = StateGraph(GraphState)


# Add nodes
graph_builder.add_node(
    "retrieve",
    retrieve
)

graph_builder.add_node(
    "generate",
    generate
)


# START → RETRIEVE
graph_builder.add_edge(
    START,
    "retrieve"
)


# RETRIEVE → CHECK CONTEXT
#
# check_context returns either:
#
#     has_context
#     no_context
#
# has_context → generate
# no_context  → END

graph_builder.add_conditional_edges(
    "retrieve",
    check_context,
    {
        "has_context": "generate",
        "no_context": END
    }
)


# GENERATE → END
graph_builder.add_edge(
    "generate",
    END
)


# Compile graph
graph = graph_builder.compile()


# ============================================================
# 9. RUN APPLICATION
# ============================================================

print("\n==========================================")
print("       LOCAL AZURE RAG ASSISTANT")
print("==========================================")
print("Model: llama3.2:3b")
print("Embedding: nomic-embed-text")
print("Vector DB: Chroma")
print("Framework: LangGraph + LangChain")
print("Type 'exit' to quit.")
print("==========================================\n")


while True:

    question = input("You: ")

    # Exit
    if question.lower() == "exit":

        print("\nGoodbye!")

        break

    # Ignore empty input
    if not question.strip():

        continue

    # Initial graph state
    result = graph.invoke(
        {
            "question": question,
            "context": "",
            "answer": "",
            "score": 999
        }
    )

    print("\nAI:", result.get("answer", ""))