"""LangGraph version of the RAG assistant with explicit no-context handling."""

from typing import TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

from rag import NO_CONTEXT_ANSWER, retrieve


class GraphState(TypedDict):
    question: str
    documents: list[Document]
    answer: str
    citations: list[str]


def retrieve_node(state: GraphState) -> dict:
    return {"documents": retrieve(state["question"])}


def route_context(state: GraphState) -> str:
    return "answer" if state["documents"] else "no_context"


def answer_node(state: GraphState) -> dict:
    # The shared function remains the single place that builds a grounded answer.
    from rag import answer_question

    answer, citations = answer_question(state["question"])
    return {"answer": answer, "citations": citations}


def no_context_node(_: GraphState) -> dict:
    return {"answer": NO_CONTEXT_ANSWER, "citations": []}


builder = StateGraph(GraphState)
builder.add_node("retrieve", retrieve_node)
builder.add_node("answer", answer_node)
builder.add_node("no_context", no_context_node)
builder.add_edge(START, "retrieve")
builder.add_conditional_edges("retrieve", route_context, {"answer": "answer", "no_context": "no_context"})
builder.add_edge("answer", END)
builder.add_edge("no_context", END)
graph = builder.compile()


def main() -> None:
    print("Local Azure RAG Assistant (LangGraph). Type 'exit' to quit.")
    while True:
        question = input("You: ").strip()
        if question.lower() == "exit":
            return
        if not question:
            continue
        result = graph.invoke({"question": question, "documents": [], "answer": "", "citations": []})
        print(f"\nAI: {result['answer']}")
        for citation in result["citations"]:
            print(f"- {citation}")
        print()


if __name__ == "__main__":
    main()
