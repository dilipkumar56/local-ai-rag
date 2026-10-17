"""Interactive local RAG assistant."""

from rag import answer_question
from settings import CHAT_MODEL, EMBEDDING_MODEL


def main() -> None:
    print("Local Azure RAG Assistant")
    print(f"Chat model: {CHAT_MODEL} | Embeddings: {EMBEDDING_MODEL}")
    print("Type 'exit' to quit. Run `python app\\vector_db.py` after adding documents.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() == "exit":
            return
        if not question:
            continue
        answer, citations = answer_question(question)
        print(f"\nAI: {answer}")
        if citations:
            print("Sources:")
            for citation in citations:
                print(f"- {citation}")
        print()


if __name__ == "__main__":
    main()
