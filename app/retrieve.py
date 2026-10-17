"""Run a cited retrieval query against the local Chroma database."""

from rag import format_citation, retrieve


def main() -> None:
    question = input("Question: ").strip()
    for index, document in enumerate(retrieve(question), start=1):
        print(f"\n--- Result {index}: {format_citation(document)} ---")
        print(document.page_content)


if __name__ == "__main__":
    main()
