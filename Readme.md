# Local AI RAG

A local, document-grounded Azure assistant built with Ollama, LangChain, ChromaDB, and LangGraph. It indexes local text and PDF files, answers from their contents, and displays the documents used for each answer.

## Requirements

- Python 3.11 or later
- [Ollama](https://ollama.com/download)

Set up the project from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

## Use the RAG assistant

1. Put `.txt` and `.pdf` files in `data/Documents/`.
2. Build or update the local database:

   ```powershell
   python app\vector_db.py
   ```

3. Start the cited chat application:

   ```powershell
   python app\main.py
   ```

Each answer lists its sources as a filename, PDF page where available, and chunk number. Type `exit` to quit.

To inspect retrieval without asking the model:

```powershell
python app\retrieve.py
```

To use the equivalent LangGraph workflow:

```powershell
python app\graph_rag.py
```

## Project layout

```text
app/
  main.py          Interactive RAG chat
  vector_db.py     Document ingestion command
  documents.py     File discovery, metadata, and chunking
  rag.py           Retrieval, grounded prompting, and citations
  settings.py      Shared paths, models, and retrieval settings
  graph_rag.py     LangGraph chat workflow
  tools/           Reusable cricket web clients
data/Documents/    Knowledge-base source files
tests/             Ingestion and citation tests
```

## Configuration

Shared settings live in `app/settings.py`. Adjust the model names, chunk size, overlap, or default retrieval count there. The database is stored locally in `data/chroma/` and is ignored by Git.

## Tests

Run the tests with:

```powershell
python -m unittest discover -s tests
```

## Notes

The assistant instructs the model to use only retrieved content. Review the listed citations before relying on an answer, particularly when you add new documents or change chunking settings.
