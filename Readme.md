# Local AI and RAG Application

A local AI learning project built with Python, Ollama, LangChain, and ChromaDB. The project explores running a local large language model (LLM), loading documents, generating embeddings, retrieving relevant content, and building a retrieval-augmented generation (RAG) workflow.

## Features

- Run a local LLM using Ollama.
- Ask general questions without requiring a document database.
- Load and process PDF documents using LangChain.
- Generate embeddings using the `nomic-embed-text` model.
- Store and retrieve document chunks using ChromaDB.
- Experiment with LangGraph-based RAG workflows.
- Explore cricket-related tools and content extraction.

## Technology Stack

- Python
- Ollama
- Llama 3.2 (3B)
- Nomic Embed Text
- LangChain
- LangGraph
- ChromaDB
- PyPDF

## Project Structure

```text
AI_Proj/
├── app/
│   ├── main.py
│   ├── langchain_app.py
│   ├── embedding_test.py
│   ├── pdf_load.py
│   ├── rag_load.py
│   ├── rag_chat.py
│   ├── retrieve.py
│   ├── graph_rag.py
│   ├── vector_db.py
│   └── tools/
│       ├── cricket.py
│       └── cricclubs.py
├── data/
│   └── documents/
│       └── azure_vnet.txt
├── .gitignore
├── requirements.txt
└── README.md
```

*The structure above represents the main project files; your local folders may contain additional files.*

## Prerequisites

Install the following before starting:

- Python 3.11 or another Python version supported by your installed dependencies.
- Git.
- Ollama for running models locally.

Download Ollama from https://ollama.com/download.

## 1. Clone the Repository

```powershell
git clone https://github.com/dilipkumar56/local-ai-rag.git
cd local-ai-rag
```

## 2. Create a Python Virtual Environment

```powershell
python -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Verify Python and pip:

```powershell
python --version
python -m pip --version
```

## 3. Install Python Dependencies

If `requirements.txt` exists:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If the dependency file has not been created yet, install the packages used by the project:

```powershell
python -m pip install langchain langchain-ollama langchain-chroma langchain-community langchain-text-splitters langgraph chromadb pypdf requests beautifulsoup4
```

## 4. Download Local AI Models

Download the chat model:

```powershell
ollama pull llama3.2:3b
```

Download the embedding model:

```powershell
ollama pull nomic-embed-text
```

Verify installed models:

```powershell
ollama list
```

## 5. Run the General AI Model

For general questions that do not require your own documents, run:

```powershell
ollama run llama3.2:3b
```

Example questions:

```text
What is a mango?
What is cloud computing?
Explain Azure Virtual Networks.
What is the difference between Azure and AWS?
```

Exit the interactive session with:

```text
/bye
```

## 6. Run the Python Application

From the repository root, try the main application:

```powershell
python app\main.py
```

Other available scripts can be run individually, depending on the workflow you want to test:

```powershell
python app\langchain_app.py
python app\embedding_test.py
python app\pdf_load.py
python app\rag_load.py
python app\vector_db.py
python app\retrieve.py
python app\rag_chat.py
python app\graph_rag.py
```

Each script may perform a different operation. Check the corresponding source file if a script requires a particular model, input document, or existing vector database.

## 7. PDF Processing

Place your PDF in the local document folder, for example:

```text
data/documents/azure_fundamentals.pdf
```

Run the PDF loader:

```powershell
python app\pdf_load.py
```

This script loads the PDF, splits its pages into chunks, and prints extracted text and metadata.

To use those chunks for RAG, generate embeddings and store them in ChromaDB using the project's vector database workflow.

## 8. Troubleshooting

### Ollama command is not recognized

Try running Ollama using its Windows installation path:

```powershell
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" list
```

### Python package is missing

Activate the virtual environment and install the required package:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install PACKAGE_NAME
```

Replace `PACKAGE_NAME` with the missing package.

### Document or database not found

Run commands from the repository root and verify that the expected files exist under `data/documents/` or that the configured ChromaDB directory has been created.

## Security and Data

- Do not commit API keys, passwords, access tokens, or `.env` files.
- Do not commit your Python virtual environment.
- Keep private documents and generated vector databases out of Git unless you have a specific reason to share them.
- Confirm that you have permission to publish any documents or extracted content.

## Future Improvements

- Add a single interactive Python chat entry point.
- Build a document-specific RAG chatbot.
- Add source citations to retrieved answers.
- Support multiple knowledge bases.
- Add automated tests and dependency version pinning.

## Author

Dilip Kumar

GitHub: https://github.com/dilipkumar56
