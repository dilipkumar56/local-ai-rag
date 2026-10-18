# Local AI RAG and AZ-104 Study Assistant

This repository includes two ways to use a document-grounded Azure assistant:

- A local command-line RAG chat backed by Ollama and your own `.txt`/`.pdf` files.
- A Streamlit AZ-104 study assistant that retrieves current content from selected official Microsoft Learn pages and uses a hosted OpenAI-compatible model API. The UI can be deployed publicly.

The assistant uses retrieval-augmented generation (RAG); it does **not** fine-tune or retrain a model. Answers are generated from retrieved reference excerpts and include links to the Microsoft Learn sources used.

## Prerequisites

- Python 3.11 or later.
- For the local CLI assistant: [Ollama](https://ollama.com/download).
- For the Streamlit assistant: an OpenAI-compatible API account, an API key, and a model provider that supports both chat completions and embeddings.
- To publish the UI: a GitHub repository and a [Streamlit Community Cloud](https://share.streamlit.io/) account.

## Install

From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Local CLI assistant

Pull the Ollama models:

```powershell
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

Place your `.txt` and `.pdf` files in `data\Documents\`, then build the local index and start the chat:

```powershell
python app\vector_db.py
python app\main.py
```

This local workflow uses Ollama and your own documents; it is separate from the hosted AZ-104 study assistant.

## Run the AZ-104 Streamlit app locally

The hosted app downloads selected Microsoft Learn pages when it starts and builds an AZ-104 retrieval index. Set the API key in the current PowerShell session (replace the placeholder; do not commit a real key):

```powershell
$env:OPENAI_API_KEY = "your-api-key"
```

For an OpenAI-compatible provider other than OpenAI, also set its API base URL and the model names it supports:

```powershell
$env:OPENAI_BASE_URL = "https://your-provider.example/v1"
$env:OPENAI_CHAT_MODEL = "your-chat-model"
$env:OPENAI_EMBEDDING_MODEL = "your-embedding-model"
```

`OPENAI_BASE_URL` is optional when using OpenAI's API. The defaults are `gpt-4o-mini` for chat and `text-embedding-3-small` for embeddings.

Start the interactive study app:

```powershell
streamlit run app\streamlit_app.py
```

Streamlit prints a local URL to open in your browser. Use the chat box to ask questions, follow the Microsoft Learn citations, and use the sidebar to refresh sources or clear the conversation.

## Publish a public Streamlit app

1. Push this repository to GitHub.
2. Sign in to Streamlit Community Cloud and create an app from the repository's `main` branch.
3. Set the app file path to `app/streamlit_app.py`.
4. Add the following values in the app's **Secrets** settings. Never commit API keys or a `.streamlit/secrets.toml` file.

   ```toml
   OPENAI_API_KEY = "your-api-key"
   # Optional for OpenAI-compatible providers:
   OPENAI_BASE_URL = "https://your-provider.example/v1"
   OPENAI_CHAT_MODEL = "your-chat-model"
   OPENAI_EMBEDDING_MODEL = "your-embedding-model"
   ```

5. Deploy the app and share the public URL Streamlit provides.

The public app uses the configured hosted provider for **both** chat and embeddings. Anyone with the URL can submit questions that consume your provider's usage. Set provider spending limits and monitor usage before sharing widely; deployment authentication is not configured by default.

### API rate-limit errors during startup

The initial study index requires embedding API requests. If startup reports that the provider has no available quota, enable API billing or increase the API project's spending limit; a ChatGPT subscription does not include API credits. If the provider reports a temporary rate limit, the app retries requests and indexes smaller batches, so wait briefly and restart the app. Check **Manage app** logs for the provider's precise error code.

## Official AZ-104 content

The app's curated URLs are maintained in `app/az104_sources.py`. It fetches those public Microsoft Learn pages at runtime rather than committing copies of their content. Use **Refresh Learn content** in the sidebar to rebuild the index; the app also refreshes its cached index periodically. Microsoft Learn pages can change, so check the linked source before relying on a study answer.

The app currently covers the certification study guide and selected topics across identity and governance, storage, compute, networking, monitoring, backup, and Azure Resource Manager. It is an exam-preparation aid, not an official Microsoft product, exam dump, or guarantee of passing. Follow the official [AZ-104 study guide](https://learn.microsoft.com/en-us/credentials/certifications/resources/study-guides/az-104) for the current skills measured.

## Tests

Run the unit tests from the activated virtual environment:

```powershell
python -m unittest discover -s tests
```

The official-source tests use mocked HTTP responses and do not download Learn pages.
