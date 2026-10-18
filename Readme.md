# Local AI RAG and AZ-104 Study Assistant

This repository includes two ways to use a document-grounded Azure assistant:

- A local command-line RAG chat backed by Ollama and your own `.txt`/`.pdf` files.
- A Streamlit AZ-104 study assistant that retrieves current content from selected official Microsoft Learn pages and uses the Gemini API for chat and embeddings. The UI can be deployed publicly.

The assistant uses retrieval-augmented generation (RAG); it does **not** fine-tune or retrain a model. Answers are generated from retrieved reference excerpts and include links to the Microsoft Learn sources used.

## Prerequisites

- Python 3.11 or later.
- For the local CLI assistant: [Ollama](https://ollama.com/download).
- For the Streamlit assistant: a Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey).
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

The hosted app downloads selected Microsoft Learn pages and uses Gemini to build an AZ-104 retrieval index. The index is stored locally and reused after its first build. Set the Gemini API key in the current PowerShell session (replace the placeholder; do not commit a real key):

```powershell
$env:GEMINI_API_KEY = "your-gemini-api-key"
```

The default models are `gemini-2.5-flash` for answers and `gemini-embedding-001` for embeddings. To use different Gemini models, set:

```powershell
$env:GEMINI_CHAT_MODEL = "your-gemini-chat-model"
$env:GEMINI_EMBEDDING_MODEL = "your-gemini-embedding-model"
```

Start the interactive study app:

```powershell
streamlit run app\streamlit_app.py
```

Streamlit prints a local URL to open in your browser. Use the chat box to ask questions, follow the Microsoft Learn citations, and use the sidebar to refresh sources or clear the conversation.

## Publish a public Streamlit app

1. Push this repository to GitHub.
2. Sign in to Streamlit Community Cloud and create an app from the repository's `main` branch.
3. Set the app file path to `app/streamlit_app.py`.
4. Add the Gemini API key in the app's **Secrets** settings. Never commit API keys or a `.streamlit/secrets.toml` file.

   ```toml
   GEMINI_API_KEY = "your-gemini-api-key"
   ```

5. Deploy the app and share the public URL Streamlit provides.

The public app uses Gemini for **both** chat and embeddings. Anyone with the URL can submit questions that use the project's Gemini API quota. Review Google's [Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing) and [rate limits](https://ai.google.dev/gemini-api/docs/rate-limits) before sharing widely; free-tier model availability and limits vary. Deployment authentication is not configured by default.

### API rate-limit errors during startup

The initial study index contains about 982 text chunks. Gemini embedding calls are paced by the
number of texts in each batch, at roughly 80 chunks per minute, so a full first index can take
around 13 minutes. Completed chunks are saved with stable IDs so indexing can resume after an
interrupted run when Streamlit's local app storage is retained. Avoid repeatedly rebooting the app
or selecting **Refresh Learn content** while indexing. If startup reports a rate limit, check the
Gemini embedding model's active quota in AI Studio and the **Manage app** logs; Gemini quotas are
shared by all API keys in the same Google Cloud project.

## Official AZ-104 content

The app's curated URLs are maintained in `app/az104_sources.py`. It fetches those public Microsoft Learn pages at runtime rather than committing copies of their content. Use **Refresh Learn content** in the sidebar to rebuild the index. Microsoft Learn pages can change, so check the linked source before relying on a study answer.

The app currently covers the certification study guide and selected topics across identity and governance, storage, compute, networking, monitoring, backup, and Azure Resource Manager. It is an exam-preparation aid, not an official Microsoft product, exam dump, or guarantee of passing. Follow the official [AZ-104 study guide](https://learn.microsoft.com/en-us/credentials/certifications/resources/study-guides/az-104) for the current skills measured.

## Tests

Run the unit tests from the activated virtual environment:

```powershell
python -m unittest discover -s tests
```

The official-source tests use mocked HTTP responses and do not download Learn pages.
