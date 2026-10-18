# Local AI RAG and Azure Assistant

This repository includes two ways to use a document-grounded Azure assistant:

- A local command-line RAG chat backed by Ollama and your own `.txt`/`.pdf` files.
- A Streamlit Azure assistant that searches official Microsoft Learn documentation for each question and uses the Gemini API to produce answers. The UI can be deployed publicly.

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

This local workflow uses Ollama and your own documents; it is separate from the hosted Azure assistant.

## Run the Azure assistant locally

The app searches Microsoft Learn documentation dynamically for each question, fetches relevant pages, and ranks their text locally. Search results and fetched pages are cached for the running process. It does not create a full vector index or call Gemini's embedding API. Gemini generates answers from the selected excerpts. Set the Gemini API key in the current PowerShell session (replace the placeholder; do not commit a real key):

```powershell
$env:GEMINI_API_KEY = "your-gemini-api-key"
```

The default chat model is `gemini-3.8-flash`. To use a different Gemini chat model, set:

```powershell
$env:GEMINI_CHAT_MODEL = "your-gemini-chat-model"
```

Start the interactive study app:

```powershell
streamlit run app\streamlit_app.py
```

Streamlit prints a local URL to open in your browser. Ask about any Azure service or solution, and follow the Microsoft Learn source links shown with each answer. Use the sidebar to refresh search/page caches or clear the conversation.

## Publish a public Streamlit app

1. Push this repository to GitHub.
2. Sign in to Streamlit Community Cloud and create an app from the repository's `main` branch.
3. Set the app file path to `app/streamlit_app.py`.
4. Add the Gemini API key in the app's **Secrets** settings. Never commit API keys or a `.streamlit/secrets.toml` file.

   ```toml
   GEMINI_API_KEY = "your-gemini-api-key"
   ```

5. Deploy the app and share the public URL Streamlit provides.

The public app uses Microsoft Learn search and Gemini for **chat only**. It does not make Gemini embedding calls or build a complete vector index at startup. Anyone with the URL can submit questions that use the project's Gemini chat quota. Review Google's [Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing) and [rate limits](https://ai.google.dev/gemini-api/docs/rate-limits) before sharing widely; free-tier model availability and limits vary. Deployment authentication is not configured by default.

### Source retrieval and caching

For each question the app calls the public Microsoft Learn search API, reranks official
documentation by query relevance, deprioritizes navigation-only documentation hubs, fetches the
best-matching pages, and ranks their text chunks locally. Search results and pages are cached in the
running app process. **Refresh Learn content** clears those caches. Retrieval does not use Gemini
embeddings, so it avoids Gemini's embedding quota; Gemini chat still uses tokens to generate the
answer. The RAG prompt asks Gemini to synthesize an explanation from substantive excerpts and not
mistake table-of-contents headings for evidence. Answers include links to the documentation pages
used. Check those sources for the latest details.

## Microsoft Learn sources

The app searches Microsoft Learn documentation across Azure services rather than relying on a
fixed set of articles. Search results and article URLs are restricted to `learn.microsoft.com`.
Microsoft Learn pages can change, so use the linked source pages to verify product behavior.

## Tests

Run the unit tests from the activated virtual environment:

```powershell
python -m unittest discover -s tests
```

The official-source tests use mocked HTTP responses and do not download Learn pages.
