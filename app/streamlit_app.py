"""Interactive, publicly deployable AZ-104 study assistant."""

import logging
import os

import streamlit as st
from google.genai.errors import APIError
from streamlit.errors import StreamlitSecretNotFoundError

from hosted_rag import (
    DEFAULT_CHAT_MODEL,
    answer_az104_question,
    clear_learn_source_cache,
    gemini_api_help,
)

logger = logging.getLogger(__name__)


def config_value(name: str, default: str = "") -> str:
    """Read a setting from the process environment or Streamlit secrets."""
    value = os.environ.get(name)
    if value:
        return value
    try:
        return str(st.secrets.get(name, default))
    except StreamlitSecretNotFoundError:
        return default


st.set_page_config(page_title="AZ-104 Study Assistant", page_icon="☁️", layout="centered")
st.title("AZ-104 Study Assistant")
st.caption(
    "Ask Azure Administrator questions grounded in official Microsoft Learn content, powered by Gemini."
)
st.info(
    "This is a study aid, not an exam dump or a guarantee of passing. "
    "Check the linked Microsoft Learn sources for the latest details."
)

api_key = config_value("GEMINI_API_KEY")
chat_model = config_value("GEMINI_CHAT_MODEL", DEFAULT_CHAT_MODEL)

with st.sidebar:
    st.header("Study tools")
    st.write("Retrieval: relevant official Microsoft Learn pages are fetched when you ask.")
    st.caption("Retrieved pages are cached for this running app; no Gemini embedding API calls are made.")
    if st.button("Refresh Learn content", use_container_width=True):
        clear_learn_source_cache()
        st.success("Cached Learn pages cleared. Pages will be fetched again when needed.")
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.markdown("**Try asking:**")
    st.markdown(
        "- How does Azure RBAC differ from Azure Policy?\n"
        "- When should I use LRS versus GRS storage?\n"
        "- How do NSGs filter traffic?\n"
        "- What should I consider for VM availability?"
    )

if not api_key:
    st.warning(
        "Add `GEMINI_API_KEY` to your environment or Streamlit deployment secrets to enable the tutor."
    )
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        for citation in message.get("citations", []):
            st.markdown(f"- [{citation['title']}]({citation['url']})")

question = st.chat_input("Ask an AZ-104 study question")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching Microsoft Learn and preparing an answer..."):
            try:
                answer, documents = answer_az104_question(
                    question,
                    api_key,
                    chat_model,
                    [
                        (message["role"], message["content"])
                        for message in st.session_state.messages[:-1]
                    ],
                )
            except APIError as error:
                logger.exception("Gemini API failed while answering an AZ-104 question.")
                answer = gemini_api_help(error)
                documents = []
            except RuntimeError as error:
                logger.exception("Could not retrieve Microsoft Learn content.")
                answer = f"Could not retrieve the reference content: {error}"
                documents = []
        st.markdown(answer)
        citations = []
        seen_urls: set[str] = set()
        for document in documents:
            url = str(document.metadata["url"])
            if url in seen_urls:
                continue
            seen_urls.add(url)
            citation = {"title": str(document.metadata["source"]), "url": url}
            citations.append(citation)
            st.markdown(f"- [{citation['title']}]({citation['url']})")
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "citations": citations}
    )
