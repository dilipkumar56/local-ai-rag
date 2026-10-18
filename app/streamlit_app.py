"""Interactive, publicly deployable AZ-104 study assistant."""

import logging
import os

import streamlit as st
from google.genai.errors import APIError
from langchain_chroma import Chroma
from streamlit.errors import StreamlitSecretNotFoundError

from hosted_rag import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    answer_az104_question,
    build_az104_store,
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
embedding_model = config_value("GEMINI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)

with st.sidebar:
    st.header("Study tools")
    st.write("Knowledge base: selected official Microsoft Learn pages for AZ-104.")
    st.caption(
        "The Learn index is reused after its first build. Refreshing makes embedding API requests."
    )
    if st.button("Refresh Learn content", use_container_width=True):
        st.cache_resource.clear()
        st.session_state.refresh_index = True
        st.rerun()
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


@st.cache_resource(
    ttl=21_600,
    show_spinner="Loading Microsoft Learn pages and building the study index...",
)
def get_store(
    key: str, embedding: str, refresh: bool
) -> tuple[Chroma, int]:
    return build_az104_store(key, embedding, refresh)


refresh_index = st.session_state.pop("refresh_index", False)
try:
    store, source_count = get_store(api_key, embedding_model, refresh_index)
except APIError as error:
    logger.exception("Gemini API failed while preparing the AZ-104 study index.")
    st.error(gemini_api_help(error))
    st.stop()
with st.sidebar:
    st.caption(f"{source_count} official source pages indexed.")

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
                    store,
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
