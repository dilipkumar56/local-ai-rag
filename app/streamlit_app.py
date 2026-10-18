"""Interactive, publicly deployable Azure Learn assistant."""

import logging
import os
from uuid import uuid4

import streamlit as st
from google.genai.errors import APIError
from streamlit.errors import StreamlitSecretNotFoundError

from hosted_rag import (
    DEFAULT_CHAT_MODEL,
    LearnSourceError,
    answer_azure_question,
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


st.set_page_config(page_title="Azure Assistant", page_icon="☁️", layout="centered")
st.markdown(
    """
    <style>
    .stApp {
        background-color: transparent;
    }
    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        background-image: url("https://azure.microsoft.com/svghandler/azure/?width=600&height=315");
        background-repeat: no-repeat;
        background-position: center 58%;
        background-size: min(46vw, 560px);
        opacity: 0.08;
        pointer-events: none;
        z-index: 0;
    }
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stHeader"] {
        background: transparent;
    }
    [data-testid="stMainBlockContainer"] {
        position: relative;
        z-index: 1;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Azure Assistant")
st.caption(
    "Explore Azure services and solutions with answers grounded in Microsoft Learn."
)
st.info(
    "Answers are based on selected Microsoft Learn documentation. "
    "Follow the source links for current product details."
)

api_key = config_value("GEMINI_API_KEY")
chat_model = config_value("GEMINI_CHAT_MODEL", DEFAULT_CHAT_MODEL)

if "conversations" not in st.session_state:
    conversation_id = uuid4().hex
    st.session_state.conversations = [
        {"id": conversation_id, "title": "New conversation", "messages": []}
    ]
    st.session_state.active_conversation_id = conversation_id

with st.sidebar:
    st.header("Chat history")
    st.caption("Conversations are kept for this app session.")
    if st.button("New chat", use_container_width=True):
        conversation_id = uuid4().hex
        st.session_state.conversations.append(
            {"id": conversation_id, "title": "New conversation", "messages": []}
        )
        st.session_state.conversation_selector = conversation_id
        st.session_state.active_conversation_id = conversation_id
        st.rerun()
    conversation_titles = {
        conversation["id"]: (
            f"New conversation ({index + 1})"
            if conversation["title"] == "New conversation"
            else conversation["title"]
        )
        for index, conversation in enumerate(st.session_state.conversations)
    }
    conversation_ids = list(conversation_titles)
    selected_id = st.session_state.get(
        "conversation_selector",
        st.session_state.active_conversation_id,
    )
    if selected_id not in conversation_ids:
        selected_id = conversation_ids[-1]
        if "conversation_selector" in st.session_state:
            st.session_state.conversation_selector = selected_id
    conversation_id = st.selectbox(
        "Conversations",
        options=conversation_ids,
        format_func=conversation_titles.__getitem__,
        key="conversation_selector",
    )
    st.session_state.active_conversation_id = conversation_id
    conversation = next(
        item for item in st.session_state.conversations if item["id"] == conversation_id
    )
    if st.button("Refresh Learn content", use_container_width=True):
        clear_learn_source_cache()
        st.success("Cached Learn search results and pages cleared.")
    if st.button("Clear current chat", use_container_width=True):
        conversation["title"] = "New conversation"
        conversation["messages"] = []
    st.divider()
    st.markdown("**Try asking:**")
    suggested_questions = (
        "How do I secure an Azure Kubernetes workload?",
        "Compare Azure SQL Database and Cosmos DB.",
        "How do private endpoints work?",
        "How can I monitor an Azure application?",
    )
    for index, suggested_question in enumerate(suggested_questions):
        if st.button(
            suggested_question,
            key=f"suggested-question-{index}",
            use_container_width=True,
        ):
            st.session_state.pending_question = suggested_question

if not api_key:
    st.warning(
        "Add `GEMINI_API_KEY` to your environment or Streamlit deployment secrets to enable the tutor."
    )
    st.stop()

for message in conversation["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        for citation in message.get("citations", []):
            st.markdown(f"- [{citation['title']}]({citation['url']})")

chat_question = st.chat_input("What are you excited to learn about Azure today?")
pending_question = st.session_state.pop("pending_question", None)
question = chat_question or pending_question
if question:
    conversation["messages"].append({"role": "user", "content": question})
    if conversation["title"] == "New conversation":
        conversation["title"] = question[:48] + ("..." if len(question) > 48 else "")
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching Microsoft Learn and preparing an answer..."):
            try:
                answer, documents = answer_azure_question(
                    question,
                    api_key,
                    chat_model,
                    [
                        (message["role"], message["content"])
                        for message in conversation["messages"][:-1]
                    ],
                )
            except APIError as error:
                logger.exception("Gemini API failed while answering an Azure question.")
                answer = gemini_api_help(error)
                documents = []
            except LearnSourceError as error:
                logger.exception("Could not retrieve Microsoft Learn content.")
                answer = f"Could not retrieve the reference content: {error}"
                documents = []
            except RuntimeError as error:
                logger.exception("Could not complete the Gemini answer request.")
                answer = f"Could not complete the answer request: {error}"
                documents = []
        st.markdown(answer)
        citations = []
        seen_urls: set[str] = set()
        if documents:
            st.markdown("**Sources**")
        for document in documents:
            url = str(document.metadata["url"])
            if url in seen_urls:
                continue
            seen_urls.add(url)
            citation = {"title": str(document.metadata["source"]), "url": url}
            citations.append(citation)
            st.markdown(f"- [{citation['title']}]({citation['url']})")
    conversation["messages"].append(
        {"role": "assistant", "content": answer, "citations": citations}
    )
