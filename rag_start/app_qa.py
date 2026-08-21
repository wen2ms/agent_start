from collections.abc import Iterator

import config_data
import streamlit as st
from file_history_store import get_session_history
from langchain_core.messages import AIMessage, HumanMessage
from rag_service import RagService

st.title("Smart Assistant")
st.divider()

if "rag_service" not in st.session_state:
    st.session_state["rag_service"] = RagService()

session_id = config_data.session_config["configurable"]["session_id"]
history = get_session_history(session_id)

st.chat_message("assistant").write("What can I do for you?")
for message in history.messages:
    if isinstance(message, HumanMessage):
        role = "user"
    elif isinstance(message, AIMessage):
        role = "assistant"
    else:
        continue
    st.chat_message(role).write(message.content)

prompt = st.chat_input("Ask a question")

if prompt:
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"), st.spinner("Thinking..."):
        stream: Iterator[str] = st.session_state["rag_service"].chain.stream(
            input={"question": prompt}, config=config_data.session_config
        )
        st.write_stream(stream)
