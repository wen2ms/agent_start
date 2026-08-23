from collections.abc import Iterator

import streamlit as st
from file_history_store import create_conversation, find_empty_conversation, get_session_history, list_conversations
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from rag_service import RagService


def switch_conversation(session_id: str) -> None:
    st.session_state["conversation_id"] = session_id
    st.query_params["conversation_id"] = session_id


def start_new_conversation() -> None:
    session_id = find_empty_conversation()
    if session_id is None:
        session_id = create_conversation()
    switch_conversation(session_id)


st.title("Smart Assistant")
st.divider()

conversations = list_conversations()
conversation_ids = {conversation["id"] for conversation in conversations}
url_session_id = st.query_params.get("conversation_id")
state_session_id = st.session_state.get("conversation_id")

if url_session_id in conversation_ids:
    session_id = url_session_id
elif state_session_id in conversation_ids:
    session_id = state_session_id
elif conversations:
    session_id = conversations[0]["id"]
else:
    session_id = create_conversation()

switch_conversation(session_id)
conversations = list_conversations()

if "rag_service" not in st.session_state:
    st.session_state["rag_service"] = RagService()

with st.sidebar:
    st.button("+ New Conversation", on_click=start_new_conversation, width="stretch")
    st.divider()
    st.caption("Conversation History")
    for conversation in conversations:
        conversation_id = conversation["id"]
        title = conversation["title"]
        if conversation_id == session_id:
            title = f"> {title}"
        st.button(
            title,
            key=f"conversation-{conversation_id}",
            on_click=switch_conversation,
            args=(conversation_id,),
            width="stretch",
        )
        st.caption(conversation["updated_at"])


session_id = st.session_state["conversation_id"]
session_config: RunnableConfig = {"configurable": {"session_id": session_id}}
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
            input={"question": prompt}, config=session_config
        )
        st.write_stream(stream)
