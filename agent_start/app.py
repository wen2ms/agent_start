from collections.abc import Iterator

import streamlit as st
from agent.react_agent import ReactAgent

st.title("Smart home energy assistant")
st.divider()


if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent()

if "messages" not in st.session_state:
    st.session_state["messages"] = []

st.chat_message("assistant").write("What can I do for you?")
for message in st.session_state["messages"]:
    st.chat_message(message["role"]).write(message["content"])

prompt = st.chat_input("Ask a question")
if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["messages"].append({"role": "user", "content": prompt})
    response_messages: list[str] = []
    with st.chat_message("assistant"), st.spinner("Thinking..."):
        stream: Iterator[str] = st.session_state["agent"].execute_stream(query=prompt, user_id="user_001")

        def capture(generator: Iterator[str], cache_list: list) -> Iterator[str]:
            for chunk in generator:
                cache_list.append(chunk)
                yield chunk

        st.write_stream(capture(stream, response_messages))
        st.session_state["messages"].append({"role": "assistant", "content": response_messages[-1]})
    st.rerun()
