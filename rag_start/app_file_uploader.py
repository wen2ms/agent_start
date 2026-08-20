import time

import streamlit as st
from knowledge_base import KnowledgeBaseService

st.title("Knowledge Update Service")

uploaded_file = st.file_uploader(label="Upload .txt file", type=["txt"])

if "knowledge_base_service" not in st.session_state:
    st.session_state["knowledge_base_service"] = KnowledgeBaseService()

if uploaded_file is not None:
    file_name = uploaded_file.name
    file_type = uploaded_file.type
    file_size = uploaded_file.size / 1024

    st.subheader(f"Filename: {file_name}")
    st.write(f"Type: {file_type} | Size: {file_size:.2f} KB")

    text = uploaded_file.getvalue().decode("utf-8")
    with st.spinner("Uploading Knowledge..."):
        time.sleep(1)
        result = st.session_state["knowledge_base_service"].upload_by_str(text, file_name)
        st.write(result)
