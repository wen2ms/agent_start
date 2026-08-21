import config_data
import streamlit as st
from knowledge_base import KnowledgeBaseService

st.title("Knowledge Base Update Service")

uploaded_files = st.file_uploader(
    label="Upload .txt files",
    type=["txt"],
    accept_multiple_files=True,
    max_upload_size=config_data.max_single_file_size // (1024 * 1024),
)
if not uploaded_files:
    st.stop()
if len(uploaded_files) > config_data.max_file_count:
    st.error(f"You can upload up to {config_data.max_file_count} files at a time.")
    st.stop()
total_file_size = sum(uploaded_file.size for uploaded_file in uploaded_files)
if total_file_size > config_data.max_total_file_size:
    st.error(f"The total upload size cannot exceed {config_data.max_total_file_size}.")
    st.stop()

if "knowledge_base_service" not in st.session_state:
    st.session_state["knowledge_base_service"] = KnowledgeBaseService()

for uploaded_file in uploaded_files:
    file_name = uploaded_file.name
    file_type = uploaded_file.type
    file_size_kib = uploaded_file.size / 1024
    st.subheader(f"Filename: {file_name}")
    st.write(f"Type: {file_type} | Size: {file_size_kib:.2f} KiB")
    text = uploaded_file.getvalue().decode("utf-8")
    with st.spinner("Uploading Knowledge..."):
        result = st.session_state["knowledge_base_service"].upload_by_str(text, file_name)
        st.write(result)
