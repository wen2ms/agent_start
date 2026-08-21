from pathlib import Path

from langchain_core.runnables import RunnableConfig

max_file_count = 5
max_single_file_size = 5 * 1024 * 1024
max_total_file_size = 20 * 1024 * 1024

collection_name = "rag"
persist_directory = "chroma_db"

chunk_size = 1000
chunk_overlap = 100
separators = ["\n\n", "\n", ".", "!", "?", " ", ""]
embedding_base_url = "https://ws-yi9oakgdflk8zstn.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
embedding_model = "text-embedding-v4"

retrieval_top_k = 2

chat_base_url = "https://ws-yi9oakgdflk8zstn.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
chat_model = "qwen3.7-flash"

history_dir = Path("chat_histories")
history_prompt_max_messages = 10

session_config: RunnableConfig = {"configurable": {"session_id": "rag001"}}
