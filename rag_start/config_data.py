from pathlib import Path

md5_path = Path("md5.txt")

collection_name = "rag"
persist_directory = Path("chroma_db")

chunk_size = 1000
chunk_overlap = 100
separators = ["\n\n", "\n", ".", "!", "?", " ", ""]
embedding_base_url = "https://ws-yi9oakgdflk8zstn.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
embedding_model = "text-embedding-v4"
