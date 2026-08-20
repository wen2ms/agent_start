import hashlib
from datetime import datetime

import config_data
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_content_hash(content: str, encoding: str = "utf-8") -> str:
    str_bytes = content.encode(encoding=encoding)
    return hashlib.sha256(str_bytes).hexdigest()


class KnowledgeBaseService:
    def __init__(self) -> None:
        config_data.persist_directory.mkdir(parents=True, exist_ok=True)
        embedding = OpenAIEmbeddings(
            base_url=config_data.embedding_base_url,
            model=config_data.embedding_model,
            check_embedding_ctx_length=False,
        )
        self.chroma = Chroma(
            collection_name=config_data.collection_name,
            embedding_function=embedding,
            persist_directory=str(config_data.persist_directory),
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config_data.chunk_size,
            chunk_overlap=config_data.chunk_overlap,
            separators=config_data.separators,
            length_function=len,
        )

    def upload_by_str(self, content: str, source: str) -> str:
        content_hash = get_content_hash(content)
        first_chunk_id = f"{content_hash}:0"
        existing = self.chroma.get(ids=[first_chunk_id])
        if existing["ids"]:
            return "[Skipped]Content already exists."
        created_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        document = Document(
            page_content=content,
            metadata={"source": source, "created_time": created_time, "content_hash": content_hash},
        )
        documents = self.text_splitter.split_documents([document])
        ids = [f"{content_hash}:{index}" for index in range(len(documents))]
        self.chroma.add_documents(documents, ids=ids)
        return "[Success]Content uploaded."


if __name__ == "__main__":
    knowledge_base_service = KnowledgeBaseService()
    print(knowledge_base_service.upload_by_str("Sass", "test_file"))
    print(knowledge_base_service.upload_by_str("Sass", "test_file"))
