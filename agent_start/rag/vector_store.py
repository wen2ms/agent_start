from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from model.factory import embedding
from rag.knowledge_base import KnowledgeBaseService
from utils.config_handler import chroma_config


class VectorStoreService:
    def __init__(self) -> None:
        self.chroma = Chroma(
            collection_name=chroma_config["collection_name"],
            embedding_function=embedding,
            persist_directory=chroma_config["persist_directory"],
        )

    def get_retriever(self) -> VectorStoreRetriever:
        return self.chroma.as_retriever(search_kwargs={"k": chroma_config["retrieval_top_k"]})

    def load_knowledge_base(self) -> None:
        knowledge_base_service = KnowledgeBaseService(self.chroma)
        knowledge_base_service.upload_by_directory(Path(chroma_config["data_dir"]))


if __name__ == "__main__":
    vector_store_service = VectorStoreService()
    vector_store_service.load_knowledge_base()
    retriever = vector_store_service.get_retriever()
    documents = retriever.invoke(input="为什么夜间基础耗电这么高？")
    for index, document in enumerate(documents):
        print("\n" + "=" * 20 + f"Document {index}" + "=" * 20)
        print(document)
