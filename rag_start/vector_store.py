import config_data
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever


class VectorStoreService:
    def __init__(self, embedding: Embeddings) -> None:
        self.embedding = embedding
        self.vector_store = Chroma(
            collection_name=config_data.collection_name,
            embedding_function=embedding,
            persist_directory=config_data.persist_directory,
        )

    def get_retriever(self) -> VectorStoreRetriever:
        return self.vector_store.as_retriever(search_kwargs={"k": config_data.retrieval_top_k})


if __name__ == "__main__":
    from langchain_openai import OpenAIEmbeddings

    embedding = OpenAIEmbeddings(
        base_url=config_data.embedding_base_url,
        model=config_data.embedding_model,
        check_embedding_ctx_length=False,
    )
    vector_store_service = VectorStoreService(embedding)
    retriever = vector_store_service.get_retriever()
    documents = retriever.invoke(input="What should I wear for a job interview?")
    for index, document in enumerate(documents):
        print("\n" + "=" * 20 + f"Document {index}" + "=" * 20)
        print(document)
