from langchain_chroma import Chroma
from langchain_community.document_loaders import CSVLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

loader = CSVLoader(
    file_path="vector_stores_example.csv",
    metadata_columns=("id", "category"),
    content_columns=("title", "description"),
)
documents = loader.load()
embedding = OpenAIEmbeddings(
    base_url="https://ws-yi9oakgdflk8zstn.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
    model="text-embedding-v4",
    check_embedding_ctx_length=False,
)
# vector_store = InMemoryVectorStore(embedding=embeddings)
vector_store = Chroma(
    collection_name="vector_stores_exmaple", embedding_function=embedding, persist_directory="./chroma_db"
)
document_ids = [str(document.metadata["id"]) for document in documents]
added_ids = vector_store.add_documents(documents=documents, ids=document_ids)
query = "How can I store text as vectors and search by meaning?"
results = vector_store.similarity_search(query=query, k=3)
print(results)
vector_store.delete(ids=["p003"])
query = "What does a vector store do?"
results = vector_store.similarity_search(query=query, k=3)
print("\n", results)
