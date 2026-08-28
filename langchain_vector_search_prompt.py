from langchain_community.document_loaders import CSVLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompt_values import PromptValue
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def print_content(prompt_value: PromptValue) -> PromptValue:
    print("\n" + "=" * 20 + "Prompt" + "=" * 20)
    print(prompt_value.to_string())
    return prompt_value


def format_documents(documents: list[Document]) -> str:
    return "\n\n".join(document.page_content for document in documents)


loader = CSVLoader(file_path="tv_shows.csv")
documents = loader.load()
embedding = OpenAIEmbeddings(
    base_url="https://ws-yi9oakgdflk8zstn.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
    model="text-embedding-v4",
    check_embedding_ctx_length=False,
)
vector_store = InMemoryVectorStore(embedding=embedding)
vector_store.add_documents(documents)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
chat_llm = ChatOpenAI(
    base_url="https://ws-yi9oakgdflk8zstn.cn-beijing.maas.aliyuncs.com/compatible-mode/v1", model="qwen3.7-flash"
)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            ("You are a TV show recommendation assistant. Answer the user's question based on the reference data."),
        ),
        (
            "human",
            ("Reference data:\n{context}\n\nQuestion:\n{question}"),
        ),
    ]
)
str_parser = StrOutputParser()
chain = (
    RunnableParallel(context=retriever | format_documents, question=RunnablePassthrough())
    # | {"question": RunnablePassthrough(), "context": retriever | format_documents}
    | prompt
    | print_content
    | chat_llm
    | str_parser
)
user_input = "Tell me about Breaking Bad."
result = chain.invoke(input=user_input)
print("\n" + "=" * 20 + "Answer" + "=" * 20)
print(result)
