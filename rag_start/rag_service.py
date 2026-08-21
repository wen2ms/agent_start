import config_data
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompt_values import PromptValue
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from vector_store import VectorStoreService


def print_content(prompt_value: PromptValue) -> PromptValue:
    print("\n" + "=" * 20 + "Prompt" + "=" * 20)
    print(prompt_value.to_string())
    return prompt_value


def format_documents(documents: list[Document]) -> str:
    return "\n\n".join(
        f"Source: {document.metadata['source']}\nContent:\n{document.page_content}" for document in documents
    )


class RagService:
    def __init__(self) -> None:
        embedding = OpenAIEmbeddings(
            base_url=config_data.embedding_base_url,
            model=config_data.embedding_model,
            check_embedding_ctx_length=False,
        )
        self.vector_store = VectorStoreService(embedding)
        self.chat_llm = ChatOpenAI(base_url=config_data.chat_base_url, model=config_data.chat_model)
        self.chain = self._get_chain()

    def _get_chain(self) -> Runnable:
        retriever = self.vector_store.get_retriever()
        prompt = ChatPromptTemplate(
            [
                (
                    "system",
                    (
                        "You are a retrieval-augmented assistant. "
                        "Answer the user's question based on the reference data."
                        "When using information from the reference context, "
                        "cite the exact source file name in the format "
                        "[Source: filename]. "
                    ),
                ),
                ("user", "Reference data:\n{context}\n\nQuestion:\n{question}"),
            ]
        )

        str_parser = StrOutputParser()
        chain = (
            RunnableParallel(context=retriever | format_documents, question=RunnablePassthrough())
            | prompt
            | print_content
            | self.chat_llm
            | str_parser
        )
        return chain


if __name__ == "__main__":
    response = RagService().chain.invoke(input="What size should I choose?")
    print("\n" + "=" * 20 + "Answer" + "=" * 20)
    print(response)
