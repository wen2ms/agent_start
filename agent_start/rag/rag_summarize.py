from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompt_values import PromptValue
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough
from model.factory import chat_llm
from rag.vector_store import VectorStoreService
from utils.prompt_loader import PromptKey, load_prompt


def format_documents(documents: list[Document]) -> str:
    return "\n\n".join(
        f"Source: {document.metadata['source']}\nContent:\n{document.page_content}" for document in documents
    )


def print_content(prompt_value: PromptValue) -> PromptValue:
    print("\n" + "=" * 20 + "Prompt" + "=" * 20)
    print(prompt_value.to_string())
    return prompt_value


class RagSummarizeService:
    def __init__(self) -> None:
        self.vector_store = VectorStoreService()
        self.retriever = self.vector_store.get_retriever()
        self.system_prompt = load_prompt(PromptKey.RAG_SUMMARIZE)
        self.prompt_template = ChatPromptTemplate.from_messages(
            [("system", self.system_prompt), ("human", "Reference data:\n{context}\n\nQuestion:\n{question}")]
        )
        self.chat_llm = chat_llm
        self.chain = self._init_chain()

    def _init_chain(self) -> Runnable:
        chain = (
            RunnableParallel(context=self.retriever | format_documents, question=RunnablePassthrough())
            | self.prompt_template
            | print_content
            | self.chat_llm
            | StrOutputParser()
        )
        return chain

    def summarize(self, question: str) -> str:
        return self.chain.invoke(input=question)


if __name__ == "__main__":
    rag_summarize_service = RagSummarizeService()
    response = rag_summarize_service.summarize("智能家庭能源管理是什么？")
    print("\n" + "=" * 20 + "Answer" + "=" * 20)
    print(response)
