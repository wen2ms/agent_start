from operator import itemgetter

import config_data
from file_history_store import get_session_history
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompt_values import PromptValue
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
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
            chunk_size=config_data.embedding_batch_size,
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
                        "Answer the user's question based on the reference data. "
                        "Use the chat history only to understand conversational context "
                        "and follow-up questions. "
                        "For factual claims, rely on the current reference context rather "
                        "than previous assistant responses. "
                        "If the latest user question conflicts with earlier messages, "
                        "follow the latest user question. "
                        "When using information from the reference context, "
                        "cite the exact source file name in the format "
                        "[Source: filename]. "
                    ),
                ),
                MessagesPlaceholder("history", n_messages=config_data.history_prompt_max_messages),
                ("user", "Reference data:\n{context}\n\nQuestion:\n{question}"),
            ]
        )

        str_parser = StrOutputParser()
        base_chain = (
            RunnablePassthrough.assign(context=itemgetter("question") | retriever | format_documents)
            | prompt
            | print_content
            | self.chat_llm
            | str_parser
        )
        conversation_chain = RunnableWithMessageHistory(
            base_chain, get_session_history, input_messages_key="question", history_messages_key="history"
        )
        return conversation_chain
