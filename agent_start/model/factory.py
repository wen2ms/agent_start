from abc import ABC, abstractmethod

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from utils.config_handler import rag_config


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Embeddings | BaseChatModel:
        pass


class ChatModelFactory(BaseModelFactory):
    def generator(self) -> ChatOpenAI:
        return ChatOpenAI(base_url=rag_config["chat_base_url"], model=rag_config["chat_model"])


class EmbeddingFactory(BaseModelFactory):
    def generator(self) -> OpenAIEmbeddings:
        return OpenAIEmbeddings(
            base_url=rag_config["embedding_base_url"],
            model=rag_config["embedding_model"],
            check_embedding_ctx_length=False,
            chunk_size=rag_config["embedding_batch_size"],
        )


chat_llm = ChatModelFactory().generator()
embedding = EmbeddingFactory().generator()
