import json
from collections.abc import Sequence
from pathlib import Path

from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

HISTORY_DIR = Path("chat_histories")


class FileChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, history_dir: Path, session_id: str) -> None:
        self.history_dir = history_dir
        self.session_id = session_id
        self.history_file = self.history_dir / self.session_id

    @property
    def messages(self) -> list[BaseMessage]:
        try:
            data = json.loads(self.history_file.read_text(encoding="utf-8"))
            return messages_from_dict(data)
        except FileNotFoundError:
            return []

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        all_messages = self.messages
        all_messages.extend(messages)
        data = messages_to_dict(all_messages)
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.history_file.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")

    def clear(self) -> None:
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.history_file.write_text("[]")


# messages_dict: dict[str, InMemoryChatMessageHistory] = {}


# def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
#     if session_id not in messages_dict:
#         messages_dict[session_id] = InMemoryChatMessageHistory()
#     return messages_dict[session_id]
def get_session_history(session_id: str) -> FileChatMessageHistory:
    return FileChatMessageHistory(history_dir=HISTORY_DIR, session_id=session_id)


def history_demo() -> None:
    json_parser = JsonOutputParser()
    str_parser = StrOutputParser()
    chat_llm = ChatOpenAI(
        base_url="https://ws-yi9oakgdflk8zstn.cn-beijing.maas.aliyuncs.com/compatible-mode/v1", model="qwen3.7-flash"
    )
    prompt1 = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a Python question rewriter.

Use the conversation history to resolve references such as
"it", "that", "this method", or "the previous one".

Rewrite the user's question as a standalone question.

Return your answer as JSON in exactly this format:
{{
    "question": "the standalone Python question",
    "style": "short"
}}""",
            ),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )

    prompt2 = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a Python expert. Answer clearly and accurately.",
            ),
            (
                "human",
                """Question: {question}
Answer style: {style}

Answer the question directly.""",
            ),
        ]
    )
    base_chain = prompt1 | chat_llm | json_parser | prompt2 | chat_llm | str_parser
    chain = RunnableWithMessageHistory(
        base_chain, get_session_history, input_messages_key="input", history_messages_key="history"
    )
    config: RunnableConfig = {"configurable": {"session_id": "langchain_history"}}
    response1 = chain.invoke(input={"input": "What does __repr__ stand for?"}, config=config)
    print("\n" + "=" * 20 + "Answer" + "=" * 20)
    print(response1)
    response2 = chain.invoke(input={"input": "Can you explain the previous method in more detail?"}, config=config)
    print("\n" + "=" * 20 + "Answer" + "=" * 20)
    print(response2)


if __name__ == "__main__":
    history_demo()
