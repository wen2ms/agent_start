import json
from collections.abc import Sequence
from pathlib import Path

import config_data
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict


class FileChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, history_dir: Path, session_id: str) -> None:
        self.history_dir = history_dir
        self.session_id = session_id
        self.history_file = history_dir / session_id

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
        self.history_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def clear(self) -> None:
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.history_file.write_text("[]")


def get_session_history(session_id: str) -> FileChatMessageHistory:
    return FileChatMessageHistory(history_dir=config_data.history_dir, session_id=session_id)
