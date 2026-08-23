import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import config_data
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, messages_from_dict, messages_to_dict


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


def create_conversation() -> str:
    session_id = uuid4().hex
    get_session_history(session_id).clear()
    return session_id


def get_conversation_title(messages: list[BaseMessage]) -> str:
    for message in messages:
        if isinstance(message, HumanMessage) and isinstance(message.content, str):
            title = " ".join(message.content.split())
            return title[:24] if len(title) > 24 else title
    return "New Conversation"


def list_conversations() -> list[dict[str, str]]:
    config_data.history_dir.mkdir(parents=True, exist_ok=True)
    conversations: list[dict[str, str]] = []
    for history_file in config_data.history_dir.iterdir():
        session_id = history_file.name
        history = get_session_history(session_id)
        messages = history.messages
        modified_at = datetime.fromtimestamp(history_file.stat().st_mtime)
        conversations.append(
            {
                "id": session_id,
                "title": get_conversation_title(messages),
                "updated_at": modified_at.strftime("%m-%d %H:%M"),
            }
        )
    conversations.sort(key=lambda conversation: conversation["updated_at"], reverse=True)
    return conversations


def find_empty_conversation() -> str | None:
    config_data.history_dir.mkdir(parents=True, exist_ok=True)
    for history_file in config_data.history_dir.iterdir():
        session_id = history_file.name
        history = get_session_history(session_id)
        if not history.messages:
            return session_id
    return None
