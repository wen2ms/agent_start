from enum import StrEnum
from pathlib import Path

from utils.config_handler import prompts_config


class PromptKey(StrEnum):
    MAIN = "main_prompt_path"
    RAG_SUMMARIZE = "rag_summarize_prompt_path"
    REPORT = "report_prompt_path"


def load_prompt(prompt_path: PromptKey, encoding: str = "utf-8") -> str:
    return Path(prompts_config[prompt_path]).read_text(encoding=encoding)


if __name__ == "__main__":
    print(load_prompt(PromptKey.MAIN))
