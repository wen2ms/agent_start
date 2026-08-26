from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: Path | str, encoding: str = "utf-8") -> dict[str, Any]:
    config_path = Path(config_path)
    with config_path.open("r", encoding=encoding) as infile:
        return yaml.safe_load(infile)


rag_config = load_config("configs/rag.yaml")
chroma_config = load_config("configs/chroma.yaml")
prompts_config = load_config("configs/prompts.yaml")
agent_config = load_config("configs/agent.yaml")


if __name__ == "__main__":
    print(rag_config["embedding_base_url"])
