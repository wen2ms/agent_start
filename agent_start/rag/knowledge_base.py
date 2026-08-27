from datetime import datetime
from pathlib import Path

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.config_handler import chroma_config
from utils.file_handler import get_file_hash, listdir, load_documents
from utils.logger_handler import logger


class KnowledgeBaseService:
    def __init__(self, chroma: Chroma) -> None:
        self.chroma = chroma
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_config["chunk_size"],
            chunk_overlap=chroma_config["chunk_overlap"],
            separators=chroma_config["separators"],
            length_function=len,
        )

    def upload_by_file(self, file_path: Path) -> bool:
        content_hash = get_file_hash(file_path)
        if content_hash is None:
            return False
        first_chunk_id = f"{content_hash}:0"
        existing = self.chroma.get(ids=[first_chunk_id])
        if existing["ids"]:
            logger.info("[Skipped] Content already exists: %s", file_path)
            return False
        documents = load_documents(file_path)
        if not documents:
            logger.info("[Skipped] No content loaded: %s", file_path)
            return False
        created_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for document in documents:
            document.metadata.update(
                {"source": file_path.name, "created_time": created_time, "content_hash": content_hash}
            )
        chunks = self.text_splitter.split_documents(documents)
        ids = [f"{content_hash}:{index}" for index in range(len(chunks))]
        self.chroma.add_documents(documents=chunks, ids=ids)
        logger.info("[Success] Indexed %s", file_path)
        return True

    def upload_by_directory(self, directory: Path) -> None:
        file_paths = listdir(directory, chroma_config["allowed_knowledge_types"])
        for file_path in file_paths:
            self.upload_by_file(file_path)
