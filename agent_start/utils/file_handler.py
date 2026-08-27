import hashlib
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from utils.logger_handler import logger


def get_file_hash(file_path: Path) -> str | None:
    if not file_path.is_file():
        logger.error("[get_file_hash] %s is not a file.", file_path)
        return None
    try:
        with file_path.open("rb") as infile:
            return hashlib.file_digest(infile, "sha256").hexdigest()
    except OSError:
        logger.error("[get_file_hash] Failed to hash %s.", file_path)
        return None


def listdir(dir_path: Path, allowed_types: tuple[str, ...]) -> tuple[Path, ...]:
    file_paths: list[Path] = []
    if not dir_path.is_dir():
        raise FileNotFoundError(dir_path)
    allowed_suffixes = {suffix.lower() for suffix in allowed_types}
    for file in dir_path.iterdir():
        if file.is_file() and file.suffix.lower() in allowed_suffixes:
            file_paths.append(file)
    return tuple(file_paths)


def load_pdf_documents(file_path: Path, password: str | None = None) -> list[Document]:
    return PyPDFLoader(file_path, password).load()


def load_text_documents(file_path: Path, encoding: str = "utf-8") -> list[Document]:
    return TextLoader(file_path, encoding=encoding).load()


def load_documents(file_path: Path) -> list[Document]:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf_documents(file_path)
    if suffix == ".txt":
        return load_text_documents(file_path)
    raise ValueError(f"Unsupported file type: {suffix}")
