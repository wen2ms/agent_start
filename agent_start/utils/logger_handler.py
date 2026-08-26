import logging
from datetime import datetime
from pathlib import Path

LOG_ROOT = Path("logs")
DEFAULT_LOG_FORMATTER = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
)


def get_logger(
    name: str = "agent", console_level: int = logging.INFO, file_level: int = logging.INFO, log_file: Path | None = None
) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(min(console_level, file_level))
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(DEFAULT_LOG_FORMATTER)
    logger.addHandler(console_handler)
    if log_file is None:
        log_file = LOG_ROOT / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(DEFAULT_LOG_FORMATTER)
    logger.addHandler(file_handler)
    return logger


logger = get_logger()


if __name__ == "__main__":
    logger.info("Information log")
    logger.error("Error log")
    logger.warning("Warning log")
    logger.debug("Debug log")
