"""Bounded persistent troubleshooting logs, independent of frame cleanup."""
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path

from .config import directory


def configure_logging() -> Path:
    folder = directory("LOG_DIR", "logs").resolve()
    folder.mkdir(parents=True, exist_ok=True)
    max_bytes = int(os.getenv("LOG_MAX_BYTES", "10485760"))
    backups = int(os.getenv("LOG_BACKUP_COUNT", "5"))
    if max_bytes < 1024 or not 1 <= backups <= 100:
        raise ValueError("LOG_MAX_BYTES must be >= 1024; LOG_BACKUP_COUNT must be 1–100.")
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    if level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        raise ValueError("Invalid LOG_LEVEL.")
    root = logging.getLogger()
    root.setLevel(level)
    formatter = logging.Formatter("%(asctime)s %(levelname)s [%(threadName)s] %(name)s: %(message)s")
    target = folder / "app.log"
    if not any(isinstance(handler, logging.StreamHandler) and
               not isinstance(handler, logging.FileHandler) for handler in root.handlers):
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        root.addHandler(console)
    if not any(isinstance(handler, logging.FileHandler) and
               Path(handler.baseFilename) == target for handler in root.handlers):
        handler = RotatingFileHandler(target, maxBytes=max_bytes, backupCount=backups,
                                      encoding="utf-8", errors="backslashreplace")
        handler.setFormatter(formatter)
        root.addHandler(handler)
    logging.captureWarnings(True)
    logging.getLogger(__name__).info("Persistent log: %s (rotation: %s bytes, %s backups)",
                                    target, max_bytes, backups)
    return target


def record_process_log(path: Path, returncode: int | None) -> None:
    """Copy child output to rotating logs before its job workspace is removed."""
    logger = logging.getLogger("pipeline.subprocess")
    label = f"{path.parent.name}/{path.name}"
    logger.info("Subprocess diagnostic %s (exit=%s)", label, returncode)
    try:
        with path.open("r", encoding="utf-8", errors="replace") as source:
            while chunk := source.read(4096):
                logger.log(logging.ERROR if returncode else logging.INFO,
                           "[%s] %s", label, chunk.rstrip())
    except OSError:
        logger.warning("Unable to preserve subprocess diagnostic %s", path, exc_info=True)