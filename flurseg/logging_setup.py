"""Consistent console and rotating-file logging."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from flurseg.paths import RepoPaths


def configure_logging(paths: RepoPaths, level: int = logging.INFO) -> logging.Logger:
    """Configure the Flurseg logger once for a repository root."""
    paths.logs_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("flurseg")
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    file_handler = RotatingFileHandler(
        paths.log_file,
        encoding="utf-8",
        maxBytes=2_000_000,
        backupCount=3,
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger
