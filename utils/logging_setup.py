"""Shared rotating file + stdout logging for BIRD pipeline scripts."""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_CONFIGURED = False
_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
_BACKUP_COUNT = 5


def setup_logging(log_path: str | os.PathLike[str], level: int = logging.INFO) -> logging.Logger:
    """Configure root logger with RotatingFileHandler + stdout StreamHandler.

    Idempotent: subsequent calls with the same or different path are no-ops
    once handlers are already attached.
    """
    global _CONFIGURED
    root = logging.getLogger()
    if _CONFIGURED or root.handlers:
        _CONFIGURED = True
        return get_logger("bird")

    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    root.setLevel(level)
    formatter = logging.Formatter(_LOG_FORMAT)

    file_handler = RotatingFileHandler(
        path,
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)

    root.addHandler(file_handler)
    root.addHandler(stream_handler)
    _CONFIGURED = True
    return get_logger("bird")


def get_logger(name: str = "bird") -> logging.Logger:
    """Thin wrapper around logging.getLogger."""
    return logging.getLogger(name)


def resolve_log_path(cli_log_file: str | None, default_path: str | os.PathLike[str]) -> str:
    """Prefer CLI --log-file, then default_path."""
    if cli_log_file:
        return cli_log_file
    return str(default_path)


def sibling_log_path(anchor: str | os.PathLike[str], suffix: str = ".log") -> str:
    """Return path next to ``anchor`` with stem + suffix (e.g. predict.json -> predict.log)."""
    p = Path(anchor)
    return str(p.with_name(p.stem + suffix))
