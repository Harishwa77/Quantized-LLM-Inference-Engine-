"""Logging helpers with an opt-in structured format."""

from __future__ import annotations

import contextlib
import logging
import os
import time
from collections.abc import Iterator
from typing import Any

__all__ = ["get_logger", "setup_logging", "timed", "log_once"]

_CONFIGURED = False
_SEEN: set[tuple[str, str]] = set()

_VERBOSE = {"1", "true", "yes", "on"}

_FORMAT = "%(asctime)s %(levelname)-7s %(name)-28s %(message)s"


def setup_logging(level: str | int | None = None) -> None:
    """Install a single stderr handler. Safe to call repeatedly."""
    global _CONFIGURED
    root = logging.getLogger("llmopt")
    if level is None:
        level = os.environ.get("LLMOPT_LOG_LEVEL", "INFO")
    if isinstance(level, str):
        level = logging.DEBUG if level.lower() in _VERBOSE else getattr(
            logging, level.upper(), logging.INFO
        )
    root.setLevel(level)
    if not _CONFIGURED:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FORMAT))
        root.addHandler(handler)
        root.propagate = False
        _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced child logger, configuring the root once."""
    setup_logging()
    if not name.startswith("llmopt"):
        name = f"llmopt.{name}"
    return logging.getLogger(name)


def log_once(logger: logging.Logger, level: int, message: str, *args: Any) -> None:
    """Emit ``message`` a single time per process, keyed on the format string."""
    key = (logger.name, message)
    if key in _SEEN:
        return
    _SEEN.add(key)
    logger.log(level, message, *args)


@contextlib.contextmanager
def timed(logger: logging.Logger, label: str, level: int = logging.DEBUG) -> Iterator[dict[str, Any]]:
    """Time a block and record the duration on the yielded dict."""
    info: dict[str, Any] = {}
    start = time.perf_counter()
    try:
        yield info
    finally:
        info["seconds"] = time.perf_counter() - start
        logger.log(level, "%s took %.3fs", label, info["seconds"])
