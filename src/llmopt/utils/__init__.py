"""Utility helpers for :mod:`llmopt`."""

from __future__ import annotations

from llmopt.utils.logging import get_logger, log_once, setup_logging, timed
from llmopt.utils.metrics import Counter, Histogram, MetricRegistry
from llmopt.utils.misc import (
    atomic_write_json,
    device_of,
    dtype_from_str,
    estimate_kv_cache_bytes,
    human_bytes,
    move_to_device,
    resolve_dtype,
    seed_everything,
    stopwatch,
)

__all__ = [
    "Counter",
    "Histogram",
    "MetricRegistry",
    "atomic_write_json",
    "device_of",
    "dtype_from_str",
    "estimate_kv_cache_bytes",
    "get_logger",
    "human_bytes",
    "log_once",
    "move_to_device",
    "resolve_dtype",
    "seed_everything",
    "setup_logging",
    "stopwatch",
    "timed",
]
