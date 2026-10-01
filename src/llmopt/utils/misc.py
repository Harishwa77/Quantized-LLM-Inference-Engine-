"""Assorted helpers: seeding, dtype resolution, device placement, timing."""

from __future__ import annotations

import contextlib
import os
import random
import time
from collections.abc import Iterator
from typing import Any

__all__ = [
    "atomic_write_json",
    "device_of",
    "dtype_from_str",
    "estimate_kv_cache_bytes",
    "human_bytes",
    "move_to_device",
    "resolve_dtype",
    "seed_everything",
    "stopwatch",
]

_DTYPES = {
    "float32": "float32",
    "fp32": "float32",
    "float": "float32",
    "float16": "float16",
    "fp16": "float16",
    "half": "float16",
    "bfloat16": "bfloat16",
    "bf16": "bfloat16",
    "float8_e4m3fn": "float8_e4m3fn",
    "float8_e5m2": "float8_e5m2",
}


def dtype_from_str(name: str) -> Any:
    """Resolve a dtype name (or ``auto``) to a real ``torch.dtype``."""
    import torch

    if name in ("auto", "none", ""):
        if torch.cuda.is_available():
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32
    key = _DTYPES.get(name.lower())
    if key is None:
        raise ValueError(f"unsupported dtype: {name!r} (known: {sorted(_DTYPES)})")
    return getattr(torch, key)


def resolve_dtype(name: str, device: str = "cpu") -> Any:
    """Like :func:`dtype_from_str` but demotes unsupported dtypes to fp32."""
    import torch

    dtype = dtype_from_str(name)
    if device == "cpu" and dtype in (torch.float16, torch.float8_e4m3fn, torch.float8_e5m2):
        return torch.float32
    return dtype


def device_of(preferred: str = "auto") -> Any:
    """Return a ``torch.device`` for ``preferred`` (``auto`` picks the best)."""
    import torch

    if preferred != "auto":
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move_to_device(obj: Any, device: Any) -> Any:
    """Recursively move tensors inside containers to ``device``."""
    import torch

    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, torch.nn.Module):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(move_to_device(v, device) for v in obj)
    return obj


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed python, numpy (if present) and torch RNGs."""
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32))
    except ImportError:  # pragma: no cover - numpy is a torch dependency
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - CPU test environment
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@contextlib.contextmanager
def stopwatch(label: str = "", verbose: bool = True) -> Iterator[dict[str, float]]:
    """Context manager that records wall time under the ``seconds`` key."""
    info: dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield info
    finally:
        info["seconds"] = time.perf_counter() - start
        if verbose and label:
            print(f"[stopwatch] {label}: {info['seconds'] * 1000:.2f} ms")


def estimate_kv_cache_bytes(
    num_layers: int,
    num_key_value_heads: int,
    head_dim: int,
    num_blocks: int,
    block_size: int,
    itemsize: int = 2,
) -> int:
    """Bytes required for a paged KV cache (keys and values)."""
    tokens = num_blocks * block_size
    return 2 * num_layers * num_key_value_heads * head_dim * tokens * itemsize


def human_bytes(num: float) -> str:
    """Render a byte count with binary units."""
    step = 1024.0
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(num) < step or unit == "TiB":
            return f"{num:,.1f} {unit}" if unit != "B" else f"{int(num)} B"
        num /= step
    return f"{num:,.1f} TiB"  # pragma: no cover - unreachable


def atomic_write_json(path: str, payload: Any, indent: int = 2) -> None:
    """Write JSON via a temp file + replace so readers never see partial data."""
    import json
    import tempfile

    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=indent, sort_keys=True)
            handle.write("\n")
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
