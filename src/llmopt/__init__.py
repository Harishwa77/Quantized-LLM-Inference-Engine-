"""llmopt: an LLM inference optimization engine.

The package layers bottom-up:

* ``llmopt.quantization`` -- GPTQ / round-to-nearest weight quantization.
* ``llmopt.cache`` -- paged KV cache, block pool, and prefix cache.
* ``llmopt.attention`` -- paged attention with flash-decoding style splits.
* ``llmopt.model`` -- decoder-only transformer with a cache-aware forward.
* ``llmopt.engine`` -- continuous batching scheduler, model runner, sampler.
* ``llmopt.serving`` -- OpenAI-compatible HTTP surface.
* ``llmopt.benchmark`` -- throughput / latency measurement harness.
"""

from __future__ import annotations

__version__ = "0.1.0"

from llmopt.config import (
    CacheConfig,
    EngineConfig,
    ModelConfig,
    QuantConfig,
    SchedulerConfig,
)

__all__ = [
    "CacheConfig",
    "EngineConfig",
    "ModelConfig",
    "QuantConfig",
    "SchedulerConfig",
    "__version__",
]


def __getattr__(name: str) -> object:
    """Import the heavy submodules lazily so ``import llmopt`` stays cheap."""
    lazy = {
        "LLMEngine": "llmopt.engine",
        "EngineOutput": "llmopt.engine",
        "SamplingParams": "llmopt.engine",
        "Request": "llmopt.engine",
        "DecoderOnlyModel": "llmopt.model",
        "tiny_model": "llmopt.model",
        "PagedKVCache": "llmopt.cache",
        "BlockPool": "llmopt.cache",
        "PrefixCache": "llmopt.cache",
        "PagedAttention": "llmopt.attention",
        "quantize_model_": "llmopt.quantization",
    }
    if name in lazy:
        import importlib

        return getattr(importlib.import_module(lazy[name]), name)
    raise AttributeError(f"module 'llmopt' has no attribute {name!r}")
