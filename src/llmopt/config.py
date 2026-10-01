"""Configuration objects for the inference engine.

Every tunable lives in a frozen-ish dataclass so that a whole engine run can be
described, serialized and reproduced from a single object graph.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from typing import Any

__all__ = [
    "CacheConfig",
    "EngineConfig",
    "ModelConfig",
    "QuantConfig",
    "SchedulerConfig",
    "ServerConfig",
    "num_blocks_for_gpu_memory",
]

_TRUE = {"1", "true", "yes", "on"}


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in _TRUE


@dataclasses.dataclass(slots=True)
class ModelConfig:
    """Architecture description for a decoder-only transformer."""

    vocab_size: int = 32000
    hidden_size: int = 1024
    intermediate_size: int = 2816
    num_hidden_layers: int = 12
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = False
    attn_implementation: str = "auto"
    torch_dtype: str = "bfloat16"

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be divisible by "
                f"num_key_value_heads ({self.num_key_value_heads})"
            )
        if self.intermediate_size % 2 != 0:
            raise ValueError("intermediate_size must be even for SwiGLU")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def num_query_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ModelConfig:
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in fields})


@dataclasses.dataclass(slots=True)
class QuantConfig:
    """Post-training weight/activation quantization settings."""

    bits: int = 8
    group_size: int = 128
    symmetric: bool = True
    strategy: str = "gptq"
    ignore_layers: tuple[str, ...] = ("lm_head", "embed_tokens")
    calibrate_num_samples: int = 128
    calibrate_seq_length: int = 512
    damping: float = 0.01
    act_order: bool = True

    def __post_init__(self) -> None:
        if self.bits not in (2, 4, 8):
            raise ValueError(f"unsupported bit width: {self.bits}")
        if self.group_size <= 0 or self.group_size % 2 != 0:
            raise ValueError("group_size must be a positive even number")
        if self.strategy not in ("gptq", "rtn", "none"):
            raise ValueError(f"unknown quantization strategy: {self.strategy}")

    @property
    def enabled(self) -> bool:
        return self.strategy != "none"

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> QuantConfig:
        fields = {f.name for f in dataclasses.fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in fields}
        if "ignore_layers" in kwargs and kwargs["ignore_layers"] is not None:
            kwargs["ignore_layers"] = tuple(kwargs["ignore_layers"])
        return cls(**kwargs)


@dataclasses.dataclass(slots=True)
class CacheConfig:
    """Paged KV-cache and prefix-cache settings."""

    block_size: int = 16
    num_gpu_blocks: int = 512
    num_cpu_blocks: int = 0
    dtype: str = "bfloat16"
    enable_prefix_caching: bool = True
    swap_threshold: int = 0

    def __post_init__(self) -> None:
        if self.block_size < 1:
            raise ValueError("block_size must be >= 1")
        if self.num_gpu_blocks < 1:
            raise ValueError("num_gpu_blocks must be >= 1")

    @property
    def num_layers_with_cache(self) -> int:
        return 0

    def tokens_per_gpu(self) -> int:
        return self.block_size * self.num_gpu_blocks

    def kv_bytes_per_block(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: str = "float16",
    ) -> int:
        """Bytes consumed by one block: both K and V, all layers, ``block_size`` tokens."""
        itemsize = {"float16": 2, "bfloat16": 2, "float32": 4}.get(dtype, 2)
        per_token = 2 * num_layers * num_kv_heads * head_dim * itemsize
        return max(1, self.block_size * per_token)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CacheConfig:
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in fields})


@dataclasses.dataclass(slots=True)
class SchedulerConfig:
    """Continuous batching and chunked-prefill settings."""

    max_num_batched_tokens: int = 8192
    max_num_seqs: int = 256
    prefill_chunk_size: int = 1024
    max_model_len: int = 4096
    enable_chunked_prefill: bool = True

    def __post_init__(self) -> None:
        if self.max_num_batched_tokens < 1:
            raise ValueError("max_num_batched_tokens must be >= 1")
        if self.max_num_seqs < 1:
            raise ValueError("max_num_seqs must be >= 1")
        if self.prefill_chunk_size < 1:
            raise ValueError("prefill_chunk_size must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SchedulerConfig:
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in fields})


@dataclasses.dataclass(slots=True)
class EngineConfig:
    """Top level engine configuration."""

    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    quant: QuantConfig = dataclasses.field(default_factory=QuantConfig)
    cache: CacheConfig = dataclasses.field(default_factory=CacheConfig)
    scheduler: SchedulerConfig = dataclasses.field(default_factory=SchedulerConfig)
    seed: int = 0
    device: str = "auto"
    enable_prefix_caching: bool = True
    chunked_prefill: bool = True

    def resolve_device(self) -> str:
        """Pick a concrete torch device, honouring ``auto``."""
        if self.device != "auto":
            return self.device
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if getattr(torch.backends, "mps", None) is not None and (
                torch.backends.mps.is_available()
            ):
                return "mps"
        except Exception:  # pragma: no cover - torch always present in practice
            pass
        return "cpu"

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.to_dict(),
            "quant": self.quant.to_dict(),
            "cache": self.cache.to_dict(),
            "scheduler": self.scheduler.to_dict(),
            "seed": self.seed,
            "device": self.device,
            "enable_prefix_caching": self.enable_prefix_caching,
            "chunked_prefill": self.chunked_prefill,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EngineConfig:
        return cls(
            model=ModelConfig.from_dict(data.get("model", {})),
            quant=QuantConfig.from_dict(data.get("quant", {})),
            cache=CacheConfig.from_dict(data.get("cache", {})),
            scheduler=SchedulerConfig.from_dict(data.get("scheduler", {})),
            seed=int(data.get("seed", 0)),
            device=str(data.get("device", "auto")),
            enable_prefix_caching=_as_bool(data.get("enable_prefix_caching", True)),
            chunked_prefill=_as_bool(data.get("chunked_prefill", True)),
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


@dataclasses.dataclass(slots=True)
class ServerConfig:
    """HTTP server settings for the OpenAI-compatible API."""

    host: str = "127.0.0.1"
    port: int = 8000
    model_id: str = "llmopt-tiny"
    api_key: str | None = None
    log_requests: bool = True
    #: Write ``/info`` to this path on shutdown when set.
    snapshot_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ServerConfig:
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in fields})


def num_blocks_for_gpu_memory(
    model: ModelConfig,
    cache: CacheConfig,
    gpu_memory_utilization: float = 0.9,
    device: str = "cuda",
    total_memory_bytes: int | None = None,
) -> int:
    """Size the KV cache from the memory budget on ``device``.

    Mirrors what production servers do: reserve ``gpu_memory_utilization`` of
    total device memory for weights plus cache, subtract the weights, then fill
    the remainder with as many blocks as fit.

    Args:
        model: Architecture, used for the per-token KV footprint.
        cache: Cache settings; ``block_size`` and ``dtype`` are used.
        gpu_memory_utilization: Fraction of device memory to use, in ``(0, 1]``.
        device: Device to measure; ignored when ``total_memory_bytes`` is given.
        total_memory_bytes: Explicit budget, mainly for tests and CPU runs.

    Returns:
        A block count of at least 1.
    """
    if not 0.0 < gpu_memory_utilization <= 1.0:
        raise ValueError("gpu_memory_utilization must be in (0, 1]")
    if total_memory_bytes is None:
        import torch

        if device == "cuda" and torch.cuda.is_available():
            free, _total = torch.cuda.mem_get_info()
            total_memory_bytes = int(free)
        else:
            # CPU has no hard budget; assume a conservative 4 GiB working set.
            total_memory_bytes = 4 * 1024**3
    budget = int(total_memory_bytes * gpu_memory_utilization)
    per_block = cache.kv_bytes_per_block(
        model.num_hidden_layers,
        model.num_key_value_heads,
        model.head_dim,
        cache.dtype,
    )
    return max(1, budget // per_block)
