"""Synthetic workload generation for throughput/latency benchmarks."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

__all__ = ["BenchRequest", "WorkloadSpec", "generate_workload", "summarize_workload"]


@dataclass(slots=True)
class BenchRequest:
    """One request in a benchmark workload."""

    prompt: str
    max_tokens: int
    request_id: str = ""
    priority: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class WorkloadSpec:
    """Distribution describing a workload to generate.

    Args:
        num_requests: How many requests to emit.
        prompt_tokens: Prompt length, or ``(min, max)`` for a range.
        output_tokens: Output budget, or ``(min, max)`` for a range.
        inter_arrival_s: Delay between submissions; ``0`` means send everything
            at once, which is how a saturated throughput run is configured.
        shared_prefix: Fraction of every prompt that is a common prefix, used
            to exercise prefix caching.
        seed: RNG seed for reproducibility.
    """

    num_requests: int = 32
    prompt_tokens: int | tuple[int, int] = 64
    output_tokens: int | tuple[int, int] = 64
    inter_arrival_s: float = 0.0
    shared_prefix: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.num_requests < 1:
            raise ValueError("num_requests must be >= 1")
        if not 0.0 <= self.shared_prefix < 1.0:
            raise ValueError("shared_prefix must be in [0, 1)")


def _sample(rng: random.Random, spec: int | tuple[int, int]) -> int:
    if isinstance(spec, tuple):
        low, high = spec
        if high < low:
            raise ValueError(f"invalid range {spec}: max < min")
        return rng.randint(low, high)
    return spec


_ALPHABET = "the quick brown fox jumps over lazy dogs while parsing tokens 0123456789"


def generate_workload(spec: WorkloadSpec) -> list[BenchRequest]:
    """Build a reproducible list of :class:`BenchRequest`.

    Prompts are random token-ish strings, so the model's output is effectively
    unpredictable and the measurement reflects real prefill/decode work.
    """
    rng = random.Random(spec.seed)
    words = _ALPHABET.split()
    prefix_len = int(spec.prompt_tokens * spec.shared_prefix) if isinstance(
        spec.prompt_tokens, int
    ) else 0
    shared = " ".join(rng.choice(words) for _ in range(prefix_len)) if prefix_len else ""

    requests: list[BenchRequest] = []
    for index in range(spec.num_requests):
        total = _sample(rng, spec.prompt_tokens)
        tail_len = max(1, total - prefix_len)
        tail = " ".join(rng.choice(words) for _ in range(tail_len))
        prompt = f"{shared} {tail}".strip() if shared else tail
        requests.append(
            BenchRequest(
                prompt=prompt,
                max_tokens=_sample(rng, spec.output_tokens),
                request_id=f"bench-{index}",
            )
        )
    return requests


def summarize_workload(requests: list[BenchRequest]) -> dict[str, Any]:
    """Summarise a workload without running it."""
    return {
        "num_requests": len(requests),
        "prompt_chars_mean": sum(len(r.prompt) for r in requests) / max(1, len(requests)),
        "output_tokens_total": sum(r.max_tokens for r in requests),
    }
