"""Request lifecycle types and sampling parameters."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__all__ = [
    "FinishReason",
    "GeneratedToken",
    "Request",
    "RequestOutput",
    "RequestStatus",
    "SamplingParams",
]


class RequestStatus(str, Enum):
    """Where a request sits in the pipeline."""

    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"
    ABORTED = "aborted"


class FinishReason(str, Enum):
    """Why generation stopped."""

    STOP = "stop"
    LENGTH = "length"
    ABORTED = "aborted"


@dataclass(slots=True)
class SamplingParams:
    """Decoding controls, mirroring the OpenAI chat-completions vocabulary."""

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    max_tokens: int = 128
    min_tokens: int = 0
    seed: int | None = None
    stop: tuple[str, ...] = ()
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    logit_bias: dict[int, float] = field(default_factory=dict)
    ignore_eos: bool = False
    n: int = 1

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")
        if self.n < 1:
            raise ValueError("n must be >= 1")
        self.stop = tuple(self.stop)

    @property
    def greedy(self) -> bool:
        return self.temperature == 0.0

    def with_overrides(self, **kwargs: Any) -> SamplingParams:
        """Return a copy with the non-``None`` fields replaced."""
        data = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
            "min_tokens": self.min_tokens,
            "seed": self.seed,
            "stop": self.stop,
            "repetition_penalty": self.repetition_penalty,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "logit_bias": dict(self.logit_bias),
            "ignore_eos": self.ignore_eos,
            "n": self.n,
        }
        data.update({k: v for k, v in kwargs.items() if v is not None and k in data})
        return SamplingParams(**data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            "stop": list(self.stop),
            "repetition_penalty": self.repetition_penalty,
            "n": self.n,
        }

    @classmethod
    def from_openai(cls, payload: dict[str, Any]) -> SamplingParams:
        """Build params from an OpenAI-compatible request body."""
        max_tokens = payload.get("max_tokens", payload.get("max_completion_tokens", 128))
        return cls(
            temperature=float(payload.get("temperature", 1.0)),
            top_p=float(payload.get("top_p", 1.0)),
            top_k=int(payload.get("top_k", 0)),
            max_tokens=int(max_tokens),
            stop=_normalize_stop(payload.get("stop")),
            seed=payload.get("seed"),
            ignore_eos=bool(payload.get("ignore_eos", False)),
            n=int(payload.get("n", 1)),
        )


def _normalize_stop(stop: Any) -> tuple[str, ...]:
    if stop is None:
        return ()
    if isinstance(stop, str):
        return (stop,)
    return tuple(str(item) for item in stop)


@dataclass(slots=True)
class Request:
    """A single generation request and its mutable engine state."""

    request_id: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams = field(default_factory=SamplingParams)
    arrival_time: float = field(default_factory=time.perf_counter)
    max_model_len: int = 4096
    lora_request: str | None = None

    status: RequestStatus = RequestStatus.WAITING
    prompt_len: int = field(init=False)
    num_computed_tokens: int = field(default=False, init=False)
    num_cached_tokens: int = field(default=0, init=False)
    output_token_ids: list[int] = field(default_factory=list)
    finish_reason: FinishReason | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    error: str | None = None

    def __post_init__(self) -> None:
        self.prompt_len = len(self.prompt_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self.output_token_ids)

    @property
    def total_tokens(self) -> int:
        return self.prompt_len + self.num_output_tokens

    @property
    def remaining_prefill(self) -> int:
        return max(0, self.prompt_len - self.num_computed_tokens)

    def is_done(self) -> bool:
        return self.status in (RequestStatus.FINISHED, RequestStatus.ABORTED)

    def can_decode(self) -> bool:
        """True once every prompt token has been processed."""
        return self.num_computed_tokens >= self.prompt_len and not self.is_done()

    def all_tokens(self) -> list[int]:
        return [*self.prompt_token_ids, *self.output_token_ids]

    def check_length(self) -> bool:
        """Whether prompt + budget fits the model window."""
        limit = min(self.max_model_len, self.sampling_params.max_tokens + self.prompt_len)
        if self.prompt_len >= limit:
            self.status = RequestStatus.FINISHED
            self.finish_reason = FinishReason.LENGTH
            self.error = (
                f"prompt of {self.prompt_len} tokens exceeds the model window "
                f"of {self.max_model_len}"
            )
            return False
        return True

    def __repr__(self) -> str:
        return (
            f"Request({self.request_id}, status={self.status.value}, "
            f"prompt={self.prompt_len}, computed={self.num_computed_tokens}, "
            f"out={self.num_output_tokens})"
        )


@dataclass(slots=True)
class GeneratedToken:
    """One sampled token with its log-probability."""

    token_id: int
    text: str = ""
    logprob: float = 0.0


@dataclass(slots=True)
class RequestOutput:
    """Result snapshot for a request at a point in time."""

    request_id: str
    prompt_token_ids: list[int]
    token_ids: list[int]
    text: str = ""
    status: RequestStatus = RequestStatus.RUNNING
    finish_reason: FinishReason | None = None
    num_cached_tokens: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    error: str | None = None

    @property
    def num_generated_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    def to_dict(self, include_tokens: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.request_id,
            "status": self.status.value,
            "finish_reason": self.finish_reason.value if self.finish_reason else None,
            "num_prompt_tokens": self.num_prompt_tokens,
            "num_generated_tokens": self.num_generated_tokens,
            "num_cached_tokens": self.num_cached_tokens,
            "metrics": self.metrics,
        }
        if include_tokens:
            payload["token_ids"] = self.token_ids
            payload["text"] = self.text
        if self.error:
            payload["error"] = self.error
        return payload
