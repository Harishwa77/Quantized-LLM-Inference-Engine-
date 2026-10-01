"""Benchmark driver that measures latency and throughput of an engine.

Submits a :mod:`~llmopt.benchmark.workload` workload through
:class:`~llmopt.engine.llm_engine.LLMEngine` and reports TTFT, inter-token
latency, output throughput, and cache behaviour.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Any

from llmopt.benchmark.workload import BenchRequest, WorkloadSpec, generate_workload
from llmopt.engine.llm_engine import LLMEngine
from llmopt.engine.request import SamplingParams
from llmopt.utils.logging import get_logger

__all__ = ["BenchmarkResult", "BenchmarkRunner", "run_benchmark", "compare_configurations"]

logger = get_logger("benchmark")


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile; ``0.0`` for an empty sample."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, min(len(ordered), int(round(pct / 100.0 * len(ordered)) + 0.5)))
    return ordered[rank - 1]


@dataclass(slots=True)
class RequestTiming:
    """Per-request measurements."""

    request_id: str
    prompt_tokens: int = 0
    output_tokens: int = 0
    ttft_ms: float = 0.0
    total_ms: float = 0.0
    enqueue_ms: float = 0.0

    @property
    def itl_ms(self) -> float:
        """Mean inter-token latency, i.e. TPOT excluding the first token."""
        if self.output_tokens <= 1:
            return 0.0
        return (self.total_ms - self.ttft_ms) / (self.output_tokens - 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "ttft_ms": round(self.ttft_ms, 3),
            "itl_ms": round(self.itl_ms, 3),
            "total_ms": round(self.total_ms, 3),
            "queue_ms": round(self.enqueue_ms, 3),
        }


@dataclass(slots=True)
class BenchmarkResult:
    """Aggregate benchmark metrics plus per-request detail."""

    name: str
    wall_ms: float
    timings: list[RequestTiming] = field(default_factory=list)
    engine_stats: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------- accessors

    @property
    def num_requests(self) -> int:
        return len(self.timings)

    @property
    def prompt_tokens(self) -> int:
        return sum(t.prompt_tokens for t in self.timings)

    @property
    def output_tokens(self) -> int:
        return sum(t.output_tokens for t in self.timings)

    @property
    def seconds(self) -> float:
        return self.wall_ms / 1000.0 or 1e-9

    def to_dict(self, include_requests: bool = False) -> dict[str, Any]:
        """Return metrics, optionally including per-request rows."""
        ttft = [t.ttft_ms for t in self.timings]
        itl = [t.itl_ms for t in self.timings if t.itl_ms > 0]
        totals = [t.total_ms for t in self.timings]
        metrics: dict[str, Any] = {
            "name": self.name,
            "num_requests": self.num_requests,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "wall_ms": round(self.wall_ms, 3),
            "output_throughput_tok_s": round(self.output_tokens / self.seconds, 3),
            "request_throughput_req_s": round(self.num_requests / self.seconds, 3),
            "total_throughput_tok_s": round(
                (self.prompt_tokens + self.output_tokens) / self.seconds, 3
            ),
            "ttft_ms_mean": round(statistics.fmean(ttft), 3) if ttft else 0.0,
            "ttft_ms_p50": round(_percentile(ttft, 50), 3),
            "ttft_ms_p95": round(_percentile(ttft, 95), 3),
            "ttft_ms_p99": round(_percentile(ttft, 99), 3),
            "itl_ms_mean": round(statistics.fmean(itl), 3) if itl else 0.0,
            "itl_ms_p95": round(_percentile(itl, 95), 3),
            "total_ms_mean": round(statistics.fmean(totals), 3) if totals else 0.0,
            "engine": self.engine_stats,
        }
        if include_requests:
            metrics["requests"] = [t.to_dict() for t in self.timings]
        return metrics

    def format_table(self) -> str:
        """Render the summary as a short aligned text table."""
        rows = [
            ("requests", f"{self.num_requests}"),
            ("prompt tokens", f"{self.prompt_tokens}"),
            ("output tokens", f"{self.output_tokens}"),
            ("wall time", f"{self.wall_ms / 1000:.2f} s"),
            (
                "output throughput",
                f"{self.output_tokens / self.seconds:.1f} tok/s",
            ),
            (
                "request throughput",
                f"{self.num_requests / self.seconds:.2f} req/s",
            ),
            ("TTFT mean", f"{self.to_dict()['ttft_ms_mean']:.1f} ms"),
            ("TTFT p95", f"{self.to_dict()['ttft_ms_p95']:.1f} ms"),
            ("ITL mean", f"{self.to_dict()['itl_ms_mean']:.2f} ms"),
            ("ITL p95", f"{self.to_dict()['itl_ms_p95']:.2f} ms"),
        ]
        width = max(len(label) for label, _ in rows)
        return "\n".join(f"{label.ljust(width)}  {value}" for label, value in rows)

    def format_comparison(self, other: BenchmarkResult) -> str:
        """Compare two results and show the relative change per metric."""
        mine, theirs = self.to_dict(), other.to_dict()
        keys = [
            "output_throughput_tok_s",
            "request_throughput_req_s",
            "ttft_ms_mean",
            "ttft_ms_p95",
            "itl_ms_mean",
        ]
        width = max(len(key) for key in keys)
        lines = [f"{'metric'.ljust(width)}  {self.name}  {other.name}  delta"]
        for key in keys:
            a, b = float(mine[key]), float(theirs[key])
            if a == 0:
                pct = "n/a"
            else:
                pct = f"{(b - a) / a * 100:+.1f}%"
            lines.append(f"{key.ljust(width)}  {a:>10.2f}  {b:>10.2f}  {pct:>8}")
        return "\n".join(lines)


class BenchmarkRunner:
    """Drives a workload against an engine and records timings.

    Args:
        engine: Engine under test.
        spec: Workload to generate, or pass explicit ``requests``.
        temperature: Sampling temperature; ``0`` is greedy.
        warmup: Requests to run first and discard, to avoid measuring lazy init.
    """

    def __init__(
        self,
        engine: LLMEngine,
        spec: WorkloadSpec | None = None,
        temperature: float = 0.0,
        warmup: int = 1,
    ) -> None:
        self.engine = engine
        self.spec = spec or WorkloadSpec()
        self.temperature = temperature
        self.warmup = max(0, warmup)
        self._last_seen: dict[str, int] = {}

    def _params(self, request: BenchRequest) -> SamplingParams:
        return SamplingParams(
            max_tokens=request.max_tokens, temperature=self.temperature, ignore_eos=True
        )

    def _track(self, request_id: str, out_tokens: int) -> None:
        """Record a first token as soon as one appears."""
        self._last_seen.setdefault(request_id, out_tokens)

    def run(
        self,
        requests: list[BenchRequest] | None = None,
        name: str = "run",
        max_steps: int | None = None,
    ) -> BenchmarkResult:
        """Submit every request, step until drained, and return metrics.

        Arrival pacing from :attr:`WorkloadSpec.inter_arrival_s` is honoured so
        the same runner measures both a saturated batch and a rate-limited
        server. TTFT is measured from each request's own submission time.
        """
        pending = requests if requests is not None else generate_workload(self.spec)
        if self.warmup:
            self._warmup(pending[: self.warmup])
            pending = pending[self.warmup :]
        if not pending:
            return BenchmarkResult(name=name, wall_ms=0.0, engine_stats=self.engine.stats())

        timings: list[RequestTiming] = []
        first_token_at: dict[str, float] = {}
        enqueued_at: dict[str, float] = {}
        arrivals: list[tuple[float, BenchRequest]] = []
        clock = time.perf_counter()
        for index, request in enumerate(pending):
            arrivals.append((clock + index * self.spec.inter_arrival_s, request))
        arrivals.sort(key=lambda item: item[0])

        submitted: set[str] = set()
        output_ids: dict[str, list[int]] = {}
        step = 0
        start = None
        while len(timings) < len(pending):
            now = time.perf_counter()
            if start is None:
                start = now
            for due, request in arrivals:
                if due > now or request.request_id in submitted:
                    continue
                submitted.add(request.request_id)
                enqueued_at[request.request_id] = time.perf_counter()
                params = self._params(request)
                future = self.engine.generate(request.prompt, params, request.request_id)
                output_ids[request.request_id] = []
                future.add_done_callback(
                    lambda fut, rid=request.request_id: output_ids.__setitem__(rid, fut.result().token_ids)
                )
            if len(submitted) == len(pending) and not self.engine.scheduler.has_unfinished():
                break
            updates = self.engine.step()
            step += 1
            for update in updates:
                if update.request_id not in first_token_at:
                    first_token_at[update.request_id] = time.perf_counter()
            if max_steps is not None and step > max_steps:
                raise TimeoutError(f"benchmark exceeded {max_steps} steps")
        wall_ms = (time.perf_counter() - start) * 1000.0

        for request in pending:
            end = time.perf_counter()
            enqueue = enqueued_at.get(request.request_id, start)
            first = first_token_at.get(request.request_id, end)
            timings.append(
                RequestTiming(
                    request_id=request.request_id,
                    prompt_tokens=len(self.engine.tokenizer.encode(request.prompt)),
                    output_tokens=len(output_ids.get(request.request_id, [])),
                    ttft_ms=(first - enqueue) * 1000.0,
                    total_ms=(end - enqueue) * 1000.0,
                )
            )
        return BenchmarkResult(
            name=name, wall_ms=wall_ms, timings=timings, engine_stats=self.engine.stats()
        )

    def _warmup(self, requests: list[BenchRequest]) -> None:
        for request in requests:
            self.engine.generate_sync(request.prompt, self._params(request))


def run_benchmark(
    engine: LLMEngine,
    spec: WorkloadSpec | None = None,
    name: str = "baseline",
    temperature: float = 0.0,
    warmup: int = 1,
) -> BenchmarkResult:
    """Convenience wrapper around :class:`BenchmarkRunner`."""
    return BenchmarkRunner(engine, spec, temperature, warmup).run(name=name)


def compare_configurations(
    engine_factory: Any,
    variants: dict[str, dict[str, Any]],
    spec: WorkloadSpec | None = None,
) -> list[BenchmarkResult]:
    """Run the same workload across engine variants.

    Args:
        engine_factory: Callable taking ``**kwargs`` and returning an engine.
        variants: Mapping of variant name to kwargs for the factory.
        spec: Workload shared by every variant.

    Returns:
        One :class:`BenchmarkResult` per variant, in insertion order.
    """
    results: list[BenchmarkResult] = []
    for name, kwargs in variants.items():
        engine = engine_factory(**kwargs)
        results.append(run_benchmark(engine, spec, name=name))
    return results
