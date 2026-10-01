"""Prometheus-style metrics without third-party dependencies."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

__all__ = ["Counter", "Histogram", "MetricRegistry", "Timer"]


@dataclass
class Counter:
    """A monotonically increasing counter."""

    name: str
    help: str = ""
    labels: tuple[str, ...] = ()
    _values: dict[tuple[str, ...], float] = field(default_factory=dict, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        key = self._key(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount

    def value(self, **labels: str) -> float:
        with self._lock:
            return self._values.get(self._key(labels), 0.0)

    def _key(self, labels: dict[str, str]) -> tuple[str, ...]:
        missing = set(self.labels) - set(labels)
        if missing:
            raise ValueError(f"counter {self.name} is missing labels: {sorted(missing)}")
        return tuple(str(labels[name]) for name in self.labels)

    def render(self) -> Iterator[str]:
        with self._lock:
            items = sorted(self._values.items())
        if not items:
            return
        if self.help:
            yield f"# HELP {self.name} {self.help}"
        yield f"# TYPE {self.name} counter"
        for key, value in items:
            yield f"{self.name}{_fmt_labels(dict(zip(self.labels, key, strict=False)))} {value}"


@dataclass
class Histogram:
    """Fixed-bucket histogram with quantile estimation."""

    name: str
    help: str = ""
    buckets: tuple[float, ...] = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
    _counts: list[int] = field(default_factory=list, repr=False)
    _sum: float = 0.0
    _total: int = 0
    _values: list[float] = field(default_factory=list, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        if not self._counts:
            self._counts = [0] * (len(self.buckets) + 1)

    def observe(self, value: float) -> None:
        with self._lock:
            for i, bound in enumerate(self.buckets):
                if value <= bound:
                    self._counts[i] += 1
                    break
            else:
                self._counts[-1] += 1
            self._sum += value
            self._total += 1
            if len(self._values) < 10_000:
                self._values.append(value)

    def count(self) -> int:
        return self._total

    def mean(self) -> float:
        return self._sum / self._total if self._total else 0.0

    def quantile(self, q: float) -> float:
        """Estimate a quantile from the retained sample buffer."""
        with self._lock:
            values = sorted(self._values)
        if not values:
            return 0.0
        if len(values) == self._total:
            idx = min(len(values) - 1, max(0, math.ceil(q * len(values)) - 1))
            return values[idx]
        return self.mean()

    def render(self) -> Iterator[str]:
        with self._lock:
            counts = list(self._counts)
            total, total_sum = self._total, self._sum
        if self.help:
            yield f"# HELP {self.name} {self.help}"
        yield f"# TYPE {self.name} histogram"
        cumulative = 0
        for bound, count in zip([*self.buckets, "+Inf"], counts, strict=True):
            cumulative += count
            yield f'{self.name}_bucket{{le="{bound}"}} {cumulative}'
        yield f"{self.name}_sum {total_sum}"
        yield f"{self.name}_count {total}"


def _fmt_labels(labels: dict[str, str]) -> str:
    if not labels:
        return ""
    inner = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
    return "{" + inner + "}"


class MetricRegistry:
    """A tiny registry so the engine can be instrumented without global state."""

    def __init__(self) -> None:
        self._counters: dict[str, Counter] = {}
        self._histograms: dict[str, Histogram] = {}

    def counter(self, name: str, help: str = "", labels: tuple[str, ...] = ()) -> Counter:
        if name not in self._counters:
            self._counters[name] = Counter(name, help, labels)
        return self._counters[name]

    def histogram(
        self,
        name: str,
        help: str = "",
        buckets: tuple[float, ...] = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    ) -> Histogram:
        if name not in self._histograms:
            self._histograms[name] = Histogram(name, help, buckets)
        return self._histograms[name]

    def render(self) -> str:
        lines: list[str] = []
        for metric in sorted(self._counters.values(), key=lambda m: m.name):
            lines.extend(metric.render())
        for metric in sorted(self._histograms.values(), key=lambda m: m.name):
            lines.extend(metric.render())
        return "\n".join(lines) + ("\n" if lines else "")

    def snapshot(self) -> dict[str, float]:
        """Flatten the registry to plain floats for JSON reporting."""
        out: dict[str, float] = {}
        for counter in self._counters.values():
            for key, value in list(counter._values.items()):
                label = _fmt_labels(dict(zip(counter.labels, key, strict=False))) or ""
                out[f"{counter.name}{label}"] = value
        for hist in self._histograms.values():
            out[f"{hist.name}_count"] = float(hist.count())
            out[f"{hist.name}_sum"] = float(hist.mean() * hist.count())
        return out


@contextmanager
def Timer(metrics: MetricRegistry, name: str) -> Iterator[None]:  # noqa: N802
    """Time a block and record the elapsed seconds into a registry histogram."""
    hist = metrics.histogram(name, help=f"Latency of {name}")
    start = time.perf_counter()
    try:
        yield
    finally:
        hist.observe(time.perf_counter() - start)
