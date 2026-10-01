"""Continuous-batching scheduler.

The loop is the classic vLLM one, kept explicit here so it can be reasoned
about (and tested) step by step:

1. Admit waiting requests while the token budget and slot count allow.
2. Spend the remaining budget on prefilling their prompt tokens, in chunks, so
   one long prompt cannot starve the batch.
3. Decode one token for every sequence whose prefill has completed.
4. Free the blocks of finished sequences and hand them to waiting requests.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from llmopt.cache.block_pool import BlockPool
from llmopt.cache.prefix_cache import PrefixCache
from llmopt.config import SchedulerConfig
from llmopt.engine.request import FinishReason, Request, RequestStatus
from llmopt.utils.logging import get_logger
from llmopt.utils.metrics import MetricRegistry

__all__ = ["Scheduler", "ScheduledStep", "SchedulerStats"]

logger = get_logger("engine.scheduler")


@dataclass(slots=True)
class ScheduledStep:
    """One iteration's work: what to prefill, what to decode."""

    running: list[Request] = field(default_factory=list)
    prefill_chunks: list[tuple[Request, int, int]] = field(default_factory=list)
    decode: list[Request] = field(default_factory=list)
    preempted: list[Request] = field(default_factory=list)
    finished: list[Request] = field(default_factory=list)
    num_batched_tokens: int = 0

    @property
    def is_empty(self) -> bool:
        return not (self.prefill_chunks or self.decode or self.preempted or self.finished)

    def describe(self) -> str:
        return (
            f"prefill={[(r.request_id, s, e) for r, s, e in self.prefill_chunks]} "
            f"decode={[r.request_id for r in self.decode]} "
            f"finished={[r.request_id for r in self.finished]}"
        )


@dataclass(slots=True)
class SchedulerStats:
    """Counters describing scheduler behaviour over the engine's lifetime."""

    num_scheduled_steps: int = 0
    num_prefill_steps: int = 0
    num_decode_steps: int = 0
    num_admitted: int = 0
    num_preempted: int = 0
    num_finished: int = 0
    num_prefix_hits: int = 0
    total_batched_tokens: int = 0
    total_prefill_tokens: int = 0

    @property
    def avg_batched_tokens(self) -> float:
        return self.total_batched_tokens / self.num_scheduled_steps if self.num_scheduled_steps else 0.0

    def to_dict(self) -> dict[str, float | int]:
        return {
            "num_scheduled_steps": self.num_scheduled_steps,
            "num_prefill_steps": self.num_prefill_steps,
            "num_decode_steps": self.num_decode_steps,
            "num_admitted": self.num_admitted,
            "num_preempted": self.num_preempted,
            "num_finished": self.num_finished,
            "num_prefix_hits": self.num_prefix_hits,
            "total_batched_tokens": self.total_batched_tokens,
            "total_prefill_tokens": self.total_prefill_tokens,
            "avg_batched_tokens": round(self.avg_batched_tokens, 2),
        }


class Scheduler:
    """Drives admission, chunked prefill, decode, and preemption.

    Args:
        config: Scheduler settings.
        block_pool: Pool that owns KV blocks.
        prefix_cache: Optional content-addressed prefix cache.
        metrics: Optional metrics registry.
    """

    def __init__(
        self,
        config: SchedulerConfig | None = None,
        block_pool: BlockPool | None = None,
        prefix_cache: PrefixCache | None = None,
        metrics: MetricRegistry | None = None,
    ) -> None:
        self.config = config or SchedulerConfig()
        self.block_pool = block_pool
        self.prefix_cache = prefix_cache
        self.metrics = metrics
        self.waiting: list[Request] = []
        self.running: list[Request] = []
        self.finished: list[Request] = []
        self.stats = SchedulerStats()
        self._request_index: dict[str, Request] = {}
        # How many of each request's chain blocks are already in the prefix cache.
        self._published: dict[str, int] = {}
        # Per-step scratch: requests holding blocks for this step's plan, and
        # requests evicted out of ``running`` while the step was being built.
        self._planned: set[str] = set()
        self._victims: list[Request] = []

    def add_request(self, request: Request) -> None:
        """Queue a request for admission."""
        request.status = RequestStatus.WAITING
        self.waiting.append(request)
        self._request_index[request.request_id] = request

    def get_request(self, request_id: str) -> Request | None:
        return self._request_index.get(request_id)

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)

    def abort(self, request_id: str) -> bool:
        """Cancel a queued or running request and release its blocks."""
        request = self._request_index.get(request_id)
        if request is None or request.is_done():
            return False
        self.waiting = [r for r in self.waiting if r.request_id != request_id]
        self.running = [r for r in self.running if r.request_id != request_id]
        request.status = RequestStatus.ABORTED
        request.finish_reason = FinishReason.ABORTED
        self._release(request)
        self.finished.append(request)
        self._counter("num_finished")
        return True

    def step(self) -> ScheduledStep:
        """Plan one engine iteration."""
        result = ScheduledStep()
        budget = self.config.max_num_batched_tokens
        # Every prefill row in a step is padded/batched to one width, because the
        # attention kernel takes a single ``q_len`` for the whole batch. The first
        # chunk scheduled fixes that width; others wait for a later step.
        prefill_width: int | None = None
        # Requests that already hold blocks for work in *this* step's plan. They
        # must not be preempted, or the runner would write into freed blocks.
        self._planned: set[str] = set()
        self._victims: list[Request] = []

        self._free_finished()

        # 1. Decode the running sequences whose prompt is fully computed. They
        #    already hold blocks, so finishing them is what returns capacity to
        #    the pool, and the runner cannot execute decode and prefill in the
        #    same step. Decode therefore goes first and, when it has work, the
        #    step is decode-only: admitting new sequences before the running ones
        #    drain would pin every block and leave nothing able to finish.
        for request in self.running:
            if request.num_computed_tokens < request.prompt_len or request.is_done():
                continue
            if not self._ensure_capacity(request, 1):
                continue
            result.decode.append(request)
            self._planned.add(request.request_id)
            result.num_batched_tokens += 1
        if result.decode:
            result.running = list(self.running)
            result.finished = list(self.finished)
            self.stats.num_scheduled_steps += 1
            self.stats.total_batched_tokens += result.num_batched_tokens
            self._counter("num_decode_steps")
            return result

        # 2. Chunked prefill for sequences already in the running set.
        for request in self.running:
            remaining = request.remaining_prefill
            if remaining == 0:
                continue
            if budget <= 0:
                break
            chunk = min(remaining, budget, self.config.prefill_chunk_size)
            if prefill_width is None:
                prefill_width = chunk
            elif chunk != prefill_width:
                continue
            start = request.num_computed_tokens
            if not self._ensure_capacity(request, chunk):
                result.preempted.append(request)
                continue
            result.prefill_chunks.append((request, start, start + chunk))
            self._planned.add(request.request_id)
            budget -= chunk
            result.num_batched_tokens += chunk
            self.stats.total_prefill_tokens += chunk
            if start == 0:
                self._counter("num_prefill_steps")

        # 3. Admit new requests with whatever budget is left.
        still_waiting: list[Request] = []
        for request in self.waiting:
            if budget <= 0 or len(self.running) >= self.config.max_num_seqs:
                still_waiting.append(request)
                continue
            if not request.check_length():
                self.finished.append(request)
                self._counter("num_finished")
                continue
            # Adopt cached prefix blocks before sizing the chunk, otherwise the
            # plan would re-prefill tokens the cache already holds.
            self.apply_prefix_cache(request)
            remaining = request.remaining_prefill
            if remaining == 0:
                # The whole prompt came out of the prefix cache, so there is no
                # chunk to prefill. Admit it straight into the decode phase,
                # otherwise it would be re-queued forever and never finish.
                self.running.append(request)
                request.status = RequestStatus.RUNNING
                self._planned.add(request.request_id)
                self._counter("num_admitted")
                continue
            chunk = min(remaining, budget, self.config.prefill_chunk_size)
            if chunk <= 0:
                still_waiting.append(request)
                continue
            if prefill_width is None:
                prefill_width = chunk
            elif chunk != prefill_width:
                still_waiting.append(request)
                continue
            if not self._ensure_capacity(request, chunk):
                still_waiting.append(request)
                continue
            self.running.append(request)
            request.status = RequestStatus.RUNNING
            result.prefill_chunks.append(
                (request, request.num_computed_tokens, request.num_computed_tokens + chunk)
            )
            self._planned.add(request.request_id)
            budget -= chunk
            result.num_batched_tokens += chunk
            self.stats.total_prefill_tokens += chunk
            self._counter("num_admitted")
            self._counter("num_prefill_steps")
        self.waiting = still_waiting

        result.running = list(self.running)
        result.finished = list(self.finished)
        # Requests preempted during admission or decode were moved out of
        # ``running`` mid-iteration. Re-queue them at the front so they are
        # neither lost nor invisible to ``has_unfinished``.
        if self._victims:
            self.waiting = self._victims + self.waiting
            self._victims = []
        if not result.is_empty:
            self.stats.num_scheduled_steps += 1
            self.stats.total_batched_tokens += result.num_batched_tokens
            if result.decode:
                self._counter("num_decode_steps")
        return result

    def _ensure_capacity(self, request: Request, num_tokens: int) -> bool:
        """Grow the request's block chain, preempting the newest if needed."""
        if self.block_pool is None:
            return True
        if self.block_pool.can_grow(request.request_id, num_tokens):
            self.block_pool.append(request.request_id, num_tokens)
            return True
        # Cached prefix blocks are the only memory that can be reclaimed without
        # losing a running sequence, so try trading them for room first.
        if self.prefix_cache is not None and self.block_pool is not None:
            need = max(1, self.block_pool.blocks_needed(request.request_id, num_tokens))
            if self.prefix_cache.evict_until_free(need) and self.block_pool.can_grow(
                request.request_id, num_tokens
            ):
                self.block_pool.append(request.request_id, num_tokens)
                return True
        return self._preempt(request, num_tokens)

    def _preempt(self, request: Request, num_tokens: int) -> bool:
        """Evict a running request to free blocks for ``request``.

        The victim is recomputed from scratch on the next admission, which is
        simpler and more predictable than a swap-in/swap-out policy and keeps
        correctness independent of the eviction order.

        Requests that already have work planned for the current step are never
        chosen: the runner is about to write into their blocks, and releasing
        them mid-plan would both corrupt the write and double-count the tokens
        once they are re-admitted. Such a request simply waits for the next step.

        Args:
            request: The request that needs the blocks.
            num_tokens: Tokens the request still needs room for.

        Returns:
            ``True`` when capacity was found, ``False`` when the request must
            keep waiting.
        """
        planned = self._planned
        candidates = [
            r
            for r in self.running
            if r.request_id != request.request_id and r.request_id not in planned
        ]
        if not candidates:
            return False
        # Evict the most recently admitted victim: it loses the least work.
        for victim in reversed(candidates):
            self.running.remove(victim)
            self._release(victim, reset_state=True)
            victim.status = RequestStatus.WAITING
            victim.num_computed_tokens = 0
            victim.num_cached_tokens = 0
            self._victims.append(victim)
            self._counter("num_preempted")
            if self.block_pool.can_grow(request.request_id, num_tokens):
                self.block_pool.append(request.request_id, num_tokens)
                return True
        return False

    def _release(self, request: Request, reset_state: bool = False) -> None:
        """Return the request's blocks to the pool.

        ``reset_state`` additionally rewinds the computed-token counter, which
        preemption needs so the sequence is prefilled again from scratch.
        """
        if self.block_pool is not None:
            self.block_pool.release(request.request_id)
        self._published.pop(request.request_id, None)
        if reset_state:
            request.num_computed_tokens = 0
            request.num_cached_tokens = 0

    def _free_finished(self) -> None:
        survivors: list[Request] = []
        for request in self.running:
            if request.is_done():
                self._release(request)
                self.finished.append(request)
                self._counter("num_finished")
            else:
                survivors.append(request)
        self.running = survivors

    def apply_prefix_cache(self, request: Request) -> int:
        """Adopt cached blocks for ``request`` and return tokens saved.

        The cached prefix is marked as computed so the engine skips recomputing
        it, and the tail is re-encoded only from the first divergent token.
        """
        if self.prefix_cache is None or request.num_computed_tokens:
            return 0
        cached = self.prefix_cache.find(request.prompt_token_ids)
        if cached is None or not cached.block_ids:
            return 0
        if self.block_pool is not None:
            self.block_pool.share_blocks(
                request.request_id, cached.block_ids, cached.num_tokens
            )
            # Shared blocks were already published by the sequence that cached
            # them, so do not re-publish them for this request.
            self._published[request.request_id] = len(cached.block_ids)
        request.num_cached_tokens = cached.num_tokens
        request.num_computed_tokens = cached.num_tokens
        self._counter("num_prefix_hits")
        return cached.num_tokens

    def cache_prefix(self, request: Request) -> int:
        """Publish newly filled full blocks of ``request`` into the prefix cache.

        Only blocks that became full since the last call are published, so
        refcounts are incremented exactly once per block per request.
        """
        if self.prefix_cache is None or self.block_pool is None:
            return 0
        chain = self.block_pool.chains.get(request.request_id, [])
        # Only blocks whose tokens have all been written are immutable enough
        # to share, so publish up to the last fully written block.
        num_full = self.block_pool.tokens_used(request.request_id) // self.block_pool.block_size
        published = self._published.get(request.request_id, 0)
        if num_full <= published:
            return 0
        self.prefix_cache.insert(
            request.prompt_token_ids[: num_full * self.block_pool.block_size],
            chain[:num_full],
        )
        self.block_pool.mark_shared(chain[published:num_full])
        self._published[request.request_id] = num_full
        return num_full - published

    def _counter(self, name: str) -> None:
        """Bump a :class:`SchedulerStats` field and mirror it to ``metrics``."""
        setattr(self.stats, name, getattr(self.stats, name) + 1)
        counter = self.metrics.counter(f"llmopt_{name}", f"Total {name}") if self.metrics else None
        if counter is not None:
            counter.inc()

    def has_capacity(self, num_tokens: int) -> bool:
        """Whether ``num_tokens`` more KV tokens would fit right now."""
        if self.block_pool is None:
            return True
        return self.block_pool.allocator.can_allocate(
            -(-num_tokens // self.block_pool.block_size)
        )

    def snapshot(self) -> dict[str, int]:
        return {
            "waiting": len(self.waiting),
            "running": len(self.running),
            "finished": len(self.finished),
        }

    def __repr__(self) -> str:
        return f"Scheduler({self.snapshot()}, prefix_cache={self.prefix_cache is not None})"
