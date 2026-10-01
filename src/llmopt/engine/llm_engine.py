"""The engine: ties scheduler, model runner, cache, and sampler together."""

from __future__ import annotations

import itertools
import threading
import time
from collections.abc import Iterable, Iterator
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any

import torch

from llmopt.cache.block_pool import BlockPool
from llmopt.cache.paged_cache import BlockTable, PagedKVCache
from llmopt.cache.prefix_cache import PrefixCache
from llmopt.config import EngineConfig
from llmopt.engine.model_runner import ModelRunner
from llmopt.engine.request import (
    FinishReason,
    Request,
    RequestOutput,
    RequestStatus,
    SamplingParams,
)
from llmopt.engine.sampler import Sampler
from llmopt.engine.scheduler import Scheduler
from llmopt.engine.tokenizer import BaseTokenizer, ByteTokenizer
from llmopt.model.modeling import DecoderOnlyModel, TinyLlamaConfig
from llmopt.utils.logging import get_logger
from llmopt.utils.metrics import MetricRegistry
from llmopt.utils.misc import device_of, resolve_dtype, seed_everything

__all__ = ["EngineOutput", "LLMEngine", "StreamingOutput"]

logger = get_logger("engine.llm")


@dataclass(slots=True)
class EngineOutput:
    """Final result for one request."""

    request_id: str
    token_ids: list[int]
    text: str = ""
    finish_reason: FinishReason | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    error: str | None = None

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    def to_request_output(self, prompt_token_ids: list[int] | None = None) -> RequestOutput:
        return RequestOutput(
            request_id=self.request_id,
            prompt_token_ids=prompt_token_ids or [],
            token_ids=self.token_ids,
            text=self.text,
            status=RequestStatus.FINISHED,
            finish_reason=self.finish_reason,
            metrics=self.metrics,
            error=self.error,
        )


@dataclass(slots=True)
class StreamingOutput:
    """Incremental update for a request mid-generation."""

    request_id: str
    delta: str = ""
    token_ids: list[int] = field(default_factory=list)
    text: str = ""
    finish_reason: FinishReason | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    done: bool = False


class LLMEngine:
    """Continuous-batching inference engine.

    Typical use::

        engine = LLMEngine.from_preset("tiny")
        future = engine.generate("hello", SamplingParams(max_tokens=16))
        print(future.result().text)

    Args:
        model: Decoder to serve.
        config: Engine configuration.
        tokenizer: Tokenizer used for text in/out.
        device: Torch device.
    """

    def __init__(
        self,
        model: DecoderOnlyModel,
        config: EngineConfig | None = None,
        tokenizer: BaseTokenizer | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        self.config = config or EngineConfig(model=model.config)
        seed_everything(self.config.seed)
        self.device = (
            torch.device(device)
            if device is not None
            else device_of(self.config.resolve_device())
        )
        self.tokenizer = tokenizer or ByteTokenizer()
        self.dtype = resolve_dtype(self.config.model.torch_dtype, str(self.device))
        self.model = model.to(self.device)
        self.model.eval()

        self.metrics = MetricRegistry()
        self.block_size = self.config.cache.block_size
        self.block_pool = BlockPool(
            self.config.cache.num_gpu_blocks, self.block_size, device=self.device
        )
        self.prefix_cache: PrefixCache | None = (
            PrefixCache(
                self.block_pool,
                self.block_size,
                max_entries=self.config.cache.num_gpu_blocks * 2,
            )
            if self.config.enable_prefix_caching
            else None
        )
        self.cache: PagedKVCache = self.model.allocate_cache(
            self.config.cache.num_gpu_blocks,
            self.block_size,
            self.dtype,
            self.device,
        )
        self.block_table = BlockTable(
            max_num_seqs=self.config.scheduler.max_num_seqs,
            max_blocks_per_seq=max(
                1, -(-self.config.scheduler.max_model_len // self.block_size)
            ),
            device=self.device,
        )
        self.block_table.set_block_size(self.block_size)
        self.runner = ModelRunner(
            self.model, self.cache, self.block_table, self.device, self.metrics
        )
        self.scheduler = Scheduler(
            self.config.scheduler, self.block_pool, self.prefix_cache, self.metrics
        )
        self.sampler = Sampler(self.device, self.config.seed)

        self._ids = itertools.count()
        self._lock = threading.RLock()
        self._futures: dict[str, Future[EngineOutput]] = {}
        self._block_map: dict[str, list[int]] = {}
        self._step_count = 0
        self._engine_start = time.perf_counter()

    # ------------------------------------------------------------------ setup

    @classmethod
    def from_preset(
        cls,
        preset: str = "tiny",
        *,
        vocab_size: int = 512,
        max_model_len: int = 1024,
        num_gpu_blocks: int | None = None,
        block_size: int = 16,
        max_num_seqs: int = 256,
        max_num_batched_tokens: int = 8192,
        gpu_memory_utilization: float = 0.9,
        enable_prefix_caching: bool = True,
        chunked_prefill: bool = True,
        device: torch.device | str | None = None,
        seed: int = 0,
    ) -> LLMEngine:
        """Build an engine around a randomly initialised model.

        This is the entry point used by tests, the benchmark harness, and
        ``examples/`` -- no checkpoint download required.

        Args:
            preset: ``"tiny"`` or ``"small"``.
            vocab_size: Vocabulary size for the generated model.
            max_model_len: Maximum total sequence length.
            num_gpu_blocks: Explicit KV-cache size. Derived from
                ``gpu_memory_utilization`` when omitted.
            block_size: Tokens per KV-cache block.
            max_num_seqs: Maximum concurrently running sequences.
            max_num_batched_tokens: Per-step token budget.
            gpu_memory_utilization: Fraction of device memory for weights+cache.
            enable_prefix_caching: Reuse shared prompt prefixes.
            chunked_prefill: Split long prefills across steps.
            device: Torch device; ``None`` resolves from the config.
            seed: Seed for weights and sampling.
        """
        from llmopt.config import num_blocks_for_gpu_memory
        from llmopt.model.modeling import tiny_model

        builders = {"tiny": TinyLlamaConfig.tiny, "small": TinyLlamaConfig.small}
        builder = builders.get(preset)
        if builder is None:
            raise ValueError(f"unknown preset {preset!r}; choose from {sorted(builders)}")

        model_config = builder(vocab_size)
        model = tiny_model(vocab_size, seed=seed, dtype=torch.float32)
        config = EngineConfig(
            model=model_config,
            seed=seed,
            device="auto" if device is None else str(device),
            enable_prefix_caching=enable_prefix_caching,
            chunked_prefill=chunked_prefill,
        )
        config.cache.block_size = block_size
        config.cache.enable_prefix_caching = enable_prefix_caching
        config.scheduler.max_num_seqs = max_num_seqs
        config.scheduler.max_num_batched_tokens = max_num_batched_tokens
        config.scheduler.max_model_len = min(
            max_model_len, model_config.max_position_embeddings
        )
        config.scheduler.prefill_chunk_size = min(
            config.scheduler.max_num_batched_tokens, config.scheduler.max_model_len
        )
        config.scheduler.enable_chunked_prefill = chunked_prefill
        resolved = config.resolve_device()
        config.cache.num_gpu_blocks = (
            num_gpu_blocks
            if num_gpu_blocks is not None
            else num_blocks_for_gpu_memory(
                model_config, config.cache, gpu_memory_utilization, resolved
            )
        )
        return cls(model=model, config=config, device=device)

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str,
        *,
        quantize: bool = False,
        num_gpu_blocks: int | None = None,
        block_size: int | None = None,
        max_model_len: int | None = None,
        max_num_seqs: int | None = None,
        max_num_batched_tokens: int | None = None,
        gpu_memory_utilization: float = 0.9,
        device: torch.device | str | None = None,
        config: EngineConfig | None = None,
    ) -> LLMEngine:
        """Load a checkpoint saved by ``DecoderOnlyModel.save_pretrained``.

        Runtime overrides are applied after the checkpoint's own config so a
        server can retune batching and cache size for a specific deployment.
        """
        from llmopt.config import num_blocks_for_gpu_memory

        model = DecoderOnlyModel.from_pretrained(model_dir, dtype=torch.float32)
        engine_config = config or EngineConfig()
        engine_config.model = model.config
        if quantize:
            from llmopt.quantization.quant_linear import quantize_model_

            quantize_model_(model, bits=4, group_size=64)
        if block_size is not None:
            engine_config.cache.block_size = block_size
        if max_num_seqs is not None:
            engine_config.scheduler.max_num_seqs = max_num_seqs
        if max_num_batched_tokens is not None:
            engine_config.scheduler.max_num_batched_tokens = max_num_batched_tokens
        if max_model_len is not None:
            engine_config.scheduler.max_model_len = min(
                max_model_len, model.config.max_position_embeddings
            )
        engine_config.scheduler.prefill_chunk_size = min(
            engine_config.scheduler.max_num_batched_tokens,
            engine_config.scheduler.max_model_len,
        )
        if device is not None:
            engine_config.device = str(device)
        engine_config.cache.num_gpu_blocks = (
            num_gpu_blocks
            if num_gpu_blocks is not None
            else num_blocks_for_gpu_memory(
                model.config,
                engine_config.cache,
                gpu_memory_utilization,
                engine_config.resolve_device(),
            )
        )
        return cls(model=model, config=engine_config, device=device)

    # --------------------------------------------------------------- requests

    def generate(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams | None = None,
        request_id: str | None = None,
    ) -> Future[EngineOutput]:
        """Submit a request and return a future for its final output.

        Args:
            prompt: Text (encoded with the engine tokenizer) or token ids.
            sampling_params: Decoding parameters.
            request_id: Optional explicit id; generated when omitted.
        """
        token_ids = self.tokenizer.encode(prompt) if isinstance(prompt, str) else list(prompt)
        if not token_ids:
            raise ValueError("prompt must contain at least one token")
        params = sampling_params or SamplingParams()
        with self._lock:
            rid = request_id or f"req-{next(self._ids)}"
        request = Request(
            request_id=rid,
            prompt_token_ids=token_ids,
            sampling_params=params,
            max_model_len=self.config.scheduler.max_model_len,
            arrival_time=time.perf_counter(),
        )
        return self.submit(request)

    def generate_sync(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams | None = None,
        request_id: str | None = None,
        max_steps: int | None = None,
    ) -> EngineOutput:
        """Submit a request and drive the loop until it finishes.

        :meth:`generate` only advances the engine one step, so the returned
        future settles after further calls to :meth:`step`. This blocking
        variant keeps stepping until this request completes, which is what a
        single-threaded client or an HTTP handler wants.

        Args:
            prompt: Text or token ids.
            sampling_params: Decoding parameters.
            request_id: Optional explicit id.
            max_steps: Guard against a non-draining scheduler.

        Returns:
            The completed :class:`EngineOutput`.
        """
        future = self.generate(prompt, sampling_params, request_id)
        steps = 0
        while not future.done():
            with self._lock:
                pending = self.scheduler.has_unfinished()
            if not pending:
                break
            self.step()
            steps += 1
            if max_steps is not None and steps > max_steps:
                raise TimeoutError(f"request did not finish within {max_steps} steps")
        return future.result()

    def submit(self, request: Request) -> Future[EngineOutput]:
        """Queue a prepared request and run one step to start it."""
        future: Future[EngineOutput] = Future()
        request.metrics["queued_at"] = time.perf_counter()
        with self._lock:
            self._futures[request.request_id] = future
            self._block_map.setdefault(request.request_id, [])
            self.block_table.allocate_row(request.request_id)
            self.scheduler.add_request(request)
        self.step()
        return future

    def abort(self, request_id: str) -> bool:
        """Cancel an in-flight request."""
        with self._lock:
            aborted = self.scheduler.abort(request_id)
            if not aborted:
                return False
            self._release_request(request_id)
            future = self._futures.pop(request_id, None)
            if future is not None and not future.done():
                future.set_result(
                    EngineOutput(
                        request_id=request_id,
                        token_ids=[],
                        finish_reason=FinishReason.ABORTED,
                        error="aborted",
                    )
                )
            return True

    # ------------------------------------------------------------------- loop

    def step(self) -> list[StreamingOutput]:
        """Run one engine iteration and return the deltas it produced.

        Finished requests are resolved even on steps that run no forward pass,
        so a request that completed on the previous iteration still settles.
        """
        with self._lock:
            self._drain_done_requests()
            planned = self.scheduler.step()
            try:
                if planned.is_empty:
                    return []
                self._apply_prefix_cache(planned)
                self._sync_block_maps(planned)
                model_input = self.runner.build_input(
                    planned.prefill_chunks, planned.decode, self._block_map
                )
                if model_input is None:
                    return []
                self._step_count += 1
                logits = self.runner.execute(model_input)
                return self._advance(planned, model_input, logits)
            finally:
                self._drain_finished(planned.finished)

    def run_until_idle(self, max_steps: int | None = None) -> int:
        """Step until no request is pending. Returns the number of steps taken."""
        steps = 0
        while self.scheduler.has_unfinished():
            self.step()
            steps += 1
            if max_steps is not None and steps > max_steps:
                raise TimeoutError(f"engine did not drain within {max_steps} steps")
        return steps

    def _drain_done_requests(self) -> None:
        """Release blocks for requests aborted or length-limited by the scheduler."""
        for request in list(self.scheduler.running):
            if request.is_done() and request.request_id in self._block_map:
                self._release_request(request.request_id)

    def _apply_prefix_cache(self, planned: object) -> None:
        """Publish blocks that filled during the previous step.

        Adopting an existing cache is the scheduler's job at admission time, so
        by the time a request reaches here its chain is final for this step.
        """
        if self.prefix_cache is None:
            return
        for request in planned.running:  # type: ignore[attr-defined]
            if not request.is_done():
                self.scheduler.cache_prefix(request)

    def _sync_block_maps(self, planned: object) -> None:
        """Refresh the block map from the pool after the scheduler grew it."""
        for request in planned.running:  # type: ignore[attr-defined]
            chain = self.block_pool.chains.get(request.request_id, [])
            self._block_map[request.request_id] = list(chain)
            self.block_table.set_blocks(request.request_id, chain)

    def _advance(
        self, planned: object, model_input: object, logits: torch.Tensor
    ) -> list[StreamingOutput]:
        """Sample tokens and update request state after a forward pass."""
        deltas: list[StreamingOutput] = []
        for request_id in model_input.seq_ids:  # type: ignore[attr-defined]
            request = self.scheduler.get_request(request_id)
            if request is not None:
                request.num_computed_tokens += model_input.q_len  # type: ignore[attr-defined]

        # The runner packs prefill ahead of decode and may drop the decode rows
        # when both are scheduled, so intersect with what was actually executed.
        rows = {rid: i for i, rid in enumerate(model_input.seq_ids)}  # type: ignore[attr-defined]
        decode_requests: list[Request] = [
            r
            for r in planned.decode  # type: ignore[attr-defined]
            if r.request_id in rows and not r.is_done()
        ]
        if not decode_requests:
            return deltas

        batch = torch.stack([logits[rows[r.request_id]] for r in decode_requests])
        token_ids, logprobs = self._sample(decode_requests, batch)

        for index, request in enumerate(decode_requests):
            token = int(token_ids[index].item())
            if self._check_stop(request, token):
                self._mark_finished(request, FinishReason.STOP)
                continue
            request.output_token_ids.append(token)
            request.metrics["last_logprob"] = float(logprobs[index].item())
            # Check the budget after appending so the last allowed token counts.
            if request.num_output_tokens >= request.sampling_params.max_tokens:
                self._mark_finished(request, FinishReason.LENGTH)
                continue
            deltas.append(
                StreamingOutput(
                    request_id=request.request_id,
                    delta=self.tokenizer.decode([token]),
                    token_ids=list(request.output_token_ids),
                )
            )
        return deltas

    def _sample(
        self, requests: list[Request], batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample one token per row, fusing the call when params are identical."""
        params_list = [r.sampling_params for r in requests]
        if all(p == params_list[0] for p in params_list[1:]):
            return self.sampler.forward(batch, params_list[0], [list(r.output_token_ids) for r in requests])

        tokens: list[int] = []
        logprobs: list[float] = []
        for index, params in enumerate(params_list):
            token, logprob = self.sampler.forward(
                batch[index : index + 1], params, [list(requests[index].output_token_ids)]
            )
            tokens.append(int(token.item()))
            logprobs.append(float(logprob.item()))
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(logprobs)

    def _check_stop(self, request: Request, token: int) -> bool:
        if request.sampling_params.ignore_eos:
            return False
        if token == self.tokenizer.eos_token_id:
            return True
        if not request.sampling_params.stop:
            return False
        window = self.tokenizer.decode(request.output_token_ids + [token])
        return any(stop and stop in window for stop in request.sampling_params.stop)

    def _mark_finished(self, request: Request, reason: FinishReason) -> None:
        request.status = RequestStatus.FINISHED
        request.finish_reason = reason
        elapsed = max(1e-9, time.perf_counter() - request.arrival_time)
        request.metrics.update(
            {
                "latency_s": round(elapsed, 6),
                "ttft_s": request.metrics.get("ttft_s", round(elapsed, 6)),
                "num_output_tokens": float(request.num_output_tokens),
                "tokens_per_s": round(request.num_output_tokens / elapsed, 3),
            }
        )

    def _drain_finished(self, finished: list[Request]) -> None:
        for request in finished:
            self._release_request(request.request_id)
            future = self._futures.pop(request.request_id, None)
            if future is None or future.done():
                continue
            future.set_result(
                EngineOutput(
                    request_id=request.request_id,
                    token_ids=list(request.output_token_ids),
                    text=self.tokenizer.decode(request.output_token_ids),
                    finish_reason=request.finish_reason,
                    metrics=dict(request.metrics),
                    error=request.error,
                )
            )

    def _release_request(self, request_id: str) -> None:
        self._block_map.pop(request_id, None)
        self.block_table.free_row(request_id)

    # ---------------------------------------------------------------- helpers

    def run(self, requests: Iterable[Request], max_steps: int | None = None) -> list[EngineOutput]:
        """Submit every request, then drive them all to completion."""
        futures = [self.submit(request) for request in requests]
        self.run_until_idle(max_steps)
        return [future.result() for future in futures]

    def stream(
        self, prompt: str | list[int], params: SamplingParams | None = None
    ) -> Iterator[StreamingOutput]:
        """Yield deltas as tokens are produced, then one final chunk."""
        request_id = f"stream-{next(self._ids)}"
        future = self.generate(prompt, params, request_id=request_id)
        while not future.done():
            deltas = self.step()
            for delta in deltas:
                if delta.request_id == request_id:
                    yield delta
        result = future.result()
        yield StreamingOutput(
            request_id=result.request_id,
            text=result.text,
            token_ids=result.token_ids,
            finish_reason=result.finish_reason,
            metrics=result.metrics,
            done=True,
        )

    def chat(
        self, messages: list[dict[str, str]], params: SamplingParams | None = None
    ) -> EngineOutput:
        """Minimal chat wrapper that flattens messages into one prompt."""
        prompt = "\n".join(
            f"{message.get('role', 'user')}: {message.get('content', '')}" for message in messages
        )
        return self.generate(prompt, params).result()

    def reset_prefix_cache(self) -> None:
        if self.prefix_cache is not None:
            self.prefix_cache.reset()

    def stats(self) -> dict[str, Any]:
        """Aggregate engine, scheduler, cache and prefix-cache statistics."""
        snapshot = self.metrics.snapshot()
        token_total = snapshot.get("llmopt_batch_tokens_sum", 0.0)
        token_count = snapshot.get("llmopt_batch_tokens_count", 0.0)
        return {
            "steps": self._step_count,
            "uptime_s": round(time.perf_counter() - self._engine_start, 3),
            "scheduler": self.scheduler.stats.to_dict(),
            "scheduler_state": self.scheduler.snapshot(),
            "block_pool": self.block_pool.stats(),
            "prefix_cache": self.prefix_cache.stats() if self.prefix_cache else None,
            "kv_cache_bytes": self.cache.memory_bytes(),
            "mean_batch_tokens": round(token_total / token_count, 2) if token_count else 0.0,
        }

    def metrics_text(self) -> str:
        return self.metrics.render()

    def __repr__(self) -> str:
        return (
            f"LLMEngine(device={self.device}, dtype={self.dtype}, "
            f"blocks={self.config.cache.num_gpu_blocks}x{self.block_size})"
        )
