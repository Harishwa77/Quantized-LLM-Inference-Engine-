"""Model runner: translates scheduled work into forward passes.

This layer owns tensor layout. The scheduler hands over ``(request, start, end)``
prefill chunks and a decode set; the runner flattens them into one packed batch,
builds the block table view, runs the model, and returns one logit row per
sequence.

Two invariants matter and are enforced here:

* ``context_lens`` counts *committed* tokens, not allocated block capacity. A
  sequence must never be able to attend to the unwritten tail of its last block,
  so padded positions are excluded via the validity mask.
* Prefill and decode never share a step: they have different query lengths, and
  the causal mask differs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from llmopt.cache.paged_cache import PagedKVCache
from llmopt.engine.request import Request
from llmopt.model.modeling import DecoderOnlyModel
from llmopt.utils.logging import get_logger
from llmopt.utils.metrics import MetricRegistry

__all__ = ["ModelInput", "ModelRunner"]

logger = get_logger("engine.runner")


@dataclass(slots=True)
class ModelInput:
    """Packed tensors for one forward pass.

    Attributes:
        input_ids: ``[batch * q_len]`` token ids for the step.
        positions: ``[batch * q_len]`` absolute positions.
        block_ids: ``[batch, max_blocks]`` block table view, ``-1`` padded.
        context_lens: ``[batch]`` committed tokens plus this step's write.
        token_offsets: ``[batch]`` where each sequence's write begins.
        q_len: Tokens per sequence; ``1`` for pure decode.
        seq_ids: Request ids, in row order.
    """

    input_ids: torch.Tensor
    positions: torch.Tensor
    block_ids: torch.Tensor
    context_lens: torch.Tensor
    token_offsets: torch.Tensor
    q_len: int
    seq_ids: list[str] = field(default_factory=list)

    @property
    def num_tokens(self) -> int:
        return int(self.input_ids.numel())

    @property
    def batch_size(self) -> int:
        return len(self.seq_ids)


class ModelRunner:
    """Owns the model, the paged cache, and the block table.

    Args:
        model: Decoder to execute.
        cache: Paged KV cache allocated for ``model``.
        block_table: Logical-to-physical block mapping.
        device: Device tensors are placed on.
        metrics: Optional metrics registry.
    """

    def __init__(
        self,
        model: DecoderOnlyModel,
        cache: PagedKVCache,
        block_table: object,
        device: torch.device | str = "cpu",
        metrics: MetricRegistry | None = None,
    ) -> None:
        self.model = model
        self.cache = cache
        self.block_table = block_table
        self.device = torch.device(device)
        self.metrics = metrics
        self.model.eval()

    @torch.inference_mode()
    def execute(self, model_input: ModelInput) -> torch.Tensor:
        """Run the model, returning ``[batch, vocab]`` last-token logits.

        A sequence that only prefilled still yields a usable last-token logit,
        which the engine reuses for eager first-token generation instead of
        paying for a separate decode step.
        """
        output = self.model(
            model_input.input_ids,
            model_input.positions,
            cache=self.cache,
            block_ids=model_input.block_ids,
            context_lens=model_input.context_lens,
            token_offsets=model_input.token_offsets,
            q_len=model_input.q_len,
        )
        if self.metrics is not None:
            self.metrics.counter("llmopt_forward_passes", "Model forward passes").inc()
            self.metrics.histogram("llmopt_batch_tokens", "Tokens per forward").observe(
                float(model_input.num_tokens)
            )
        return output.last_logits()

    def build_input(
        self,
        prefill_chunks: Sequence[tuple[Request, int, int]],
        decode_requests: Sequence[Request],
        block_map: dict[str, list[int]],
    ) -> ModelInput | None:
        """Pack this step's work into a single batch.

        Prefill takes priority when both are present; the decode rows are left
        for the following step rather than being silently reshaped.
        """
        if prefill_chunks:
            return self._build_prefill(prefill_chunks, block_map)
        if decode_requests:
            return self._build_decode(decode_requests, block_map)
        return None

    def _build_prefill(
        self,
        prefill_chunks: Sequence[tuple[Request, int, int]],
        block_map: dict[str, list[int]],
    ) -> ModelInput:
        width = prefill_chunks[0][2] - prefill_chunks[0][1]
        for _, start, end in prefill_chunks:
            if end - start != width:  # pragma: no cover - scheduler invariant
                raise ValueError("prefill chunks within a step must share a length")

        input_chunks: list[torch.Tensor] = []
        position_chunks: list[torch.Tensor] = []
        seq_ids: list[str] = []
        offsets: list[int] = []
        context_lens: list[int] = []

        for request, start, end in prefill_chunks:
            tokens = request.prompt_token_ids[start:end]
            if len(tokens) != width:  # pragma: no cover - scheduler invariant
                raise ValueError("prefill chunk is shorter than the declared width")
            input_chunks.append(torch.tensor(tokens, dtype=torch.long))
            position_chunks.append(torch.arange(start, end, dtype=torch.long))
            seq_ids.append(request.request_id)
            offsets.append(start)
            context_lens.append(end)

        return self._finalize(
            torch.cat(input_chunks),
            torch.cat(position_chunks),
            seq_ids,
            offsets,
            context_lens,
            width,
            block_map,
        )

    def _build_decode(
        self, decode_requests: Sequence[Request], block_map: dict[str, list[int]]
    ) -> ModelInput:
        input_chunks: list[torch.Tensor] = []
        position_chunks: list[torch.Tensor] = []
        seq_ids: list[str] = []
        offsets: list[int] = []
        context_lens: list[int] = []

        for request in decode_requests:
            token = self._last_input_token(request)
            input_chunks.append(torch.tensor([token], dtype=torch.long))
            position_chunks.append(torch.tensor([request.num_computed_tokens], dtype=torch.long))
            seq_ids.append(request.request_id)
            offsets.append(request.num_computed_tokens)
            context_lens.append(request.num_computed_tokens + 1)

        return self._finalize(
            torch.cat(input_chunks),
            torch.cat(position_chunks),
            seq_ids,
            offsets,
            context_lens,
            1,
            block_map,
        )

    @staticmethod
    def _last_input_token(request: Request) -> int:
        """The token to feed on a decode step: the one just produced, else the last prompt token."""
        if request.output_token_ids:
            return request.output_token_ids[-1]
        return request.prompt_token_ids[-1]

    def _finalize(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        seq_ids: list[str],
        offsets: list[int],
        context_lens: list[int],
        q_len: int,
        block_map: dict[str, list[int]],
    ) -> ModelInput:
        """Move tensors to the device and build the block table view."""
        width = max((len(blocks) for blocks in block_map.values()), default=1)
        width = max(width, 1)
        table = torch.full((len(seq_ids), width), -1, dtype=torch.int32, device=self.device)
        for row, request_id in enumerate(seq_ids):
            blocks = block_map.get(request_id, [])
            if blocks:
                table[row, : len(blocks)] = torch.tensor(blocks, dtype=torch.int32)

        return ModelInput(
            input_ids=input_ids.to(self.device),
            positions=positions.to(self.device),
            block_ids=table,
            context_lens=torch.tensor(context_lens, dtype=torch.long, device=self.device),
            token_offsets=torch.tensor(offsets, dtype=torch.long, device=self.device),
            q_len=q_len,
            seq_ids=seq_ids,
        )

    def reset(self) -> None:
        self.cache.reset()

    def __repr__(self) -> str:
        return f"ModelRunner(device={self.device}, {self.model!r})"
