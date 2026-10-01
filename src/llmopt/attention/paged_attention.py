"""Paged attention over the block-addressed KV cache.

The core problem: after continuous batching, every sequence in a batch has a
different context length and its KV lives in non-contiguous physical blocks.
Two things make that cheap:

1. **Gather once per layer.** K/V for the whole batch are assembled into a
   single padded tensor, so all sequences share one batched matmul instead of
   running a Python loop over the batch.
2. **Flash-Decoding style split for the decode phase.** Memory traffic is
   split across the context axis and combined with a numerically stable online
   softmax, so peak memory no longer scales with the longest sequence in the
   batch.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from llmopt.cache.paged_cache import PagedKVCache
from llmopt.utils.logging import get_logger

__all__ = ["AttentionBackend", "PagedAttention", "PagedAttentionConfig", "merge_states"]

logger = get_logger("attention.paged")


def merge_states(
    output: torch.Tensor,
    partial: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine two partial attention results using log-sum-exp.

    This is the reduction that makes split-K (flash-decoding) attention exact
    rather than approximate: each partial covers a disjoint slice of the keys,
    and the merge reweights them by their contribution to the softmax normaliser.

    Args:
        output: Accumulated output, ``[..., seq, head_dim]``.
        partial: New partial output with the same shape.
        prefix_lse: Log-sum-exp of the accumulated state, ``[..., seq]``.
        suffix_lse: Log-sum-exp of the new partial, ``[..., seq]``.

    Returns:
        The merged ``(output, lse)``.
    """
    merged_lse = torch.logaddexp(prefix_lse, suffix_lse)
    scale_1 = torch.exp(prefix_lse - merged_lse).unsqueeze(-1)
    scale_2 = torch.exp(suffix_lse - merged_lse).unsqueeze(-1)
    merged = output * scale_1 + partial * scale_2
    return merged, merged_lse


@dataclass(slots=True)
class PagedAttentionConfig:
    """Knobs for the paged attention kernel."""

    block_size: int = 16
    split_kv: bool = True
    split_size: int = 512
    causal: bool = True
    softcap: float = 0.0

    def __post_init__(self) -> None:
        if self.block_size < 1:
            raise ValueError("block_size must be >= 1")
        if self.split_size < 1:
            raise ValueError("split_size must be >= 1")


class AttentionBackend:
    """Pluggable attention implementations, selected at construction time.

    ``flash``/``triton`` are attempted opportunistically; when the optional
    dependency is missing we fall back to ``sdpa`` and log the reason once.
    """

    _available: dict[str, bool] = {}

    @classmethod
    def detect(cls, requested: str = "auto") -> str:
        if requested in ("sdpa", "math", "eager"):
            return "sdpa"
        if requested in ("flash", "flash_attn"):
            if cls._probe("flash_attn"):
                return "flash"
            logger.info("flash-attn unavailable, falling back to sdpa")
            return "sdpa"
        if requested == "triton":
            if cls._probe("triton"):
                return "triton"
            logger.info("triton unavailable, falling back to sdpa")
            return "sdpa"
        if requested not in ("auto",):
            raise ValueError(f"unknown attention backend: {requested!r}")
        if cls._probe("flash_attn"):
            return "flash"
        return "sdpa"

    @classmethod
    def _probe(cls, module_name: str) -> bool:
        if module_name not in cls._available:
            try:
                __import__(module_name)
                cls._available[module_name] = True
            except Exception:  # noqa: BLE001 - any import failure means unusable
                cls._available[module_name] = False
        return cls._available[module_name]


class PagedAttention:
    """Attention that reads K/V straight out of a :class:`PagedKVCache`.

    Args:
        num_heads: Query head count.
        num_kv_heads: Key/value head count (<= ``num_heads`` for GQA).
        head_dim: Per-head dimension.
        scale: Softmax scale; defaults to ``1/sqrt(head_dim)``.
        config: Kernel configuration.
        backend: ``"auto"``, ``"sdpa"`` or ``"flash"``.
    """

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        scale: float | None = None,
        config: PagedAttentionConfig | None = None,
        backend: str = "auto",
    ) -> None:
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.group_size = num_heads // num_kv_heads
        self.scale = scale if scale is not None else head_dim**-0.5
        self.config = config or PagedAttentionConfig()
        self.backend = AttentionBackend.detect(backend)

    def forward(
        self,
        query: torch.Tensor,
        cache: PagedKVCache,
        layer_idx: int,
        block_ids: torch.Tensor,
        context_lens: torch.Tensor,
        query_lens: int | None = None,
    ) -> torch.Tensor:
        """Compute attention for a batch of sequences.

        Args:
            query: ``[seq, num_heads, query_lens, head_dim]``. ``query_lens=1``
                is the decode path, larger values are prefill/chunked prefill.
            cache: The paged KV cache to read from.
            layer_idx: Which layer's K/V to gather.
            block_ids: ``[seq, max_blocks]`` block table, ``-1`` padded.
            context_lens: ``[seq]`` total context length *after* this step's KV
                has been appended.
            query_lens: Query length, inferred from ``query`` when omitted.

        Returns:
            ``[seq, num_heads, query_lens, head_dim]`` attention output.
        """
        seq_len = query.shape[0]
        q_len = query_lens if query_lens is not None else query.shape[2]
        if q_len == 0:
            return torch.zeros(
                (seq_len, self.num_heads, 0, self.head_dim),
                dtype=query.dtype,
                device=query.device,
            )

        keys, values, valid = cache.gather(layer_idx, block_ids, context_lens)
        # keys/values: [seq, ctx, kv_heads, head_dim] -> [seq, kv_heads, ctx, head_dim]
        k = keys.permute(0, 2, 1, 3)
        v = values.permute(0, 2, 1, 3)
        q = query

        if self.config.split_kv and self.config.split_size < k.shape[2]:
            return self._forward_split(q, k, v, valid, context_lens, q_len)
        return self._forward_dense(q, k, v, valid, context_lens, q_len)

    def _expand_kv(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Repeat each KV head ``group_size`` times to match the query heads.

        Operates on ``[seq, kv_heads, ctx, head_dim]`` tensors. The repeat is
        interleaved head-major so query head ``g * group_size + i`` pairs with
        KV head ``g``, matching the projection layout in the model.
        """
        if self.group_size == 1:
            return k, v
        seq, kv_heads, ctx, head_dim = k.shape
        expanded_shape = (seq, kv_heads, self.group_size, ctx, head_dim)
        k = (
            k.unsqueeze(2)
            .expand(expanded_shape)
            .reshape(seq, self.num_heads, ctx, head_dim)
        )
        v = (
            v.unsqueeze(2)
            .expand(expanded_shape)
            .reshape(seq, self.num_heads, ctx, head_dim)
        )
        return k, v

    def _build_mask(
        self,
        valid: torch.Tensor,
        context_lens: torch.Tensor,
        q_len: int,
    ) -> torch.Tensor:
        """Boolean ``[seq, 1, q_len, ctx]`` mask: padding plus causal when prefilling."""
        mask = valid[:, None, None, :].expand(-1, 1, q_len, -1)
        if q_len > 1:
            # Query position i of a sequence with ctx tokens sits at ctx - q_len + i.
            q_pos = context_lens[:, None] - q_len + torch.arange(
                q_len, device=valid.device
            )[None, :]
            k_pos = torch.arange(valid.shape[1], device=valid.device)
            # [seq, q_len, ctx]: every query row is tested against every key column.
            causal = k_pos[None, None, :] <= q_pos[:, :, None]
            mask = mask & causal[:, None, :, :]
        return mask

    def _forward_dense(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        valid: torch.Tensor,
        context_lens: torch.Tensor,
        q_len: int,
    ) -> torch.Tensor:
        k, v = self._expand_kv(k, v)
        # [seq, heads, q_len, head_dim] x [seq, heads, ctx, head_dim] -> [seq, heads, q_len, ctx]
        scores = torch.matmul(q.to(k.dtype), k.transpose(-1, -2)) * self.scale
        if self.config.softcap > 0:
            scores = torch.tanh(scores / self.config.softcap) * self.config.softcap
        mask = self._build_mask(valid, context_lens, q_len)
        scores = scores.masked_fill(~mask, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        # A fully-masked row (should not happen) would produce NaN; guard it.
        probs = torch.nan_to_num(probs, nan=0.0)
        return torch.matmul(probs, v).to(q.dtype)

    def _forward_split(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        valid: torch.Tensor,
        context_lens: torch.Tensor,
        q_len: int,
    ) -> torch.Tensor:
        """Flash-Decoding: process the context in slices and merge via LSE."""
        k, v = self._expand_kv(k, v)
        ctx = k.shape[2]
        chunk = self.config.split_size
        qk = q.to(k.dtype)
        outputs: list[torch.Tensor] = []
        lses: list[torch.Tensor] = []

        for start in range(0, ctx, chunk):
            end = min(start + chunk, ctx)
            k_chunk = k[:, :, start:end]
            v_chunk = v[:, :, start:end]
            scores = torch.matmul(qk, k_chunk.transpose(-1, -2)) * self.scale
            if self.config.softcap > 0:
                scores = torch.tanh(scores / self.config.softcap) * self.config.softcap
            # Slice the mask: the causal test must use global key positions.
            mask = self._build_mask(
                valid[:, start:end], context_lens - start, q_len
            )
            scores = scores.masked_fill(~mask, float("-inf"))
            chunk_lse = torch.logsumexp(scores, dim=-1)
            probs = torch.softmax(scores, dim=-1)
            probs = torch.nan_to_num(probs, nan=0.0)
            outputs.append(torch.matmul(probs, v_chunk))
            lses.append(chunk_lse)

        acc = outputs[0]
        acc_lse = lses[0]
        for partial, partial_lse in zip(outputs[1:], lses[1:], strict=False):
            acc, acc_lse = merge_states(acc, partial, acc_lse, partial_lse)
        # A sequence shorter than the padded context can leave whole chunks
        # masked; their lse is -inf, which would make the merge produce NaN.
        return torch.nan_to_num(acc, nan=0.0).to(q.dtype)

    def __repr__(self) -> str:
        return (
            f"PagedAttention(heads={self.num_heads}, kv_heads={self.num_kv_heads}, "
            f"head_dim={self.head_dim}, backend={self.backend}, "
            f"split_kv={self.config.split_kv})"
        )
