"""Rotary position embeddings.

Uses the Llama-style interleaving-free layout where the rotation is split into
two halves of ``head_dim``. ``precompute`` materialises the cos/sin tables once
and slices them per call, which is cheaper than recomputing them per step.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["RotaryEmbedding", "apply_rotary_pos_emb", "rotate_half"]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the two halves of the last dimension: ``[a, b] -> [-b, a]``."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to query and key tensors.

    ``q`` is ``[seq, heads, q_len, head_dim]`` and ``k`` is
    ``[seq, kv_heads, q_len, head_dim]``. ``cos``/``sin`` must be shaped
    ``[seq, 1, q_len, head_dim]`` so each sequence's positions broadcast over
    its own heads -- a flat ``[seq * q_len, head_dim]`` table would silently
    misalign whenever a batch holds more than one sequence.

    Args:
        q: Query tensor.
        k: Key tensor.
        cos: Cosine table, broadcastable to ``q``'s shape.
        sin: Sine table, broadcastable to ``q``'s shape.

    Returns:
        The rotated ``(q, k)``.
    """
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class RotaryEmbedding(nn.Module):
    """Precomputed rotary position embedding tables.

    Args:
        head_dim: Per-head dimension; must be even.
        max_position_embeddings: Table length.
        base: Rotary base (``theta`` in the model config).
        dtype: Table dtype.
    """

    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int = 4096,
        base: float = 10000.0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary embeddings")
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        positions = torch.arange(max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(
        self, positions: torch.Tensor, dtype: torch.dtype | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Look up cos/sin for absolute ``positions``.

        Args:
            positions: 1-D tensor of position indices.
            dtype: Optional cast for the returned tables.

        Returns:
            ``(cos, sin)`` each ``[len(positions), head_dim]``.
        """
        if int(positions.max().item()) >= self.max_position_embeddings:
            raise ValueError(
                f"position {int(positions.max().item())} exceeds table length "
                f"{self.max_position_embeddings}"
            )
        cos = self.cos_cached[positions]
        sin = self.sin_cached[positions]
        if dtype is not None:
            cos, sin = cos.to(dtype), sin.to(dtype)
        return cos, sin
