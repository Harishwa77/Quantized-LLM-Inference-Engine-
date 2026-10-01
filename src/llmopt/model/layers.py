"""Transformer building blocks: RMSNorm, SwiGLU MLP, decoder layers."""

from __future__ import annotations

import torch
from torch import nn

from llmopt.attention.paged_attention import PagedAttention, PagedAttentionConfig
from llmopt.model.rotary import RotaryEmbedding, apply_rotary_pos_emb

__all__ = ["RMSNorm", "SwiGLUMLP", "PagedAttentionLayer", "DecoderLayer"]


class RMSNorm(nn.Module):
    """Root-mean-square layer norm, computed in fp32 for stability.

    Args:
        hidden_size: Normalized dimension.
        eps: Epsilon added to the mean square before the square root.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x32 = x.to(torch.float32)
        normed = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (normed * self.weight.to(torch.float32)).to(input_dtype)


class SwiGLUMLP(nn.Module):
    """Gated feed-forward block: ``down(silu(gate(x)) * up(x))``."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class PagedAttentionLayer(nn.Module):
    """Attention that writes its KV into a :class:`PagedKVCache`.

    Args:
        config: Model architecture config.
        layer_idx: Index of this layer in the stack, used to address the cache.
        backend: Attention backend selector.
    """

    def __init__(self, config: object, layer_idx: int, backend: str = "auto") -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.rotary = RotaryEmbedding(
            self.head_dim, config.max_position_embeddings, config.rope_theta
        )
        self.attn = PagedAttention(
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            config=PagedAttentionConfig(block_size=16),
            backend=backend,
        )

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        cache: object | None = None,
        block_ids: torch.Tensor | None = None,
        context_lens: torch.Tensor | None = None,
        token_offsets: torch.Tensor | None = None,
        q_len: int = 1,
    ) -> torch.Tensor:
        """Run attention, persisting K/V when a cache is supplied.

        Args:
            hidden: ``[seq * q_len, hidden_size]`` tokens of the current step.
            positions: ``[seq * q_len]`` absolute positions.
            cache: Target KV cache, or ``None`` for the stateless path.
            block_ids: ``[seq, max_blocks]`` block table for the batch.
            context_lens: ``[seq]`` context length *after* this step's write.
            token_offsets: ``[seq]`` where each sequence's write starts.
            q_len: Query tokens per sequence (``1`` for decode, more for prefill).

        Returns:
            ``[seq * q_len, hidden_size]`` attention output.
        """
        total = hidden.shape[0]
        seq_len = total // max(1, q_len)
        # [seq * q_len, hidden] -> [seq, q_len, heads, head_dim] -> [seq, heads, q_len, head_dim]
        q = self.q_proj(hidden).view(seq_len, q_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden).view(seq_len, q_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden).view(seq_len, q_len, self.num_kv_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Table is [seq, q_len, head_dim] -> [seq, 1, q_len, head_dim] so it
        # broadcasts over each sequence's own heads.
        cos, sin = self.rotary(positions, dtype=q.dtype)
        cos = cos.view(seq_len, q_len, self.head_dim).unsqueeze(1)
        sin = sin.view(seq_len, q_len, self.head_dim).unsqueeze(1)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        # Only q and k carry positional information; v passes through unchanged.
        batched_k = k.permute(0, 2, 1, 3).contiguous()  # [seq, q_len, kv_heads, head_dim]
        batched_v = v.permute(0, 2, 1, 3).contiguous()

        if cache is not None and block_ids is not None and context_lens is not None:
            offsets = token_offsets if token_offsets is not None else context_lens - q_len
            if q_len == 1:
                cache.write_decode(
                    self.layer_idx,
                    block_ids,
                    offsets,
                    batched_k[:, 0],
                    batched_v[:, 0],
                )
            else:
                cache.write(self.layer_idx, block_ids, offsets, batched_k, batched_v)
            attn_out = self.attn.forward(
                q, cache, self.layer_idx, block_ids, context_lens, q_len  # type: ignore[arg-type]
            )
        else:
            attn_out = self._dense_attention(q, k, v, q_len)

        attn_out = attn_out.permute(0, 2, 1, 3).reshape(total, -1)
        return self.o_proj(attn_out)

    def _dense_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_len: int
    ) -> torch.Tensor:
        """Stateless reference path: per-sequence dense attention, no cache.

        ``q``/``k``/``v`` arrive as ``[seq, heads, q_len, head_dim]``.
        """
        seq_len = q.shape[0]
        if seq_len == 1:
            valid = torch.ones(1, k.shape[2], dtype=torch.bool, device=k.device)
            lens = torch.full((1,), k.shape[2], dtype=torch.long, device=k.device)
            return self.attn._forward_dense(q, k, v, valid, lens, q_len)
        outputs: list[torch.Tensor] = []
        for i in range(seq_len):
            valid = torch.ones(1, k.shape[2], dtype=torch.bool, device=k.device)
            lens = torch.full((1,), k.shape[2], dtype=torch.long, device=k.device)
            outputs.append(
                self.attn._forward_dense(q[i : i + 1], k[i : i + 1], v[i : i + 1], valid, lens, q_len)
            )
        return torch.cat(outputs, dim=0)


class DecoderLayer(nn.Module):
    """Pre-norm decoder block: attention then SwiGLU MLP, both residual."""

    def __init__(self, config: object, layer_idx: int, backend: str = "auto") -> None:
        super().__init__()
        self.self_attn = PagedAttentionLayer(config, layer_idx, backend)
        self.mlp = SwiGLUMLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        cache: object | None = None,
        block_ids: torch.Tensor | None = None,
        context_lens: torch.Tensor | None = None,
        token_offsets: torch.Tensor | None = None,
        q_len: int = 1,
    ) -> torch.Tensor:
        residual = hidden
        hidden = self.input_layernorm(hidden)
        hidden = self.self_attn(
            hidden, positions, cache, block_ids, context_lens, token_offsets, q_len
        )
        hidden = residual + hidden
        residual = hidden
        hidden = self.post_attention_layernorm(hidden)
        hidden = self.mlp(hidden)
        return residual + hidden
