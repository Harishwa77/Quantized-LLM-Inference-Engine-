"""Model definition: layers, rotary embeddings, and the decoder."""

from __future__ import annotations

from llmopt.model.layers import (
    DecoderLayer,
    PagedAttentionLayer,
    RMSNorm,
    SwiGLUMLP,
)
from llmopt.model.modeling import (
    DecoderOnlyModel,
    ModelOutput,
    TinyLlamaConfig,
    estimate_flops_per_token,
    tiny_model,
)
from llmopt.model.rotary import RotaryEmbedding, apply_rotary_pos_emb, rotate_half

__all__ = [
    "DecoderLayer",
    "DecoderOnlyModel",
    "ModelOutput",
    "PagedAttentionLayer",
    "RMSNorm",
    "RotaryEmbedding",
    "SwiGLUMLP",
    "TinyLlamaConfig",
    "apply_rotary_pos_emb",
    "estimate_flops_per_token",
    "rotate_half",
    "tiny_model",
]
