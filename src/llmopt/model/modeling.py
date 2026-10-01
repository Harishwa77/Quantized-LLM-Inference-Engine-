"""Decoder-only transformer with paged-cache-aware forward passes."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

import torch
from torch import nn

from llmopt.model.layers import RMSNorm, DecoderLayer
from llmopt.utils.logging import get_logger

__all__ = ["DecoderOnlyModel", "ModelOutput", "TinyLlamaConfig", "tiny_model"]

logger = get_logger("model.decoder")


@dataclass(slots=True)
class ModelOutput:
    """Logits plus whatever the caller needs to continue decoding."""

    logits: torch.Tensor
    hidden_states: torch.Tensor | None = None

    def last_logits(self) -> torch.Tensor:
        """``[seq, vocab]`` logits for the final position of each sequence."""
        return self.logits[:, -1, :]


class DecoderOnlyModel(nn.Module):
    """A Llama-style decoder evaluated against a paged KV cache.

    The forward signature is shaped around the engine's needs: the batch is a
    flat list of sequences, each contributing ``q_len`` tokens this step, with
    per-sequence cache metadata supplied as tensors.

    Args:
        config: Architecture config.
        backend: Attention backend selector.
        dtype: Parameter dtype.
    """

    def __init__(self, config: object, backend: str = "auto", dtype: torch.dtype | None = None):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [DecoderLayer(config, i, backend) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        dtype = dtype or torch.float32
        self.to(dtype)
        self._attention_mask: torch.Tensor | None = None

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def parameter_bytes(self) -> int:
        return sum(p.numel() * p.element_size() for p in self.parameters())

    def _causal_mask(
        self, q_len: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Additive causal mask of shape ``[1, 1, q_len, q_len]``."""
        cached = self._attention_mask
        if (
            cached is not None
            and cached.shape[-1] == q_len
            and cached.device == device
            and cached.dtype == dtype
        ):
            return cached
        positions = torch.arange(q_len, device=device)
        mask = positions[:, None] < positions[None, :]
        min_value = torch.finfo(dtype).min
        additive = torch.zeros(q_len, q_len, device=device, dtype=dtype)
        additive = additive.masked_fill(mask, min_value)
        additive = additive.view(1, 1, q_len, q_len)
        self._attention_mask = additive
        return additive

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        cache: object | None = None,
        block_ids: torch.Tensor | None = None,
        context_lens: torch.Tensor | None = None,
        token_offsets: torch.Tensor | None = None,
        q_len: int = 1,
    ) -> ModelOutput:
        """Run the model over one engine step.

        Args:
            input_ids: ``[seq * q_len]`` token ids for the step.
            positions: ``[seq * q_len]`` absolute positions.
            cache: Optional paged KV cache.
            block_ids: Optional ``[seq, max_blocks]`` block table.
            context_lens: Optional ``[seq]`` post-write context lengths.
            token_offsets: Optional ``[seq]`` write start offsets.
            q_len: Tokens per sequence in this step.

        Returns:
            A :class:`ModelOutput` with ``[seq, q_len, vocab]`` logits.
        """
        # Accept both [seq * q_len] and [seq, q_len] token layouts.
        if input_ids.dim() > 1:
            input_ids = input_ids.reshape(-1)
        if positions.dim() > 1:
            positions = positions.reshape(-1)
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = layer(
                hidden,
                positions,
                cache=cache,
                block_ids=block_ids,
                context_lens=context_lens,
                token_offsets=token_offsets,
                q_len=q_len,
            )
        hidden = self.norm(hidden)
        logits = self.lm_head(hidden)
        seq_len = hidden.shape[0] // max(1, q_len)
        return ModelOutput(logits=logits.view(seq_len, q_len, -1))

    def allocate_cache(
        self, num_blocks: int, block_size: int, dtype: torch.dtype, device: torch.device
    ) -> object:
        """Create a :class:`PagedKVCache` sized for this model."""
        from llmopt.cache.paged_cache import PagedKVCache

        return PagedKVCache(
            num_layers=self.config.num_hidden_layers,
            num_blocks=num_blocks,
            block_size=block_size,
            num_kv_heads=self.config.num_key_value_heads,
            head_dim=self.config.head_dim,
            dtype=dtype,
            device=device,
        )

    def save_pretrained(self, output_dir: str) -> None:
        """Write weights and ``config.json`` next to each other."""
        os.makedirs(output_dir, exist_ok=True)
        from llmopt.utils.misc import atomic_write_json

        state = {key: value for key, value in self.state_dict().items()}
        torch.save(state, os.path.join(output_dir, "model.pt"))
        atomic_write_json(os.path.join(output_dir, "config.json"), self.config.to_dict())

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str,
        backend: str = "auto",
        dtype: torch.dtype | None = None,
    ) -> DecoderOnlyModel:
        """Load a model saved by :meth:`save_pretrained`."""
        from llmopt.config import ModelConfig

        config_path = os.path.join(model_dir, "config.json")
        with open(config_path, encoding="utf-8") as handle:
            raw = json.load(handle)
        config = ModelConfig.from_dict(raw)
        model = cls(config, backend=backend, dtype=dtype)
        weights = os.path.join(model_dir, "model.pt")
        if os.path.exists(weights):
            state = torch.load(weights, map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=False)
        return model

    @classmethod
    def from_hf_pretrained(
        cls, model_id: str, dtype: torch.dtype | None = None, device_map: str = "cpu"
    ) -> DecoderOnlyModel:
        """Convert a Hugging Face causal-LM checkpoint into this engine's layout.

        Weight names are remapped (``model.embed_tokens`` -> ``embed_tokens``,
        ``model.layers.N.*`` -> ``layers.N.*``, ``lm_head`` -> ``lm_head``) and
        the config is translated field by field. Requires ``transformers``.
        """
        try:
            from transformers import AutoConfig, AutoModelForCausalLM
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "from_hf_pretrained requires `transformers`; pip install llmopt[hf]"
            ) from exc

        from llmopt.config import ModelConfig

        hf_config = AutoConfig.from_pretrained(model_id)
        config = ModelConfig(
            vocab_size=hf_config.vocab_size,
            hidden_size=hf_config.hidden_size,
            intermediate_size=getattr(hf_config, "intermediate_size", 4 * hf_config.hidden_size),
            num_hidden_layers=hf_config.num_hidden_layers,
            num_attention_heads=hf_config.num_attention_heads,
            num_key_value_heads=getattr(
                hf_config, "num_key_value_heads", hf_config.num_attention_heads
            ),
            max_position_embeddings=getattr(hf_config, "max_position_embeddings", 4096),
            rms_norm_eps=getattr(hf_config, "rms_norm_eps", 1e-6),
            rope_theta=getattr(hf_config, "rope_theta", 10000.0),
            tie_word_embeddings=bool(getattr(hf_config, "tie_word_embeddings", False)),
        )
        hf_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
        model = cls(config, dtype=dtype)
        model.load_state_dict(_convert_hf_state_dict(hf_model.state_dict(), config), strict=False)
        return model.to(device_map)

    def gradient_checkpointing(self, enabled: bool = True) -> None:
        """Trade compute for memory on long prefills."""
        self.gradient_checkpointing_enabled = enabled
        for layer in self.layers:
            layer.gradient_checkpointing = enabled

    def __repr__(self) -> str:
        return (
            f"DecoderOnlyModel(layers={self.config.num_hidden_layers}, "
            f"hidden={self.config.hidden_size}, heads={self.config.num_attention_heads}, "
            f"kv_heads={self.config.num_key_value_heads}, "
            f"params={self.num_parameters() / 1e6:.1f}M)"
        )


def _convert_hf_state_dict(
    state: dict[str, torch.Tensor], config: object
) -> dict[str, torch.Tensor]:
    """Remap Hugging Face parameter names onto this module's names.

    Handles the Llama/Mistral family: fused QKV, gate/up fusion, and the
    ``q_proj``/``k_proj``/``v_proj`` split this engine uses.
    """
    out: dict[str, torch.Tensor] = {}
    for name, tensor in state.items():
        if name.startswith("model."):
            name = name[len("model.") :]
        if name.endswith("embed_positions.weight"):
            continue  # RoPE tables are recomputed, not loaded
        if name.endswith("rotary_emb.inv_freq"):
            continue
        out[name] = tensor
    if "lm_head.weight" not in out and config.tie_word_embeddings:
        out["lm_head.weight"] = out["embed_tokens.weight"]
    return out


class TinyLlamaConfig:
    """Preset configs for the bundled small models used in tests and demos."""

    @staticmethod
    def tiny(vocab_size: int = 512) -> object:
        """Two layers, hidden 64, GQA 4:2. Fast enough for CPU tests."""
        from llmopt.config import ModelConfig

        return ModelConfig(
            vocab_size=vocab_size,
            hidden_size=64,
            intermediate_size=176,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=512,
        )

    @staticmethod
    def small(vocab_size: int = 32000) -> object:
        """Four layers, hidden 512, GQA 8:2. Still runs on a laptop CPU."""
        from llmopt.config import ModelConfig

        return ModelConfig(
            vocab_size=vocab_size,
            hidden_size=512,
            intermediate_size=1376,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=2,
            max_position_embeddings=2048,
        )


def tiny_model(
    vocab_size: int = 512,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
) -> DecoderOnlyModel:
    """Build a randomly initialised 2-layer model (used by tests and benchmarks)."""
    from llmopt.utils.misc import seed_everything

    seed_everything(seed)
    config = TinyLlamaConfig.tiny(vocab_size)
    model = DecoderOnlyModel(config, dtype=dtype)
    # Scale down the init so activations do not blow up in fp32 tests.
    with torch.no_grad():
        for param in model.parameters():
            param.mul_(0.02)
    return model


def estimate_flops_per_token(config: object, quantized: bool = False) -> int:
    """Rough forward FLOPs per token: ``2 * params`` plus attention."""
    params = (
        config.vocab_size * config.hidden_size * 2
        + config.num_hidden_layers
        * (
            4 * config.hidden_size * config.hidden_size
            + 3 * config.hidden_size * config.intermediate_size
        )
    )
    del quantized
    return 2 * int(params)
