"""Shared helpers for the llmopt test suite.

Kept in a module rather than ``conftest`` so test files can import it directly.
An unrelated top-level ``tests`` package exists in some site-packages
installations, so ``from tests.conftest import ...`` is not safe here.
"""

from __future__ import annotations

import torch

from llmopt.engine import LLMEngine

MAX_MODEL_LEN = 512


def build_engine(**overrides) -> LLMEngine:
    """Create a small deterministic engine for tests."""
    kwargs: dict = {
        "vocab_size": 256,
        "max_model_len": MAX_MODEL_LEN,
        "num_gpu_blocks": 256,
        "block_size": 16,
    }
    kwargs.update(overrides)
    return LLMEngine.from_preset("tiny", **kwargs)


def greedy_reference(model, prompt: list[int], num_tokens: int) -> list[int]:
    """Run ``model`` with no KV cache, one full forward pass per token.

    This is the ground truth for the paged engine: it shares nothing with the
    block pool, block table, scheduler, or prefix cache, so any divergence is a
    bug in the serving machinery rather than in the model.
    """
    sequence = list(prompt)
    with torch.no_grad():
        for _ in range(num_tokens):
            out = model(
                torch.tensor([sequence]),
                torch.arange(len(sequence)),
                cache=None,
                q_len=len(sequence),
            )
            sequence.append(int(out.logits[0, -1].argmax()))
    return sequence[len(prompt) :]
