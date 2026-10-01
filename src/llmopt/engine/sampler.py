"""Sampling: temperature, top-k, top-p, penalties, and seeded RNG."""

from __future__ import annotations

import torch

from llmopt.engine.request import SamplingParams

__all__ = ["Sampler", "apply_penalties"]


def apply_penalties(
    logits: torch.Tensor,
    generated: list[list[int]],
    params: SamplingParams,
) -> torch.Tensor:
    """Apply repetition, presence and frequency penalties in place.

    Args:
        logits: ``[batch, vocab]`` raw logits.
        generated: ``[batch]`` lists of token ids already produced.
        params: Decoding parameters.

    Returns:
        The penalised logits (the same tensor is modified).
    """
    if params.repetition_penalty == 1.0 and not generated:
        if not params.logit_bias:
            return logits
    for row, tokens in enumerate(generated):
        if not tokens:
            continue
        indices = torch.tensor(sorted(set(tokens)), dtype=torch.long, device=logits.device)
        if params.repetition_penalty != 1.0:
            values = logits[row, indices]
            # Positive logits are divided, negative ones multiplied, so the
            # penalised direction is always downwards.
            logits[row, indices] = torch.where(
                values > 0, values / params.repetition_penalty, values * params.repetition_penalty
            )
        if params.presence_penalty:
            logits[row, indices] -= params.presence_penalty
        if params.frequency_penalty:
            counts = torch.bincount(indices, minlength=logits.shape[1]).to(logits.dtype)
            logits[row] -= params.frequency_penalty * counts
    if params.logit_bias:
        for token, bias in params.logit_bias.items():
            if 0 <= token < logits.shape[1]:
                logits[:, token] += bias
    return logits


class Sampler:
    """Stateless sampler; the generator owns the RNG state.

    Args:
        device: Device the sampling ops run on.
        seed: Default seed for the internal generator.
    """

    def __init__(self, device: torch.device | str = "cpu", seed: int = 0) -> None:
        self.device = torch.device(device)
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed)

    def _seeded_generator(self, seed: int) -> torch.Generator:
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        return generator

    def forward(
        self,
        logits: torch.Tensor,
        params: SamplingParams,
        generated: list[list[int]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample token ids and return their log-probabilities.

        Args:
            logits: ``[batch, vocab]`` logits for the current step.
            params: Decoding parameters.
            generated: Token ids produced so far, for the penalty terms.

        Returns:
            ``(token_ids, logprobs)`` each shaped ``[batch]``.
        """
        logits = logits.to(torch.float32)
        if generated:
            logits = apply_penalties(logits.clone(), generated, params)

        generator = (
            self._seeded_generator(params.seed) if params.seed is not None else self.generator
        )
        if params.greedy:
            token_ids = torch.argmax(logits, dim=-1)
        else:
            scaled = logits / max(1e-6, params.temperature)
            scaled = _top_k_top_p_filter(scaled, params.top_k, params.top_p)
            probs = torch.softmax(scaled, dim=-1)
            token_ids = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)

        logprobs = _gather_logprobs(logits, token_ids)
        return token_ids, logprobs


def _top_k_top_p_filter(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    """Mask everything outside the top-k / nucleus set to ``-inf``."""
    filtered = logits
    if top_k and top_k > 0:
        k = min(top_k, filtered.shape[-1])
        threshold = torch.topk(filtered, k, dim=-1).values[..., -1:]
        filtered = filtered.masked_fill(filtered < threshold, float("-inf"))
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = probs.cumsum(dim=-1)
        # Keep the smallest set whose cumulative mass reaches top_p.
        remove = cumulative - probs > top_p
        remove[..., 0] = False
        mask = torch.zeros_like(remove).scatter(1, sorted_indices, remove)
        filtered = filtered.masked_fill(mask, float("-inf"))
    return filtered


def _gather_logprobs(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    logprobs = torch.log_softmax(logits, dim=-1)
    return logprobs.gather(1, token_ids.unsqueeze(1)).squeeze(1)
