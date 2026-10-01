"""Tokenizers: a dependency-free byte-level fallback plus a HF adapter."""

from __future__ import annotations

import abc
from collections.abc import Sequence

__all__ = ["BaseTokenizer", "ByteTokenizer", "build_tokenizer", "load_tokenizer"]


class BaseTokenizer(abc.ABC):
    """Minimal tokenizer interface the engine relies on."""

    eos_token_id: int
    pad_token_id: int

    @abc.abstractmethod
    def encode(self, text: str) -> list[int]:
        """Encode ``text`` into token ids."""

    @abc.abstractmethod
    def decode(self, token_ids: Sequence[int]) -> str:
        """Decode token ids back into text."""


class ByteTokenizer(BaseTokenizer):
    """UTF-8 byte-level tokenizer.

    Not subword-optimal, but it needs no vocabulary file and round-trips
    arbitrary text exactly, which makes it the right default for tests, the
    benchmark harness, and smoke-testing the server.
    """

    def __init__(self, special_tokens: Sequence[str] = ("<pad>", "<eos>")) -> None:
        specials = list(special_tokens)
        self.special_tokens = {token: 256 + i for i, token in enumerate(specials)}
        self.inv_specials = {v: k for k, v in self.special_tokens.items()}
        self.vocab_size = 256 + len(specials)
        self.eos_token_id = self.special_tokens.get("<eos>", 0)
        self.pad_token_id = self.special_tokens.get("<pad>", 0)

    def encode(self, text: str) -> list[int]:
        """Encode text, mapping known special tokens to their ids."""
        ids: list[int] = []
        specials = sorted(self.special_tokens, key=len, reverse=True)
        for chunk in _split_on_specials(text, specials):
            if chunk in self.special_tokens:
                ids.append(self.special_tokens[chunk])
            else:
                ids.extend(chunk.encode("utf-8"))
        return ids

    def decode(self, token_ids: Sequence[int]) -> str:
        """Decode ids, replacing special tokens with their text form."""
        parts: list[str] = []
        pending = bytearray()
        for token in token_ids:
            if token in self.inv_specials:
                if pending:
                    parts.append(pending.decode("utf-8", errors="replace"))
                    pending = bytearray()
                parts.append(self.inv_specials[token])
            elif 0 <= token < 256:
                pending.append(token)
        if pending:
            parts.append(pending.decode("utf-8", errors="replace"))
        return "".join(parts)

    def __repr__(self) -> str:
        return f"ByteTokenizer(vocab_size={self.vocab_size})"


def _split_on_specials(text: str, specials: Sequence[str]) -> list[str]:
    if not specials:
        return [text]
    import re

    pattern = "(" + "|".join(re.escape(token) for token in specials) + ")"
    return [chunk for chunk in re.split(pattern, text) if chunk]


class HFTokenizer(BaseTokenizer):
    """Thin adapter over a Hugging Face fast tokenizer."""

    def __init__(self, tokenizer: object) -> None:
        self._tokenizer = tokenizer
        self.eos_token_id = int(tokenizer.eos_token_id or 0)
        self.pad_token_id = int(getattr(tokenizer, "pad_token_id", None) or self.eos_token_id)

    def encode(self, text: str) -> list[int]:
        return list(self._tokenizer.encode(text, add_special_tokens=False))

    def decode(self, token_ids: Sequence[int]) -> str:
        return str(self._tokenizer.decode(list(token_ids), skip_special_tokens=True))

    @property
    def vocab_size(self) -> int:
        return int(self._tokenizer.vocab_size)


def build_tokenizer(path_or_name: str | None = None) -> BaseTokenizer:
    """Return the best tokenizer available for ``path_or_name``.

    Falls back to :class:`ByteTokenizer` when ``transformers`` is not installed
    or the name cannot be resolved, so callers never need a try/except.
    """
    if path_or_name is None:
        return ByteTokenizer()
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return ByteTokenizer()
    try:
        return HFTokenizer(AutoTokenizer.from_pretrained(path_or_name))
    except Exception:  # noqa: BLE001 - offline or unknown model id
        return ByteTokenizer()


def load_tokenizer(path_or_name: str | None = None) -> BaseTokenizer:
    """Alias of :func:`build_tokenizer` kept for readability at call sites."""
    return build_tokenizer(path_or_name)
