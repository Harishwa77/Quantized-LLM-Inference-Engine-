"""Calibration data collection and Hessian accumulation for GPTQ.

GPTQ needs, for every linear layer, the second-moment matrix of its inputs:

    H = 2/N * sum_i x_i x_i^T

We gather those statistics with forward hooks on the target modules rather than
materialising activations, so memory stays bounded by the batch size.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

import torch
from torch import nn

from llmopt.quantization.quantizers import GPTQQuantizer, QuantizedTensor, build_quantizer
from llmopt.utils.logging import get_logger

__all__ = ["CalibrationHarness", "CalibrationResult", "HessianCollector", "apply_quantization"]

logger = get_logger("quant.calibration")


class HessianCollector:
    """Accumulates ``X^T X`` per module via forward hooks."""

    def __init__(self, layers: Sequence[nn.Module], device: torch.device | str = "cpu") -> None:
        self.layers = list(layers)
        self.device = torch.device(device)
        self._states: dict[int, dict[str, object]] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self.count = 0

    def _state(self, module: nn.Module) -> dict[str, object]:
        key = id(module)
        if key not in self._states:
            weight = module.weight
            self._states[key] = {
                "hessian": torch.zeros(
                    weight.shape[1], weight.shape[1], dtype=torch.float32, device=self.device
                ),
                "samples": 0,
            }
        return self._states[key]

    def _hook(self, module: nn.Module, inputs: tuple[torch.Tensor, ...], _output: object) -> None:
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            return
        x = inputs[0].detach()
        x = x.reshape(-1, x.shape[-1]).to(device=self.device, dtype=torch.float32)
        state = self._state(module)
        hessian = state["hessian"]
        assert isinstance(hessian, torch.Tensor)
        hessian.addmm_(x.T, x)
        state["samples"] = int(state["samples"]) + x.shape[0]  # type: ignore[arg-type]
        self.count += 1

    def __enter__(self) -> HessianCollector:
        for module in self.layers:
            self._handles.append(module.register_forward_hook(self._hook))
        return self

    def __exit__(self, *exc: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def hessian_for(self, module: nn.Module) -> torch.Tensor | None:
        state = self._states.get(id(module))
        if state is None or not state["samples"]:
            return None
        hessian = state["hessian"]
        assert isinstance(hessian, torch.Tensor)
        samples = int(state["samples"])  # type: ignore[arg-type]
        return (2.0 / samples) * hessian

    def results(self) -> dict[nn.Module, torch.Tensor]:
        return {
            module: hessian
            for module in self.layers
            if (hessian := self.hessian_for(module)) is not None
        }


@dataclass(slots=True)
class CalibrationResult:
    """Outcome of quantizing a model: per-module packed tensors plus accuracy."""

    quantizers: dict[str, QuantizedTensor] = field(default_factory=dict)
    mean_relative_error: float = 0.0
    per_layer_error: dict[str, float] = field(default_factory=dict)
    baseline_bits: int = 0
    quant_bits: int = 0

    @property
    def compression_ratio(self) -> float:
        if not self.quantizers:
            return 1.0
        dense = sum(q.dense_nbytes() for q in self.quantizers.values())
        packed = sum(q.nbytes() for q in self.quantizers.values())
        return dense / max(1, packed)

    def summary(self) -> str:
        return (
            f"quantized {len(self.quantizers)} layers to {self.quant_bits}-bit, "
            f"mean relative error {self.mean_relative_error:.5f}, "
            f"{self.compression_ratio:.2f}x compression"
        )


def _module_targets(model: nn.Module, ignore: Sequence[str]) -> list[nn.Linear]:
    targets: list[nn.Linear] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(token in name for token in ignore):
            continue
        if module.weight.shape[1] % 2 != 0:
            logger.debug("skipping %s: in_features=%d is odd", name, module.weight.shape[1])
            continue
        targets.append(module)
    return targets


@torch.no_grad()
def _relative_error(original: torch.Tensor, approx: torch.Tensor) -> float:
    denom = torch.linalg.vector_norm(original.float())
    if float(denom) == 0.0:
        return 0.0
    return float(torch.linalg.vector_norm((original - approx).float()) / denom)


@torch.no_grad()
def apply_quantization(
    model: nn.Module,
    *,
    strategy: str = "gptq",
    bits: int = 4,
    group_size: int = 128,
    symmetric: bool = True,
    damping: float = 0.01,
    act_order: bool = True,
    ignore_layers: Sequence[str] = ("lm_head", "embed_tokens"),
    calibration_batches: Sequence[torch.Tensor] | None = None,
    in_place: bool = False,
) -> CalibrationResult:
    """Quantize every eligible ``nn.Linear`` in ``model``.

    Args:
        model: Module to quantize. Modified in place unless ``in_place=False``.
        strategy: ``"gptq"`` or ``"rtn"``.
        bits: Target bit width.
        group_size: Input channels per scale/zero.
        symmetric: Use a symmetric integer grid.
        damping: Hessian damping factor for the Cholesky solve.
        act_order: Sort channels by Hessian diagonal before quantizing.
        ignore_layers: Substrings of module names to leave untouched.
        calibration_batches: Token id batches used to build Hessians. GPTQ
            without them silently degrades to round-to-nearest.
        in_place: When ``False`` a quantized copy is returned in the result.

    Returns:
        A :class:`CalibrationResult` describing the applied quantization.
    """
    targets = _module_targets(model, ignore_layers)
    if not targets:
        logger.warning("no quantizable linear layers found")
        return CalibrationResult(quant_bits=bits)

    result = CalibrationResult(quant_bits=bits)

    hessians: dict[nn.Module, torch.Tensor] = {}
    if strategy == "gptq" and calibration_batches:
        for module in targets:
            modules = [module]
            with HessianCollector(modules, device=module.weight.device) as collector:
                for batch in calibration_batches:
                    inputs = batch.to(module.weight.device)
                    if inputs.dim() == 1:
                        inputs = inputs.unsqueeze(0)
                    with contextlib.suppress(Exception):
                        model(inputs)
                hessians.update(collector.results())
        logger.info("collected Hessians for %d/%d layers", len(hessians), len(targets))

    for name, module in model.named_modules():
        if module not in targets:
            continue
        original = module.weight.detach().clone()
        quantizer = build_quantizer(
            strategy,
            bits=bits,
            group_size=group_size,
            symmetric=symmetric,
            damping=damping,
            act_order=act_order,
        )
        if not isinstance(quantizer, GPTQQuantizer):
            quantizer = build_quantizer(
                "rtn", bits=bits, group_size=group_size, symmetric=symmetric
            )
        quantized = quantizer.quantize(original, hessians.get(module))
        approx = quantized.dequantize().to(original.dtype)
        error = _relative_error(original, approx)
        result.quantizers[name] = quantized
        result.per_layer_error[name] = error
        result.baseline_bits += original.numel() * original.element_size() * 8

        module.weight.data = approx.to(module.weight.device)
        module.extra_repr = (  # type: ignore[attr-defined]
            f"quantized={quantizer.name}-{bits}bit, group_size={group_size}, err={error:.4f}"
        )

    errors = list(result.per_layer_error.values())
    result.mean_relative_error = sum(errors) / len(errors) if errors else 0.0
    logger.info(result.summary())
    return result


class CalibrationHarness:
    """Generates token batches for calibration from text or synthetic data."""

    def __init__(self, num_samples: int = 128, seq_length: int = 512, seed: int = 0) -> None:
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.seed = seed

    def synthetic(self, vocab_size: int) -> list[torch.Tensor]:
        """Structured random batches: lower entropy than pure noise, more realistic."""
        generator = torch.Generator().manual_seed(self.seed)
        batches: list[torch.Tensor] = []
        for _ in range(self.num_samples):
            batch = torch.randint(
                0, vocab_size, (1, self.seq_length), generator=generator, dtype=torch.long
            )
            batches.append(batch)
        return batches

    def from_tokenizer(self, tokenizer: object, texts: Sequence[str]) -> list[torch.Tensor]:
        """Encode ``texts`` into calibration batches, padding to a fixed length."""
        encode = getattr(tokenizer, "encode", None)
        if encode is None:
            raise TypeError("tokenizer must expose an encode() method")
        batches: list[torch.Tensor] = []
        for text in texts:
            ids = encode(text)
            ids = list(ids)[: self.seq_length]
            if not ids:
                continue
            pad = self.seq_length - len(ids)
            row = ids + [0] * pad
            batches.append(torch.tensor([row], dtype=torch.long))
        return batches

    def __iter__(self) -> Iterator[torch.Tensor]:  # pragma: no cover - convenience
        raise NotImplementedError("call synthetic() or from_tokenizer() explicitly")
