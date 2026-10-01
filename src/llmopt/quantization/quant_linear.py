"""Quantized linear layers and model-wide quantization helpers."""

from __future__ import annotations

import math

import torch
from torch import nn

from llmopt.quantization.quantizers import QuantizedTensor, dequantize
from llmopt.utils.logging import get_logger

__all__ = ["QuantLinear", "QuantLinearConfig", "quantize_model_", "replace_linear_modules"]

logger = get_logger("quant.nn")


class QuantLinearConfig:
    """Configuration for a :class:`QuantLinear` layer."""

    __slots__ = ("bits", "group_size", "symmetric", "compute_dtype")

    def __init__(
        self,
        bits: int = 4,
        group_size: int = 128,
        symmetric: bool = True,
        compute_dtype: torch.dtype = torch.float16,
    ) -> None:
        self.bits = bits
        self.group_size = group_size
        self.symmetric = symmetric
        self.compute_dtype = compute_dtype


class QuantLinear(nn.Module):
    """A linear layer whose weight lives in packed integer form.

    The weight is stored as a buffer rather than a ``Parameter``, so it is
    excluded from optimizer state and from gradient computation. Each forward
    dequantizes into a small scratch buffer, which keeps accuracy identical to
    the original layer while the resident footprint stays at ``bits`` per weight.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: QuantLinearConfig | None = None,
        bias: bool = True,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config or QuantLinearConfig()
        self.qweight = None
        self.register_buffer("qzeros", torch.zeros(0), persistent=True)
        self.bias = (
            nn.Parameter(torch.zeros(out_features, device=device))
            if bias
            else self.register_buffer("bias", None)  # type: ignore[assignment]
        )
        self._cached_weight: torch.Tensor | None = None

    @classmethod
    def from_float(
        cls,
        linear: nn.Linear,
        quantized: QuantizedTensor,
        compute_dtype: torch.dtype | None = None,
    ) -> QuantLinear:
        """Build a quantized layer from an existing ``nn.Linear`` and its packed weight."""
        config = QuantLinearConfig(
            bits=quantized.bits,
            group_size=quantized.group_size,
            symmetric=quantized.symmetric,
            compute_dtype=compute_dtype or linear.weight.dtype,
        )
        module = cls(linear.in_features, linear.out_features, config, bias=linear.bias is not None)
        module.qweight = quantized.qweight
        module.register_buffer("scales", quantized.scales)
        module.register_buffer("qzeros", quantized.zeros)
        if linear.bias is not None:
            module.bias.data = linear.bias.data.clone()  # type: ignore[union-attr]
        module.invalidate()
        return module

    def invalidate(self) -> None:
        """Drop the dequantized weight so the next forward recomputes it."""
        self._cached_weight = None

    def _dequantized_weight(self) -> torch.Tensor:
        if self._cached_weight is None:
            if self.qweight is None or not hasattr(self, "scales"):
                raise RuntimeError("QuantLinear has no quantized weight; call from_float()")
            tensor = QuantizedTensor(
                qweight=self.qweight,
                scales=self.scales,  # type: ignore[arg-type]
                zeros=self.qzeros,
                bits=self.config.bits,
                group_size=self.config.group_size,
                symmetric=self.config.symmetric,
                shape=(self.out_features, self.in_features),
            )
            weight = dequantize(tensor, self.config.compute_dtype)
            self._cached_weight = weight.to(self.qweight.device)  # type: ignore[union-attr]
        return self._cached_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self._dequantized_weight()
        bias = self.bias
        return torch.nn.functional.linear(x, weight, bias)  # type: ignore[arg-type]

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bits={self.config.bits}, bias={self.bias is not None}"
        )


def replace_linear_modules(
    model: nn.Module,
    replacements: dict[nn.Module, nn.Module],
) -> nn.Module:
    """Swap modules in ``model`` for the entries of ``replacements`` (in place)."""
    if not replacements:
        return model
    parents: dict[nn.Module, tuple[nn.Module, str]] = {}
    for parent in model.modules():
        for name, child in parent.named_children():
            if child in replacements:
                parents[child] = (parent, name)
    for child, (parent, name) in parents.items():
        setattr(parent, name, replacements[child])
    return model


def quantize_model_(
    model: nn.Module,
    *,
    bits: int = 4,
    group_size: int = 128,
    symmetric: bool = True,
    strategy: str = "rtn",
    calibration_batches: list[torch.Tensor] | None = None,
    ignore_layers: tuple[str, ...] = ("lm_head", "embed_tokens"),
    inplace: bool = True,
) -> nn.Module:
    """Quantize all linear layers of ``model`` in place and return it.

    Layers matching ``ignore_layers`` and layers whose ``in_features`` is not a
    multiple of ``group_size`` are left in floating point.
    """
    from llmopt.quantization.calibration import apply_quantization
    from llmopt.quantization.quantizers import build_quantizer

    named: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(token in name for token in ignore_layers):
            continue
        remainder = module.in_features % group_size
        if remainder != 0:
            # Fall back to a group size that divides in_features exactly.
            divisor = math.gcd(module.in_features, group_size)
            logger.debug(
                "%s: in_features=%d is not a multiple of group_size=%d, using %d instead",
                name,
                module.in_features,
                group_size,
                divisor,
            )
        named.append((name, module))

    hessians: dict[nn.Module, torch.Tensor] = {}
    if strategy == "gptq" and calibration_batches:
        from llmopt.quantization.calibration import HessianCollector

        for _, module in named:
            with HessianCollector([module], device=module.weight.device) as collector:
                for batch in calibration_batches:
                    try:
                        model(batch.to(module.weight.device))
                    except Exception:  # noqa: BLE001 - calibration is best effort
                        break
                hessians.update(collector.results())

    replacements: dict[nn.Module, nn.Module] = {}
    for _, module in named:
        effective_group = math.gcd(module.in_features, group_size) or group_size
        weight = module.weight.detach().to(torch.float32)
        quantizer = build_quantizer(
            strategy,
            bits=bits,
            group_size=effective_group,
            symmetric=symmetric,
        )
        quantized = quantizer.quantize(weight, hessians.get(module))
        replacements[module] = QuantLinear.from_float(
            module, quantized, compute_dtype=module.weight.dtype
        )
        module.weight.data = quantized.dequantize().to(module.weight.dtype)

    replace_linear_modules(model, replacements)
    return model
