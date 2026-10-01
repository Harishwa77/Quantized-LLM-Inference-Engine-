"""Weight quantizers: round-to-nearest and GPTQ error-compensated variants.

The quantizers operate on 2D weight matrices shaped ``[out_features, in_features]``
and return packed integer codes plus per-group scale/zero-point metadata that is
small enough to keep resident in GPU memory.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

import torch

__all__ = [
    "QuantizedTensor",
    "Quantizer",
    "build_quantizer",
    "choose_qparams",
    "pack_int4",
    "unpack_int4",
]


@dataclass(frozen=True, slots=True)
class QuantizedTensor:
    """Packed weights plus the metadata required to reconstruct them.

    Attributes:
            qweight: Integer codes. Shape ``[out_features, in_features // pack_factor]``
            for sub-8-bit widths, otherwise ``[out_features, in_features]``.
        scales: Per-group scales, shaped ``[out_features, num_groups]``.
        zeros: Per-group zero points, same shape as ``scales``. Zero for
            symmetric schemes.
        bits: Bit width of the codes.
        group_size: Number of input channels sharing one scale/zero.
        symmetric: Whether the grid is symmetric around zero.
        shape: Original ``[out_features, in_features]`` weight shape.
    """

    qweight: torch.Tensor
    scales: torch.Tensor
    zeros: torch.Tensor
    bits: int
    group_size: int
    symmetric: bool
    shape: tuple[int, int]

    @property
    def code_offset(self) -> int:
        """Bias added to signed codes so they fit in an unsigned container.

        Symmetric grids span negative values, which cannot be stored in
        ``uint8`` packing. Shifting by ``2**(bits-1)`` keeps them unsigned and is
        undone exactly on dequantize.
        """
        return (1 << (self.bits - 1)) if self.symmetric else 0

    @property
    def pack_factor(self) -> int:
        return 32 // self.bits if self.bits < 8 else 1

    def dequantize(self) -> torch.Tensor:
        """Reconstruct a dense float weight from the packed representation."""
        return dequantize(self)

    def nbytes(self) -> int:
        """Total bytes of the packed representation, scales and zeros included."""
        return sum(
            t.numel() * t.element_size()
            for t in (self.qweight, self.scales, self.zeros)
            if t is not None
        )

    def dense_nbytes(self) -> int:
        """Bytes the equivalent fp16 tensor would occupy."""
        return self.shape[0] * self.shape[1] * 2

    def compression_ratio(self) -> float:
        return self.dense_nbytes() / max(1, self.nbytes())


def pack_int4(codes: torch.Tensor) -> torch.Tensor:
    """Pack two 4-bit codes per ``uint8`` along the last dimension."""
    if codes.shape[-1] % 2 != 0:
        raise ValueError("4-bit packing requires an even number of columns")
    codes = codes.to(torch.uint8)
    packed = codes[..., 0::2] | (codes[..., 1::2] << 4)
    return packed.contiguous()


def unpack_int4(packed: torch.Tensor, columns: int) -> torch.Tensor:
    """Inverse of :func:`pack_int4`."""
    low = packed & 0x0F
    high = packed >> 4
    out = torch.stack((low, high), dim=-1).reshape(*packed.shape[:-1], -1)
    return out[..., :columns].contiguous()


def choose_qparams(
    weight: torch.Tensor,
    bits: int,
    group_size: int,
    symmetric: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-group scales and zero points via min/max range search.

    Args:
        weight: ``[out_features, in_features]`` float tensor.
        bits: Bit width of the integer grid.
        group_size: Columns per group. Values ``<= 0`` mean one group per row.
        symmetric: Use ``[-qmax, qmax]`` instead of ``[0, qmax]``.

    Returns:
        ``(scales, zeros)`` each shaped ``[out_features, num_groups]``, so
        group metadata lines up with the ``[out, in // group, group]`` view
        used by :func:`quantize_to_codes` without a transpose.
    """
    if weight.dim() != 2:
        raise ValueError(f"expected a 2D weight, got shape {tuple(weight.shape)}")
    out_features, in_features = weight.shape
    gsize = in_features if group_size is None or group_size <= 0 else group_size
    if in_features % gsize != 0:
        raise ValueError(f"in_features={in_features} not divisible by group_size={gsize}")

    grouped = weight.reshape(out_features, in_features // gsize, gsize)
    # [out_features, num_groups]
    wmin = grouped.amin(dim=-1).to(torch.float32)
    wmax = grouped.amax(dim=-1).to(torch.float32)

    if symmetric:
        amax = torch.maximum(wmin.abs(), wmax.abs())
        amax = torch.clamp(amax, min=1e-8)
        scales = amax / (2 ** (bits - 1) - 1)
        zeros = torch.zeros_like(scales)
    else:
        wmin = torch.clamp(wmin, max=0.0)
        wmax = torch.clamp(wmax, min=0.0)
        scales = torch.clamp((wmax - wmin) / (2**bits - 1), min=1e-8)
        zeros = torch.round(-wmin / scales)

    return scales.contiguous(), zeros.contiguous()


def expand_group_metadata(
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int,
    in_features: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Broadcast per-group metadata to a per-column table.

    Args:
        scales: ``[out_features, num_groups]``.
        zeros: Same shape as ``scales``.
        group_size: Input channels per group.
        in_features: Total input channels to cover.

    Returns:
        ``(scale_per_col, zero_per_col)`` each ``[out_features, in_features]``.
    """
    per_col = scales.repeat_interleave(group_size, dim=1)
    zero_col = zeros.repeat_interleave(group_size, dim=1)
    return per_col[:, :in_features], zero_col[:, :in_features]


def quantize_to_codes(
    weight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    bits: int,
    group_size: int,
    symmetric: bool = True,
) -> torch.Tensor:
    """Apply scale/zero to ``weight`` and round onto the integer grid.

    Symmetric schemes use a signed range centred on zero; asymmetric schemes use
    ``[0, 2**bits - 1]`` with the group's zero point subtracted on dequantize.
    Signed codes narrower than a byte are biased by ``2**(bits - 1)`` so they
    survive ``uint8`` packing; the bias is undone on dequantize.

    Returns:
        ``int8`` for symmetric 8-bit, otherwise ``uint8``.
    """
    out_features, in_features = weight.shape
    gsize = in_features if group_size <= 0 else group_size
    w = weight.to(torch.float32).reshape(out_features, in_features // gsize, gsize)
    s = scales.unsqueeze(-1)
    z = zeros.unsqueeze(-1)
    if symmetric:
        half = 2 ** (bits - 1) - 1
        codes = torch.clamp(torch.round(w / s), -half - 1, half)
    else:
        codes = torch.clamp(torch.round(w / s) + z, 0, 2**bits - 1)
    if symmetric and bits == 8:
        codes = codes.to(torch.int8)
    else:
        if symmetric:
            codes = codes + (1 << (bits - 1))
        codes = codes.to(torch.uint8)
    return codes.reshape(out_features, in_features).contiguous()


def dequantize_codes(
    codes: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    bits: int,
    group_size: int,
    dtype: torch.dtype = torch.float32,
    symmetric: bool = True,
) -> torch.Tensor:
    """Reconstruct float weights from integer codes and group metadata."""
    out_features, in_features = codes.shape
    gsize = in_features if group_size <= 0 else group_size
    q = codes.reshape(out_features, in_features // gsize, gsize).to(torch.float32)
    s = scales.unsqueeze(-1)
    z = zeros.unsqueeze(-1)
    if symmetric:
        # Undo the unsigned bias applied by quantize_to_codes. Sub-byte grids are
        # stored biased in uint8; the 8-bit grid is stored signed in int8 and
        # needs no shift.
        if bits < 8:
            q = q - (1 << (bits - 1))
        w = q * s
    else:
        w = (q - z) * s
    return w.reshape(out_features, in_features).to(dtype)


def dequantize(qt: QuantizedTensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Expand a :class:`QuantizedTensor` back to a dense weight matrix."""
    if qt.bits == 8:
        codes = qt.qweight.to(torch.float32)
    else:
        codes = unpack_int4(qt.qweight, qt.shape[1]).to(torch.float32)
    return dequantize_codes(
        codes, qt.scales, qt.zeros, qt.bits, qt.group_size, dtype, qt.symmetric
    )


class Quantizer(abc.ABC):
    """Base class for weight quantizers.

    Subclasses only need to implement :meth:`quantize`; everything else
    (packing, metadata bookkeeping, stats) is handled here.
    """

    name: str = "base"

    def __init__(self, bits: int, group_size: int = 128, symmetric: bool = True) -> None:
        if bits not in (2, 4, 8):
            raise ValueError(f"unsupported bit width: {bits}")
        if group_size <= 0 or group_size % 2 != 0:
            raise ValueError("group_size must be a positive even number")
        self.bits = bits
        self.group_size = group_size
        self.symmetric = symmetric

    @abc.abstractmethod
    def quantize(
        self, weight: torch.Tensor, hessian: torch.Tensor | None = None
    ) -> QuantizedTensor:
        """Quantize ``weight`` ``[out, in]``; ``hessian`` is ``[in, in]`` if available."""

    def _finalize(
        self,
        codes: torch.Tensor,
        scales: torch.Tensor,
        zeros: torch.Tensor,
        shape: tuple[int, int],
    ) -> QuantizedTensor:
        qweight = codes if self.bits == 8 else pack_int4(codes)
        return QuantizedTensor(
            qweight=qweight,
            scales=scales,
            zeros=zeros,
            bits=self.bits,
            group_size=self.group_size,
            symmetric=self.symmetric,
            shape=shape,
        )


class RoundToNearestQuantizer(Quantizer):
    """Classic asymmetric/symmetric min-max round-to-nearest.

    No calibration data needed, which makes it the right choice for activation
    quantizers and for fast smoke tests.
    """

    name = "rtn"

    def quantize(
        self, weight: torch.Tensor, hessian: torch.Tensor | None = None
    ) -> QuantizedTensor:
        del hessian  # RTN ignores second-order information.
        weight = weight.detach()
        if weight.dim() != 2:
            raise ValueError(f"expected a 2D weight, got {tuple(weight.shape)}")
        scales, zeros = choose_qparams(weight, self.bits, self.group_size, self.symmetric)
        codes = quantize_to_codes(
            weight, scales, zeros, self.bits, self.group_size, self.symmetric
        )
        return self._finalize(codes, scales, zeros, tuple(weight.shape))


class GPTQQuantizer(Quantizer):
    """GPTQ: error-compensated rounding driven by the input Hessian.

    Columns are quantized left to right in blocks. After each column the
    rounding residual is propagated to the not-yet-quantized columns using the
    upper Cholesky factor of the inverse Hessian, which cancels part of the
    error that the current column introduced.
    """

    name = "gptq"

    def __init__(
        self,
        bits: int = 4,
        group_size: int = 128,
        symmetric: bool = True,
        damping: float = 0.01,
        act_order: bool = True,
        block_size: int | None = None,
    ) -> None:
        super().__init__(bits, group_size, symmetric)
        self.damping = damping
        self.act_order = act_order
        # Blocks are aligned to group boundaries so one block refreshes one group grid.
        self.block_size = block_size or group_size

    @torch.no_grad()
    def quantize(
        self, weight: torch.Tensor, hessian: torch.Tensor | None = None
    ) -> QuantizedTensor:
        """Quantize ``weight`` column-wise, cancelling rounding error as we go.

        Args:
            weight: ``[out_features, in_features]`` float matrix.
            hessian: ``[in_features, in_features]`` second-moment matrix of the
                layer inputs, i.e. ``2/N * X^T X``. When ``None`` the method
                degrades to round-to-nearest.

        Returns:
            The packed :class:`QuantizedTensor` in the original column order.
        """
        weight = weight.detach().to(torch.float32)
        if weight.dim() != 2:
            raise ValueError(f"expected a 2D weight, got {tuple(weight.shape)}")
        out_features, in_features = weight.shape

        if hessian is None:
            fallback = RoundToNearestQuantizer(self.bits, self.group_size, self.symmetric)
            return fallback.quantize(weight)

        h = self._prepare_hessian(hessian, in_features, weight.device)
        scales, zeros = choose_qparams(weight, self.bits, self.group_size, self.symmetric)

        # Group metadata is defined on the original column order, so expand it to
        # a per-column table. The quantisation order can then be permuted freely
        # (activation ordering) without invalidating the group boundaries.
        scale_per_col, zero_per_col = expand_group_metadata(
            scales, zeros, self.group_size, in_features
        )

        perm = self._activation_order(h, in_features) if self.act_order else None
        w_work = weight[:, perm] if perm is not None else weight.clone()
        h_work = h[perm][:, perm] if perm is not None else h
        scale_work = scale_per_col[:, perm] if perm is not None else scale_per_col
        zero_work = zero_per_col[:, perm] if perm is not None else zero_per_col

        codes = torch.empty(out_features, in_features, dtype=torch.float32, device=weight.device)
        for i1 in range(0, in_features, self.block_size):
            i2 = min(i1 + self.block_size, in_features)
            width = i2 - i1
            block = w_work[:, i1:i2].clone()
            block_scale = scale_work[:, i1:i2]
            block_zero = zero_work[:, i1:i2]
            block_codes = torch.zeros(out_features, width, dtype=torch.float32, device=weight.device)
            errors = torch.zeros_like(block)
            h_block = h_work[i1:i2, i1:i2]

            for j in range(width):
                diagonal = h_block[j, j].clamp(min=1e-8)
                column = block[:, j]
                scale_j = block_scale[:, j]
                zero_j = block_zero[:, j]
                qj = _quantize_vector(column, scale_j, zero_j, self.bits, self.symmetric)
                block_codes[:, j] = qj
                errors[:, j] = (
                    column - _dequantize_vector(qj, scale_j, zero_j, self.symmetric)
                ) / diagonal
                if j + 1 < width:
                    block[:, j + 1 :] -= errors[:, j : j + 1] * h_block[j, j + 1 :].unsqueeze(0)

            if i2 < in_features:
                w_work[:, i2:] -= errors @ h_work[i1:i2, i2:]
            codes[:, i1:i2] = block_codes

        if perm is not None:
            # Undo the activation ordering so codes line up with the original columns.
            inverse = torch.argsort(perm)
            codes = codes[:, inverse]

        if self.symmetric and self.bits == 8:
            codes = codes.to(torch.int8)
        else:
            if self.symmetric:
                codes = codes + (1 << (self.bits - 1))
            codes = codes.to(torch.uint8)
        return QuantizedTensor(
            qweight=codes if self.bits == 8 else pack_int4(codes),
            scales=scales,
            zeros=zeros,
            bits=self.bits,
            group_size=self.group_size,
            symmetric=self.symmetric,
            shape=(out_features, in_features),
        )

    def _prepare_hessian(
        self, hessian: torch.Tensor, in_features: int, device: torch.device
    ) -> torch.Tensor:
        h = hessian.detach().to(device=device, dtype=torch.float32)
        h = 0.5 * (h + h.T)
        dead = torch.diag(h) == 0
        h[dead, dead] = 1.0
        w_diag = torch.diag(h).mean().clamp(min=1e-8)
        h[range(in_features), range(in_features)] += w_diag * self.damping

        jitter = w_diag * self.damping
        for _ in range(8):
            try:
                factor = torch.linalg.cholesky(h)
                break
            except Exception:  # noqa: BLE001, PERF203 - increase the ridge and retry
                h[range(in_features), range(in_features)] += jitter
                jitter *= 10
        else:  # pragma: no cover - pathologically ill-conditioned input
            factor = torch.eye(in_features, device=device)

        # Hinv = upper Cholesky factor of H^-1. The error propagation in the main
        # loop needs the upper triangle; taking triu() of the lower-triangular
        # cholesky_inverse would silently yield all zeros.
        inverse = torch.cholesky_inverse(factor)
        upper = torch.linalg.cholesky(inverse, upper=True)
        return torch.nan_to_num(upper, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _activation_order(h: torch.Tensor, in_features: int) -> torch.Tensor:
        """Quantize the highest-energy input channels first."""
        diagonal = torch.diag(h)
        if not torch.isfinite(diagonal).all() or bool((diagonal <= 0).all()):
            return torch.arange(in_features, device=h.device)
        return torch.argsort(diagonal, descending=True)


def _quantize_vector(
    w: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, bits: int, symmetric: bool
) -> torch.Tensor:
    """Round one column onto the integer grid defined by its scale/zero."""
    if symmetric:
        half = 2 ** (bits - 1) - 1
        return torch.clamp(torch.round(w / scale), -half - 1, half)
    return torch.clamp(torch.round(w / scale) + zero, 0, 2**bits - 1)


def _dequantize_vector(
    q: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, symmetric: bool = True
) -> torch.Tensor:
    """Invert :func:`_quantize_vector` for a single column.

    Works on signed codes; the unsigned bias is only applied at storage time.
    """
    return q * scale if symmetric else (q - zero) * scale


def build_quantizer(strategy: str, **kwargs: object) -> Quantizer:
    """Factory used by the config layer.

    Args:
        strategy: ``"gptq"``, ``"rtn"`` or ``"none"``.
        **kwargs: Forwarded to the concrete quantizer constructor.
    """
    if strategy == "gptq":
        allowed = {"bits", "group_size", "symmetric", "damping", "act_order"}
        return GPTQQuantizer(**{k: v for k, v in kwargs.items() if k in allowed})
    if strategy == "rtn":
        allowed = {"bits", "group_size", "symmetric"}
        return RoundToNearestQuantizer(**{k: v for k, v in kwargs.items() if k in allowed})
    raise ValueError(f"unknown quantization strategy: {strategy!r}")
