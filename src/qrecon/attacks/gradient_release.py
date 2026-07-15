from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Sequence

import torch
from torch import nn


@dataclass(frozen=True)
class GradientReleaseSpec:
    """Declared transformation from exact private gradients to an attacker view.

    Operations are applied in this fixed order:

    1. global L2 clipping over the complete parameter-gradient tuple;
    2. independent Gaussian noise on every parameter tensor;
    3. global-scale symmetric signed quantization;
    4. selection of visible parameter tensors.

    The quantization scale is included in the released metadata and is assumed
    public. The noise realization is not supplied to the attack algorithm even
    though its seed is retained for artifact reproducibility.
    """

    clip_norm: float | None = None
    noise_std: float = 0.0
    noise_seed: int = 0
    quantization_bits: int | None = None
    quantization_scale: float | None = None
    visible_parameter_indices: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.clip_norm is not None and (
            not math.isfinite(float(self.clip_norm)) or self.clip_norm <= 0.0
        ):
            raise ValueError("clip_norm must be finite and positive")
        if not math.isfinite(float(self.noise_std)) or self.noise_std < 0.0:
            raise ValueError("noise_std must be finite and non-negative")
        if self.quantization_bits is not None and self.quantization_bits < 2:
            raise ValueError("quantization_bits must be at least two")
        if self.quantization_scale is not None and (
            not math.isfinite(float(self.quantization_scale))
            or self.quantization_scale <= 0.0
        ):
            raise ValueError("quantization_scale must be finite and positive")
        if self.quantization_bits is None and self.quantization_scale is not None:
            raise ValueError("quantization_scale requires quantization_bits")
        if self.visible_parameter_indices is not None:
            if not self.visible_parameter_indices:
                raise ValueError("visible_parameter_indices must be non-empty")
            if any(index < 0 for index in self.visible_parameter_indices):
                raise ValueError("visible parameter indices must be non-negative")
            if len(set(self.visible_parameter_indices)) != len(
                self.visible_parameter_indices
            ):
                raise ValueError("visible parameter indices must be unique")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        if self.visible_parameter_indices is not None:
            payload["visible_parameter_indices"] = list(
                self.visible_parameter_indices
            )
        return payload


@dataclass(frozen=True)
class GradientRelease:
    gradients: tuple[torch.Tensor, ...]
    visible_parameter_indices: tuple[int, ...]
    visible_parameter_names: tuple[str, ...]
    total_parameter_tensors: int
    clip_norm: float | None
    raw_l2_norm: float
    clipped_l2_norm: float
    clipping_factor: float
    noise_std: float
    noise_seed: int
    quantization_bits: int | None
    quantization_scale: float | None
    quantized_saturation_count: int
    quantized_value_count: int

    @property
    def quantized_saturation_rate(self) -> float:
        return self.quantized_saturation_count / max(self.quantized_value_count, 1)

    def to_dict(self) -> dict[str, object]:
        return {
            "visible_parameter_indices": list(self.visible_parameter_indices),
            "visible_parameter_names": list(self.visible_parameter_names),
            "visible_parameter_tensors": len(self.visible_parameter_indices),
            "total_parameter_tensors": self.total_parameter_tensors,
            "clip_norm": self.clip_norm,
            "raw_l2_norm": self.raw_l2_norm,
            "clipped_l2_norm": self.clipped_l2_norm,
            "clipping_factor": self.clipping_factor,
            "noise_std": self.noise_std,
            "noise_seed": self.noise_seed,
            "quantization_bits": self.quantization_bits,
            "quantization_scale": self.quantization_scale,
            "quantized_saturation_count": self.quantized_saturation_count,
            "quantized_value_count": self.quantized_value_count,
            "quantized_saturation_rate": self.quantized_saturation_rate,
        }

    def deterministic_attack_contract(self) -> dict[str, object]:
        """Known deterministic operations an attacker can apply to candidates.

        Gaussian noise is intentionally absent: its realization is not public.
        The declared clipping threshold remains part of the contract even when the
        observed private gradient happened not to trigger clipping.
        """

        return {
            "visible_parameter_indices": self.visible_parameter_indices,
            "clip_norm": self.clip_norm,
            "quantization_bits": self.quantization_bits,
            "quantization_scale": self.quantization_scale,
        }


def gradient_tuple_l2_norm(gradients: Sequence[torch.Tensor]) -> torch.Tensor:
    values = tuple(gradients)
    if not values:
        raise ValueError("gradient tuple must be non-empty")
    return torch.sqrt(sum(gradient.square().sum() for gradient in values))


def clip_gradient_tuple(
    gradients: Sequence[torch.Tensor],
    clip_norm: float | None,
    *,
    epsilon: float = 1e-12,
) -> tuple[tuple[torch.Tensor, ...], float]:
    values = tuple(gradients)
    if not values:
        raise ValueError("gradient tuple must be non-empty")
    if clip_norm is None:
        return values, 1.0
    threshold = float(clip_norm)
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("clip_norm must be finite and positive")
    norm = gradient_tuple_l2_norm(values)
    factor_tensor = torch.clamp(
        torch.as_tensor(threshold, device=norm.device, dtype=norm.dtype)
        / (norm + epsilon),
        max=1.0,
    )
    factor = float(factor_tensor.detach())
    return tuple(gradient * factor_tensor for gradient in values), factor


def quantize_gradient_tuple(
    gradients: Sequence[torch.Tensor],
    bits: int,
    *,
    scale: float | None = None,
    straight_through: bool = False,
) -> tuple[tuple[torch.Tensor, ...], float, int, int]:
    values = tuple(gradients)
    if not values:
        raise ValueError("gradient tuple must be non-empty")
    width = int(bits)
    if width < 2:
        raise ValueError("bits must be at least two")
    qmax = (1 << (width - 1)) - 1
    if scale is None:
        maximum = max(float(gradient.detach().abs().max()) for gradient in values)
        used_scale = maximum / qmax if maximum > 0.0 else 1.0
    else:
        used_scale = float(scale)
        if not math.isfinite(used_scale) or used_scale <= 0.0:
            raise ValueError("scale must be finite and positive")

    result: list[torch.Tensor] = []
    saturation_count = 0
    value_count = 0
    for gradient in values:
        scaled = gradient / used_scale
        rounded = torch.round(scaled)
        saturation_count += int((rounded.abs() > qmax).sum().detach())
        value_count += gradient.numel()
        quantized = rounded.clamp(-qmax, qmax) * used_scale
        if straight_through:
            quantized = gradient + (quantized - gradient).detach()
        result.append(quantized)
    return tuple(result), used_scale, saturation_count, value_count


def _add_gaussian_noise(
    gradients: Sequence[torch.Tensor],
    standard_deviation: float,
    seed: int,
) -> tuple[torch.Tensor, ...]:
    values = tuple(gradients)
    if standard_deviation == 0.0:
        return values
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    result: list[torch.Tensor] = []
    for gradient in values:
        noise = torch.randn(
            gradient.shape,
            generator=generator,
            dtype=gradient.dtype,
            device="cpu",
        ).to(gradient.device)
        result.append(gradient + float(standard_deviation) * noise)
    return tuple(result)


def release_gradients(
    model: nn.Module,
    exact_gradients: Sequence[torch.Tensor],
    spec: GradientReleaseSpec,
) -> GradientRelease:
    parameters = tuple(model.named_parameters())
    gradients = tuple(gradient.detach().clone() for gradient in exact_gradients)
    if len(parameters) != len(gradients) or not gradients:
        raise ValueError("exact gradients must match every model parameter tensor")

    raw_norm = float(gradient_tuple_l2_norm(gradients))
    clipped, factor = clip_gradient_tuple(gradients, spec.clip_norm)
    clipped_norm = float(gradient_tuple_l2_norm(clipped))
    noisy = _add_gaussian_noise(clipped, spec.noise_std, spec.noise_seed)

    quantization_scale: float | None = None
    saturation_count = 0
    value_count = sum(gradient.numel() for gradient in noisy)
    released_all = noisy
    if spec.quantization_bits is not None:
        released_all, quantization_scale, saturation_count, value_count = (
            quantize_gradient_tuple(
                noisy,
                spec.quantization_bits,
                scale=spec.quantization_scale,
            )
        )

    indices = (
        tuple(range(len(parameters)))
        if spec.visible_parameter_indices is None
        else tuple(spec.visible_parameter_indices)
    )
    if any(index >= len(parameters) for index in indices):
        raise ValueError("visible parameter index is outside the model")
    names = tuple(parameters[index][0] for index in indices)
    visible = tuple(released_all[index] for index in indices)
    return GradientRelease(
        gradients=visible,
        visible_parameter_indices=indices,
        visible_parameter_names=names,
        total_parameter_tensors=len(parameters),
        clip_norm=(None if spec.clip_norm is None else float(spec.clip_norm)),
        raw_l2_norm=raw_norm,
        clipped_l2_norm=clipped_norm,
        clipping_factor=factor,
        noise_std=float(spec.noise_std),
        noise_seed=int(spec.noise_seed),
        quantization_bits=spec.quantization_bits,
        quantization_scale=quantization_scale,
        quantized_saturation_count=saturation_count,
        quantized_value_count=value_count,
    )


def last_biased_linear_parameter_indices(model: nn.Module) -> tuple[int, int]:
    """Return weight/bias indices of the final biased Linear in parameter order."""

    candidates = [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and module.bias is not None
    ]
    if not candidates:
        raise ValueError("model contains no biased Linear")
    module_name = candidates[-1]
    weight_name = f"{module_name}.weight" if module_name else "weight"
    bias_name = f"{module_name}.bias" if module_name else "bias"
    mapping = {name: index for index, (name, _) in enumerate(model.named_parameters())}
    return mapping[weight_name], mapping[bias_name]
