from __future__ import annotations

import torch
from torch import nn


def _first_linear_parameter_names(model: nn.Module) -> tuple[str, str]:
    for module_name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.bias is not None:
            prefix = f"{module_name}." if module_name else ""
            return f"{prefix}weight", f"{prefix}bias"
    raise ValueError("model has no biased linear layer")


def invert_first_linear_gradient(
    model: nn.Module,
    observed_gradients: tuple[torch.Tensor, ...],
    input_shape: tuple[int, ...],
    minimum_bias_norm: float = 1e-12,
) -> torch.Tensor:
    """Exactly recover a batch-one input to the model's first linear layer.

    For one sample, grad(W) = delta outer x and grad(b) = delta. The
    least-squares solution across all non-zero delta rows is therefore

        x = grad(b)^T grad(W) / ||grad(b)||^2.

    Recovery is valid only when the first linear layer consumes the raw input
    (possibly after flattening) and the leaked gradient is not aggregated over
    multiple samples.
    """
    weight_name, bias_name = _first_linear_parameter_names(model)
    gradient_by_name = {
        name: gradient
        for (name, _), gradient in zip(model.named_parameters(), observed_gradients)
    }
    weight_gradient = gradient_by_name[weight_name]
    bias_gradient = gradient_by_name[bias_name].reshape(-1)
    denominator = bias_gradient.square().sum()
    if float(denominator) <= minimum_bias_norm:
        raise RuntimeError("first-layer bias gradient is too small for stable recovery")
    flattened = (bias_gradient[:, None] * weight_gradient).sum(dim=0) / denominator
    expected = 1
    for dimension in input_shape:
        expected *= dimension
    if flattened.numel() != expected:
        raise ValueError(
            "the first linear layer does not directly consume the raw input: "
            f"expected {expected} features, observed {flattened.numel()}"
        )
    return flattened.reshape(input_shape).detach()


def infer_class_label_from_last_bias(
    model: nn.Module, observed_gradients: tuple[torch.Tensor, ...]
) -> torch.Tensor:
    """iDLG-style batch-one class inference from the final linear bias."""
    candidates: list[tuple[str, nn.Linear]] = [
        (name, module) for name, module in model.named_modules() if isinstance(module, nn.Linear)
    ]
    if not candidates:
        raise ValueError("model has no linear classifier")
    module_name, module = candidates[-1]
    if module.bias is None:
        raise ValueError("final linear classifier has no bias")
    bias_name = f"{module_name}.bias" if module_name else "bias"
    gradient_by_name = {
        name: gradient
        for (name, _), gradient in zip(model.named_parameters(), observed_gradients)
    }
    return gradient_by_name[bias_name].argmin().reshape(1).detach()

