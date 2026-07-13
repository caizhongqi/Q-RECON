from __future__ import annotations

import math

import torch


def reconstruction_metrics(
    reference: torch.Tensor, estimate: torch.Tensor, mode: str | None = None
) -> dict[str, float]:
    reference = reference.detach().float().reshape(-1)
    estimate = estimate.detach().float().reshape(-1)
    error = reference - estimate
    mse = float(error.square().mean())
    mae = float(error.abs().mean())
    centered_ref = reference - reference.mean()
    centered_est = estimate - estimate.mean()
    correlation = float(
        (centered_ref * centered_est).sum()
        / (centered_ref.norm() * centered_est.norm() + 1e-12)
    )
    metrics = {
        "mse": mse,
        "mae": mae,
        "rmse": math.sqrt(mse),
        "max_absolute_error": float(error.abs().max()),
        "relative_l2_error": float(error.norm() / (reference.norm() + 1e-12)),
        "psnr_unit_range": -10.0 * math.log10(max(mse, 1e-12)),
        "correlation": correlation,
        "bitwise_equal_percent": float((reference == estimate).float().mean() * 100),
    }
    for tolerance in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1):
        key = f"within_{tolerance:g}_percent"
        metrics[key] = float((error.abs() <= tolerance).float().mean() * 100)
    if mode == "image":
        reference_8bit = (reference.clamp(0, 1) * 255).round()
        estimate_8bit = (estimate.clamp(0, 1) * 255).round()
        metrics["uint8_equal_percent"] = float(
            (reference_8bit == estimate_8bit).float().mean() * 100
        )
    return metrics
