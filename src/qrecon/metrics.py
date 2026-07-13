from __future__ import annotations

import math

import torch


def reconstruction_metrics(reference: torch.Tensor, estimate: torch.Tensor) -> dict[str, float]:
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
    return {
        "mse": mse,
        "mae": mae,
        "rmse": math.sqrt(mse),
        "psnr_unit_range": -10.0 * math.log10(max(mse, 1e-12)),
        "correlation": correlation,
        "exact_match_1e-3": float(torch.allclose(reference, estimate, atol=1e-3, rtol=0)),
    }

