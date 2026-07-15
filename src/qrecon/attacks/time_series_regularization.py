from __future__ import annotations

import math
from typing import Literal

import torch
from torch.nn import functional as F

PenaltyLoss = Literal["l1", "l2"]


def _as_btc(sequence: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if sequence.ndim == 2:
        return sequence.unsqueeze(-1), True
    if sequence.ndim == 3:
        return sequence, False
    raise ValueError(
        "time-series regularizers require [batch, time] or [batch, time, channels]"
    )


def _penalty(error: torch.Tensor, loss: PenaltyLoss) -> torch.Tensor:
    if loss == "l1":
        return error.abs().mean()
    if loss == "l2":
        return error.square().mean()
    raise ValueError("loss must be 'l1' or 'l2'")


def linear_trend_penalty(
    sequence: torch.Tensor,
    *,
    loss: PenaltyLoss = "l1",
    detach_trend: bool = True,
) -> torch.Tensor:
    """Penalize deviations from a per-sample, per-channel linear trend.

    ``detach_trend=True`` matches the regularization geometry used by the public
    TS-Inverse implementation: the fitted trend is treated as a fixed target for
    the current optimization step. Setting it to false makes the least-squares
    trend fully differentiable. Neither option uses the private reference series.
    """

    values, _ = _as_btc(sequence)
    if values.shape[1] <= 1:
        return torch.zeros((), device=values.device, dtype=values.dtype)
    time = torch.arange(
        values.shape[1], device=values.device, dtype=values.dtype
    ).view(1, -1, 1)
    centered_time = time - time.mean(dim=1, keepdim=True)
    centered_values = values - values.mean(dim=1, keepdim=True)
    denominator = centered_time.square().sum(dim=1, keepdim=True).clamp_min(1e-12)
    slope = (centered_time * centered_values).sum(dim=1, keepdim=True) / denominator
    trend = slope * centered_time + values.mean(dim=1, keepdim=True)
    if detach_trend:
        trend = trend.detach()
    return _penalty(values - trend, loss)


def periodicity_penalty(
    sequence: torch.Tensor,
    period: int,
    *,
    loss: PenaltyLoss = "l1",
) -> torch.Tensor:
    """Penalize disagreement between points separated by a declared period."""

    values, _ = _as_btc(sequence)
    lag = int(period)
    if lag <= 0:
        raise ValueError("period must be positive")
    if lag >= values.shape[1]:
        raise ValueError("period must be smaller than the time dimension")
    return _penalty(values[:, lag:, :] - values[:, :-lag, :], loss)


def resolution_consistency_penalty(
    sequence: torch.Tensor,
    factor: int = 2,
    *,
    loss: PenaltyLoss = "l1",
) -> torch.Tensor:
    """Penalize high-frequency content removed by coarse average-pool interpolation."""

    values, _ = _as_btc(sequence)
    reduction = int(factor)
    if reduction <= 1:
        raise ValueError("factor must exceed one")
    time_steps = int(values.shape[1])
    if reduction > time_steps:
        raise ValueError("factor must not exceed the time dimension")
    channels_first = values.transpose(1, 2)
    coarse = F.avg_pool1d(
        channels_first,
        kernel_size=reduction,
        stride=reduction,
        ceil_mode=True,
    )
    restored = F.interpolate(
        coarse,
        size=time_steps,
        mode="linear",
        align_corners=False,
    ).transpose(1, 2)
    return _penalty(values - restored, loss)


def validate_regularization_weight(name: str, value: float) -> float:
    converted = float(value)
    if not math.isfinite(converted) or converted < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return converted
