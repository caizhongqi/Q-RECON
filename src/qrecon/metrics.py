from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class BatchAlignmentReport:
    assignment: tuple[int, ...]
    assignment_total_mse: float
    assignment_mean_mse: float
    exact_batch_within_tolerance: bool
    record_success_count: int
    record_success_rate: float
    tolerance: float
    aligned_metrics: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["assignment"] = list(self.assignment)
        payload["aligned_metrics"] = dict(self.aligned_metrics)
        return payload


@dataclass(frozen=True)
class ChannelAlignmentReport:
    """Exact small-channel assignment report for anonymous multivariate records."""

    assignment: tuple[int, ...]
    assignment_total_mse: float
    assignment_mean_mse: float
    exact_tensor_within_tolerance: bool
    channel_success_count: int
    channel_success_rate: float
    tolerance: float
    aligned_metrics: dict[str, float]

    @property
    def identity_assignment(self) -> bool:
        return self.assignment == tuple(range(len(self.assignment)))

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["assignment"] = list(self.assignment)
        payload["identity_assignment"] = self.identity_assignment
        payload["aligned_metrics"] = dict(self.aligned_metrics)
        return payload


def reconstruction_metrics(
    reference: torch.Tensor, estimate: torch.Tensor, mode: str | None = None
) -> dict[str, float]:
    reference = reference.detach().float().reshape(-1)
    estimate = estimate.detach().float().reshape(-1)
    if reference.shape != estimate.shape:
        raise ValueError("reference and estimate must contain the same number of values")
    if reference.numel() == 0:
        raise ValueError("reference and estimate must be non-empty")
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
    if mode == "timeseries":
        denominator = reference.abs() + estimate.abs()
        smape = torch.where(
            denominator > 1e-12,
            2.0 * error.abs() / denominator,
            torch.zeros_like(denominator),
        )
        metrics["smape_percent"] = float(smape.mean() * 100.0)
    return metrics


def _minimum_cost_assignment(cost: torch.Tensor) -> tuple[float, tuple[int, ...]]:
    """Solve a small square assignment exactly using bitmask dynamic programming."""

    if cost.ndim != 2 or cost.shape[0] != cost.shape[1]:
        raise ValueError("cost must be a square matrix")
    size = int(cost.shape[0])
    if size <= 0:
        raise ValueError("cost must be non-empty")
    if size > 12:
        raise ValueError(
            "exact permutation-invariant alignment is limited to size 12; "
            "use a smaller object or add a declared approximate matcher"
        )

    # mask -> (cost, assignment for reference rows processed so far)
    states: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    for row in range(size):
        next_states: dict[int, tuple[float, tuple[int, ...]]] = {}
        for mask, (prefix_cost, assignment) in states.items():
            for column in range(size):
                bit = 1 << column
                if mask & bit:
                    continue
                new_mask = mask | bit
                candidate = (
                    prefix_cost + float(cost[row, column]),
                    assignment + (column,),
                )
                incumbent = next_states.get(new_mask)
                if incumbent is None or candidate < incumbent:
                    next_states[new_mask] = candidate
        states = next_states
    return states[(1 << size) - 1]


def permutation_invariant_batch_metrics(
    reference: torch.Tensor,
    estimate: torch.Tensor,
    *,
    mode: str | None = None,
    tolerance: float = 1e-2,
) -> BatchAlignmentReport:
    """Evaluate an aggregate-gradient batch modulo record permutation.

    Mean training gradients are invariant to batch order. Reporting only ordered
    elementwise error would therefore count a correct recovered set as a failure.
    This helper finds the exact minimum-MSE assignment for small batches, reports
    the aligned reconstruction metrics, and separately records all-record and
    per-record tolerance success.
    """

    if reference.shape != estimate.shape:
        raise ValueError("reference and estimate must have identical shapes")
    if reference.ndim < 2:
        raise ValueError("batch alignment requires a leading batch dimension")
    if not math.isfinite(float(tolerance)) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and non-negative")

    ref = reference.detach().float()
    est = estimate.detach().float()
    batch = int(ref.shape[0])
    if batch <= 0:
        raise ValueError("batch must be non-empty")
    ref_flat = ref.reshape(batch, -1)
    est_flat = est.reshape(batch, -1)
    cost = (ref_flat[:, None, :] - est_flat[None, :, :]).square().mean(dim=-1)
    total_cost, assignment = _minimum_cost_assignment(cost)
    aligned = est[list(assignment)]
    absolute_error = (ref - aligned).abs().reshape(batch, -1)
    record_success = absolute_error.max(dim=1).values <= float(tolerance)
    successes = int(record_success.sum())
    return BatchAlignmentReport(
        assignment=assignment,
        assignment_total_mse=float(total_cost),
        assignment_mean_mse=float(total_cost / batch),
        exact_batch_within_tolerance=bool(record_success.all()),
        record_success_count=successes,
        record_success_rate=successes / batch,
        tolerance=float(tolerance),
        aligned_metrics=reconstruction_metrics(ref, aligned, mode=mode),
    )


def permutation_invariant_channel_metrics(
    reference: torch.Tensor,
    estimate: torch.Tensor,
    *,
    mode: str | None = "timeseries",
    tolerance: float = 1e-2,
) -> ChannelAlignmentReport:
    """Evaluate a multivariate object modulo one global channel permutation.

    Inputs must use ``[..., time, channels]`` semantics, such as
    ``[batch, time, channels]``. The same assignment is applied to every batch and
    time coordinate, matching the anonymous-channel orbit used by Q-RECON's
    identifiability theorem. The assignment is selected by the complete supplied
    private object; callers should concatenate context and target time axes when
    both are part of the recovery target.
    """

    if reference.shape != estimate.shape:
        raise ValueError("reference and estimate must have identical shapes")
    if reference.ndim < 2:
        raise ValueError("channel alignment requires a trailing channel dimension")
    if not math.isfinite(float(tolerance)) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and non-negative")

    ref = reference.detach().float()
    est = estimate.detach().float()
    channels = int(ref.shape[-1])
    if channels <= 0:
        raise ValueError("channel dimension must be non-empty")
    ref_channels = ref.movedim(-1, 0).reshape(channels, -1)
    est_channels = est.movedim(-1, 0).reshape(channels, -1)
    cost = (
        ref_channels[:, None, :] - est_channels[None, :, :]
    ).square().mean(dim=-1)
    total_cost, assignment = _minimum_cost_assignment(cost)
    aligned = est[..., list(assignment)]
    absolute_error = (ref - aligned).abs().movedim(-1, 0).reshape(channels, -1)
    channel_success = absolute_error.max(dim=1).values <= float(tolerance)
    successes = int(channel_success.sum())
    return ChannelAlignmentReport(
        assignment=assignment,
        assignment_total_mse=float(total_cost),
        assignment_mean_mse=float(total_cost / channels),
        exact_tensor_within_tolerance=bool(channel_success.all()),
        channel_success_count=successes,
        channel_success_rate=successes / channels,
        tolerance=float(tolerance),
        aligned_metrics=reconstruction_metrics(ref, aligned, mode=mode),
    )
