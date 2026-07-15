from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .gradient_inversion import AttackResult
from .time_series_regularization import (
    PenaltyLoss,
    linear_trend_penalty,
    periodicity_penalty,
    resolution_consistency_penalty,
    validate_regularization_weight,
)


TS_INVERSE_REFERENCE = {
    "repository": "Capsar/ts-inverse",
    "commit": "2015946906a693f836e6418cdeb3b64a3f6f2d6e",
    "implemented_components": (
        "sum-L1 gradient distance",
        "input and target total variation",
        "joint input-target linear-trend penalty",
        "joint input-target periodicity penalty",
        "joint input-target low-resolution consistency penalty",
        "optional quantile-bound hinge penalty",
    ),
    "not_implemented_components": (
        "learned gradient-to-input initializer",
        "learned probabilistic or quantile inversion network",
    ),
}


def l1_gradient_distance(
    candidate: tuple[torch.Tensor, ...],
    observed: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    """Exact sum-L1 gradient distance used by the public TS-Inverse code."""

    if len(candidate) != len(observed) or not candidate:
        raise ValueError("candidate and observed gradients must have equal non-zero length")
    loss = torch.zeros((), device=candidate[0].device, dtype=candidate[0].dtype)
    for left, right in zip(candidate, observed):
        if left.shape != right.shape:
            raise ValueError("candidate and observed gradient tensors must have equal shapes")
        loss = loss + F.l1_loss(left, right, reduction="sum")
    return loss


def temporal_total_variation_l1(sequence: torch.Tensor) -> torch.Tensor:
    """Mean L1 temporal variation for ``[B,T]`` or ``[B,T,C]`` sequences."""

    if sequence.ndim not in (2, 3):
        raise ValueError("time-series tensors must have shape [B,T] or [B,T,C]")
    if sequence.shape[1] <= 1:
        return torch.zeros((), device=sequence.device, dtype=sequence.dtype)
    return (sequence[:, 1:, ...] - sequence[:, :-1, ...]).abs().mean()


def joint_forecasting_sequence(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    channel: int = 0,
) -> torch.Tensor:
    """Concatenate one input variate with its forecast target along time.

    The public TS-Inverse implementation regularizes the first input feature and
    the target as one continuous sequence. Q-RECON makes the selected channel
    explicit so the same contract can be audited on multivariate data.
    """

    if inputs.ndim == 2:
        input_series = inputs
    elif inputs.ndim == 3:
        if channel < 0 or channel >= inputs.shape[2]:
            raise ValueError("joint regularization channel is outside the input tensor")
        input_series = inputs[:, :, channel]
    else:
        raise ValueError("inputs must have shape [B,T] or [B,T,C]")

    if targets.ndim == 2:
        target_series = targets
    elif targets.ndim == 3:
        if channel < 0 or channel >= targets.shape[2]:
            raise ValueError("joint regularization channel is outside the target tensor")
        target_series = targets[:, :, channel]
    else:
        raise ValueError("targets must have shape [B,H] or [B,H,C]")

    if input_series.shape[0] != target_series.shape[0]:
        raise ValueError("input and target batch dimensions must match")
    return torch.cat((input_series, target_series), dim=1)


def quantile_bound_hinge_penalty(
    sequence: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    """Paired hinge penalty for values outside declared quantile bounds."""

    if sequence.shape != lower.shape or sequence.shape != upper.shape:
        raise ValueError("sequence and quantile bounds must have identical shapes")
    if torch.any(lower > upper):
        raise ValueError("quantile lower bounds must not exceed upper bounds")
    return 0.5 * (
        F.relu(sequence - upper).mean() + F.relu(lower - sequence).mean()
    )


@dataclass(frozen=True)
class TSInverseStyleComponents:
    gradient_l1: float
    input_total_variation: float
    target_total_variation: float
    trend: float
    periodicity: float
    low_resolution: float
    quantile_bounds: float

    def to_dict(self) -> dict[str, float]:
        return {
            "gradient_l1": self.gradient_l1,
            "input_total_variation": self.input_total_variation,
            "target_total_variation": self.target_total_variation,
            "trend": self.trend,
            "periodicity": self.periodicity,
            "low_resolution": self.low_resolution,
            "quantile_bounds": self.quantile_bounds,
        }


class TSInverseStyleAttack:
    """Optimization baseline reproducing the public TS-Inverse objective family.

    This class intentionally does not claim to reproduce the learned
    gradient-to-quantile initializer from TS-Inverse. It implements the public
    optimization losses exactly enough to provide a strong, provenance-tracked
    classical baseline on the same victim, data, gradients and restart seeds.
    """

    def __init__(
        self,
        model: nn.Module,
        observed_gradients: tuple[torch.Tensor, ...],
        prior: nn.Module,
        *,
        known_target: torch.Tensor | None,
        target_shape: tuple[int, ...],
        steps: int = 300,
        learning_rate: float = 0.05,
        optimizer_name: str = "adam",
        gradient_l1_weight: float = 1.0,
        input_total_variation_weight: float = 0.0,
        target_total_variation_weight: float = 0.0,
        trend_weight: float = 0.0,
        trend_loss: PenaltyLoss = "l1",
        trend_detach: bool = True,
        periodicity_weight: float = 0.0,
        periodicity_period: int | None = None,
        periodicity_loss: PenaltyLoss = "l1",
        low_resolution_weight: float = 0.0,
        low_resolution_factor: int = 2,
        low_resolution_loss: PenaltyLoss = "l1",
        quantile_bound_weight: float = 0.0,
        input_quantile_lower: torch.Tensor | None = None,
        input_quantile_upper: torch.Tensor | None = None,
        target_quantile_lower: torch.Tensor | None = None,
        target_quantile_upper: torch.Tensor | None = None,
        joint_channel: int = 0,
        gradient_clip_norm: float | None = None,
        record_every: int | None = None,
    ) -> None:
        if steps <= 0:
            raise ValueError("steps must be positive")
        if not math.isfinite(float(learning_rate)) or learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        self.gradient_l1_weight = validate_regularization_weight(
            "gradient_l1_weight", gradient_l1_weight
        )
        self.input_total_variation_weight = validate_regularization_weight(
            "input_total_variation_weight", input_total_variation_weight
        )
        self.target_total_variation_weight = validate_regularization_weight(
            "target_total_variation_weight", target_total_variation_weight
        )
        self.trend_weight = validate_regularization_weight("trend_weight", trend_weight)
        self.periodicity_weight = validate_regularization_weight(
            "periodicity_weight", periodicity_weight
        )
        self.low_resolution_weight = validate_regularization_weight(
            "low_resolution_weight", low_resolution_weight
        )
        self.quantile_bound_weight = validate_regularization_weight(
            "quantile_bound_weight", quantile_bound_weight
        )
        if trend_loss not in ("l1", "l2"):
            raise ValueError("trend_loss must be 'l1' or 'l2'")
        if periodicity_loss not in ("l1", "l2"):
            raise ValueError("periodicity_loss must be 'l1' or 'l2'")
        if low_resolution_loss not in ("l1", "l2"):
            raise ValueError("low_resolution_loss must be 'l1' or 'l2'")
        if self.periodicity_weight > 0.0 and (
            periodicity_period is None or int(periodicity_period) <= 0
        ):
            raise ValueError("positive periodicity_weight requires periodicity_period")
        if int(low_resolution_factor) <= 1:
            raise ValueError("low_resolution_factor must exceed one")
        if gradient_clip_norm is not None and (
            not math.isfinite(float(gradient_clip_norm)) or gradient_clip_norm <= 0.0
        ):
            raise ValueError("gradient_clip_norm must be finite and positive")
        if record_every is not None and record_every <= 0:
            raise ValueError("record_every must be positive")

        bound_tensors = (
            input_quantile_lower,
            input_quantile_upper,
            target_quantile_lower,
            target_quantile_upper,
        )
        if self.quantile_bound_weight > 0.0 and any(
            tensor is None for tensor in bound_tensors
        ):
            raise ValueError(
                "positive quantile_bound_weight requires input and target lower/upper bounds"
            )

        self.model = model
        self.observed = observed_gradients
        self.prior = prior
        self.known_target = known_target
        self.target_shape = target_shape
        self.steps = int(steps)
        self.learning_rate = float(learning_rate)
        self.optimizer_name = str(optimizer_name).lower()
        self.trend_loss = trend_loss
        self.trend_detach = bool(trend_detach)
        self.periodicity_period = (
            None if periodicity_period is None else int(periodicity_period)
        )
        self.periodicity_loss = periodicity_loss
        self.low_resolution_factor = int(low_resolution_factor)
        self.low_resolution_loss = low_resolution_loss
        self.input_quantile_lower = input_quantile_lower
        self.input_quantile_upper = input_quantile_upper
        self.target_quantile_lower = target_quantile_lower
        self.target_quantile_upper = target_quantile_upper
        self.joint_channel = int(joint_channel)
        self.gradient_clip_norm = gradient_clip_norm
        self.record_every = record_every

    @property
    def provenance(self) -> Mapping[str, object]:
        return TS_INVERSE_REFERENCE

    def run(self) -> AttackResult:
        parameters = list(self.prior.parameters())
        if not parameters:
            raise ValueError("the reconstruction prior must expose trainable parameters")
        target_parameter: nn.Parameter | None = None
        if self.known_target is None:
            target_parameter = nn.Parameter(torch.zeros(self.target_shape))
            parameters.append(target_parameter)

        if self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(parameters, lr=self.learning_rate)
        elif self.optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(parameters, lr=self.learning_rate)
        else:
            raise ValueError("TS-Inverse-style attack supports adam or adamw")

        history: list[dict[str, float]] = []
        record_every = self.record_every or max(1, self.steps // 20)
        best_objective = math.inf
        best_gradient = math.inf
        best_step = 0
        best_reconstruction: torch.Tensor | None = None
        best_target: torch.Tensor | None = None

        def clear_model_gradients() -> None:
            for parameter in self.model.parameters():
                parameter.grad = None

        def bounds_to_device(
            tensor: torch.Tensor | None, reference: torch.Tensor
        ) -> torch.Tensor | None:
            return None if tensor is None else tensor.to(reference.device, reference.dtype)

        def evaluate() -> tuple[
            torch.Tensor,
            torch.Tensor,
            TSInverseStyleComponents,
            torch.Tensor,
            torch.Tensor,
        ]:
            candidate = self.prior()
            candidate_target = (
                self.known_target if target_parameter is None else target_parameter
            )
            prediction = self.model(candidate)
            task_loss = F.mse_loss(prediction, candidate_target)
            candidate_gradients = torch.autograd.grad(
                task_loss, tuple(self.model.parameters()), create_graph=True
            )
            gradient_l1 = l1_gradient_distance(candidate_gradients, self.observed)

            input_tv = temporal_total_variation_l1(candidate)
            target_tv = temporal_total_variation_l1(candidate_target)
            joint = joint_forecasting_sequence(
                candidate, candidate_target, channel=self.joint_channel
            )
            trend = linear_trend_penalty(
                joint, loss=self.trend_loss, detach_trend=self.trend_detach
            )
            if self.periodicity_weight > 0.0:
                assert self.periodicity_period is not None
                periodicity = periodicity_penalty(
                    joint,
                    self.periodicity_period,
                    loss=self.periodicity_loss,
                )
            else:
                periodicity = torch.zeros((), device=joint.device, dtype=joint.dtype)
            if self.low_resolution_weight > 0.0:
                low_resolution = resolution_consistency_penalty(
                    joint,
                    self.low_resolution_factor,
                    loss=self.low_resolution_loss,
                )
            else:
                low_resolution = torch.zeros((), device=joint.device, dtype=joint.dtype)

            quantile_bounds = torch.zeros((), device=joint.device, dtype=joint.dtype)
            if self.quantile_bound_weight > 0.0:
                input_lower = bounds_to_device(self.input_quantile_lower, candidate)
                input_upper = bounds_to_device(self.input_quantile_upper, candidate)
                target_lower = bounds_to_device(
                    self.target_quantile_lower, candidate_target
                )
                target_upper = bounds_to_device(
                    self.target_quantile_upper, candidate_target
                )
                assert input_lower is not None and input_upper is not None
                assert target_lower is not None and target_upper is not None
                quantile_bounds = 0.5 * (
                    quantile_bound_hinge_penalty(candidate, input_lower, input_upper)
                    + quantile_bound_hinge_penalty(
                        candidate_target, target_lower, target_upper
                    )
                )

            objective = (
                self.gradient_l1_weight * gradient_l1
                + self.input_total_variation_weight * input_tv
                + self.target_total_variation_weight * target_tv
                + self.trend_weight * trend
                + self.periodicity_weight * periodicity
                + self.low_resolution_weight * low_resolution
                + self.quantile_bound_weight * quantile_bounds
            )
            components = TSInverseStyleComponents(
                gradient_l1=float(gradient_l1.detach()),
                input_total_variation=float(input_tv.detach()),
                target_total_variation=float(target_tv.detach()),
                trend=float(trend.detach()),
                periodicity=float(periodicity.detach()),
                low_resolution=float(low_resolution.detach()),
                quantile_bounds=float(quantile_bounds.detach()),
            )
            return objective, gradient_l1, components, candidate, candidate_target

        for step in range(1, self.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            clear_model_gradients()
            objective, gradient_l1, components, candidate, candidate_target = evaluate()
            objective_value = float(objective.detach())
            gradient_value = float(gradient_l1.detach())
            if not math.isfinite(objective_value) or not math.isfinite(gradient_value):
                raise FloatingPointError("TS-Inverse-style objective became non-finite")
            if objective_value < best_objective:
                best_objective = objective_value
                best_gradient = gradient_value
                best_step = step
                best_reconstruction = candidate.detach().clone()
                best_target = candidate_target.detach().clone()
            objective.backward()
            if self.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(parameters, self.gradient_clip_norm)
            optimizer.step()

            if step == 1 or step % record_every == 0 or step == self.steps:
                history.append(
                    {
                        "step": float(step),
                        "objective": objective_value,
                        "gradient_match": gradient_value,
                        "regularizer": objective_value
                        - self.gradient_l1_weight * gradient_value,
                        "best_objective": best_objective,
                        "best_gradient_match": best_gradient,
                        **components.to_dict(),
                    }
                )

        clear_model_gradients()
        final_objective_tensor, final_gradient_tensor, _, final_x, final_target = evaluate()
        final_objective = float(final_objective_tensor.detach())
        final_gradient = float(final_gradient_tensor.detach())
        if final_objective < best_objective:
            best_objective = final_objective
            best_gradient = final_gradient
            best_step = self.steps + 1
            best_reconstruction = final_x.detach().clone()
            best_target = final_target.detach().clone()
        clear_model_gradients()

        if best_reconstruction is None or best_target is None:
            raise RuntimeError("TS-Inverse-style attack produced no finite candidate")
        return AttackResult(
            reconstruction=best_reconstruction,
            reconstructed_target=best_target,
            history=history,
            best_objective=best_objective,
            best_gradient_match=best_gradient,
            best_step=best_step,
            final_objective=final_objective,
            final_gradient_match=final_gradient,
        )
