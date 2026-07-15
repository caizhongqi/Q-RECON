from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

from .gradient_inversion import _regularizer


@dataclass(frozen=True)
class LinearHeadLeakageReport:
    module_name: str
    effective_samples: int
    feature_dimension: int
    recovered_feature: torch.Tensor
    bias_gradient_norm: float
    rank_one_relative_residual: float

    def to_dict(self, *, include_feature: bool = False) -> dict[str, object]:
        payload: dict[str, object] = {
            "module_name": self.module_name,
            "effective_samples": self.effective_samples,
            "feature_dimension": self.feature_dimension,
            "bias_gradient_norm": self.bias_gradient_norm,
            "rank_one_relative_residual": self.rank_one_relative_residual,
        }
        if include_feature:
            payload["recovered_feature"] = self.recovered_feature.detach().cpu().tolist()
        return payload


@dataclass(frozen=True)
class HeadRepresentationAttackResult:
    reconstruction: torch.Tensor
    recovered_feature: torch.Tensor
    history: tuple[dict[str, float], ...]
    best_objective: float
    best_representation_loss: float
    best_step: int
    final_objective: float
    final_representation_loss: float
    leakage: LinearHeadLeakageReport


def find_last_biased_linear(model: nn.Module) -> tuple[str, nn.Linear]:
    candidates = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and module.bias is not None
    ]
    if not candidates:
        raise ValueError("model contains no biased linear layer")
    return candidates[-1]


def _parameter_name(module_name: str, leaf: str) -> str:
    return f"{module_name}.{leaf}" if module_name else leaf


def recover_single_effective_head_input(
    model: nn.Module,
    observed_gradients: tuple[torch.Tensor, ...],
    *,
    effective_samples: int = 1,
    epsilon: float = 1e-12,
) -> LinearHeadLeakageReport:
    """Recover the input of the final biased Linear from one effective sample.

    For ``y = Wz + b`` and any scalar loss, one effective sample gives
    ``grad_W = delta z^T`` and ``grad_b = delta``. Hence

    ``z = grad_b^T grad_W / ||grad_b||^2``.

    PatchTST and iTransformer often apply a shared head to ``batch * channels``
    effective samples. The exact decoder is therefore deliberately rejected unless
    that public product equals one (or channel-specific heads are handled
    separately). A low rank-one residual is a consistency check, not a substitute
    for the effective-sample assumption.
    """

    if effective_samples != 1:
        raise ValueError(
            "exact head-input recovery requires one effective sample; shared heads "
            "aggregate batch/channel contributions"
        )
    if len(observed_gradients) != len(tuple(model.parameters())):
        raise ValueError("observed gradient count does not match model parameters")
    if not math.isfinite(float(epsilon)) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")

    module_name, linear = find_last_biased_linear(model)
    name_to_index = {
        name: index for index, (name, _) in enumerate(model.named_parameters())
    }
    weight_name = _parameter_name(module_name, "weight")
    bias_name = _parameter_name(module_name, "bias")
    if weight_name not in name_to_index or bias_name not in name_to_index:
        raise RuntimeError("final linear parameters were not found in named_parameters")
    weight_gradient = observed_gradients[name_to_index[weight_name]].detach()
    bias_gradient = observed_gradients[name_to_index[bias_name]].detach()
    if weight_gradient.shape != linear.weight.shape or bias_gradient.shape != linear.bias.shape:
        raise ValueError("observed final-head gradient shapes do not match the model")

    energy = bias_gradient.square().sum()
    energy_value = float(energy)
    if energy_value <= epsilon:
        raise ZeroDivisionError(
            "final-head bias gradient is zero; the head representation is not recoverable"
        )
    feature = torch.matmul(bias_gradient.unsqueeze(0), weight_gradient).squeeze(0) / energy
    reconstructed_weight_gradient = bias_gradient.unsqueeze(1) * feature.unsqueeze(0)
    residual = float(
        (weight_gradient - reconstructed_weight_gradient).norm()
        / (weight_gradient.norm() + epsilon)
    )
    return LinearHeadLeakageReport(
        module_name=module_name,
        effective_samples=1,
        feature_dimension=int(feature.numel()),
        recovered_feature=feature.detach().clone(),
        bias_gradient_norm=float(bias_gradient.norm()),
        rank_one_relative_residual=residual,
    )


def capture_final_linear_input(
    model: nn.Module,
    candidate: torch.Tensor,
) -> torch.Tensor:
    """Run a candidate forward pass and capture the input tensor of the final Linear."""

    _, linear = find_last_biased_linear(model)
    captured: list[torch.Tensor] = []

    def pre_hook(_module: nn.Module, args: tuple[torch.Tensor, ...]) -> None:
        if not args:
            raise RuntimeError("final linear hook received no positional input")
        captured.append(args[0])

    handle = linear.register_forward_pre_hook(pre_hook)
    try:
        model(candidate)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(
            "the final biased Linear must be invoked exactly once per model forward"
        )
    return captured[0]


class HeadRepresentationInversionAttack:
    """Invert a private input by matching an analytically leaked head representation.

    This baseline is first-order in the victim encoder: unlike full gradient
    matching it does not differentiate through parameter gradients. It is a strong
    classical opponent for single-record, single-variate Transformer/PatchTST
    experiments and must be evaluated before claiming a quantum search advantage.
    """

    def __init__(
        self,
        model: nn.Module,
        observed_gradients: tuple[torch.Tensor, ...],
        prior: nn.Module,
        *,
        mode: str = "timeseries",
        effective_samples: int = 1,
        steps: int = 300,
        learning_rate: float = 0.05,
        regularization: float = 0.0,
        gradient_clip_norm: float | None = None,
        record_every: int | None = None,
        callback: Callable[[int, torch.Tensor], None] | None = None,
    ) -> None:
        if steps <= 0:
            raise ValueError("steps must be positive")
        if not math.isfinite(float(learning_rate)) or learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(float(regularization)) or regularization < 0.0:
            raise ValueError("regularization must be finite and non-negative")
        if gradient_clip_norm is not None and (
            not math.isfinite(float(gradient_clip_norm)) or gradient_clip_norm <= 0.0
        ):
            raise ValueError("gradient_clip_norm must be finite and positive")
        if record_every is not None and record_every <= 0:
            raise ValueError("record_every must be positive")
        self.model = model
        self.prior = prior
        self.mode = mode
        self.steps = int(steps)
        self.learning_rate = float(learning_rate)
        self.regularization = float(regularization)
        self.gradient_clip_norm = gradient_clip_norm
        self.record_every = record_every
        self.callback = callback
        self.leakage = recover_single_effective_head_input(
            model,
            observed_gradients,
            effective_samples=effective_samples,
        )

    def _representation(self, candidate: torch.Tensor) -> torch.Tensor:
        captured = capture_final_linear_input(self.model, candidate)
        flattened = captured.reshape(-1, captured.shape[-1])
        if flattened.shape[0] != 1:
            raise ValueError(
                "candidate forward produced multiple effective final-head samples"
            )
        return flattened[0]

    def run(self) -> HeadRepresentationAttackResult:
        parameters = list(self.prior.parameters())
        if not parameters:
            raise ValueError("the reconstruction prior must expose trainable parameters")
        optimizer = torch.optim.Adam(parameters, lr=self.learning_rate)
        target_feature = self.leakage.recovered_feature.to(parameters[0].device)
        target_energy = target_feature.square().sum().clamp_min(1e-12)
        record_every = self.record_every or max(1, self.steps // 20)
        history: list[dict[str, float]] = []
        best_objective = math.inf
        best_representation_loss = math.inf
        best_step = 0
        best_reconstruction: torch.Tensor | None = None

        def evaluate() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            candidate = self.prior()
            feature = self._representation(candidate)
            if feature.shape != target_feature.shape:
                raise ValueError("candidate and leaked head representations differ in shape")
            representation_loss = (feature - target_feature).square().sum() / target_energy
            prior_penalty = _regularizer(candidate, self.mode)
            objective = representation_loss + self.regularization * prior_penalty
            return objective, representation_loss, prior_penalty, candidate

        for step in range(1, self.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            for parameter in self.model.parameters():
                parameter.grad = None
            objective, representation_loss, prior_penalty, candidate = evaluate()
            objective_value = float(objective.detach())
            representation_value = float(representation_loss.detach())
            if not math.isfinite(objective_value):
                raise FloatingPointError("head-representation inversion became non-finite")
            if objective_value < best_objective:
                best_objective = objective_value
                best_representation_loss = representation_value
                best_step = step
                best_reconstruction = candidate.detach().clone()
            objective.backward()
            if self.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(parameters, self.gradient_clip_norm)
            optimizer.step()
            if step == 1 or step % record_every == 0 or step == self.steps:
                history.append(
                    {
                        "step": float(step),
                        "objective": objective_value,
                        "representation_loss": representation_value,
                        "regularizer": float(prior_penalty.detach()),
                        "best_objective": best_objective,
                    }
                )
                if self.callback is not None:
                    self.callback(step, candidate.detach())

        for parameter in self.model.parameters():
            parameter.grad = None
        final_objective_tensor, final_representation_tensor, _, final_candidate = evaluate()
        final_objective = float(final_objective_tensor.detach())
        final_representation = float(final_representation_tensor.detach())
        if final_objective < best_objective:
            best_objective = final_objective
            best_representation_loss = final_representation
            best_step = self.steps + 1
            best_reconstruction = final_candidate.detach().clone()
        if best_reconstruction is None:
            raise RuntimeError("head-representation inversion produced no finite iterate")
        return HeadRepresentationAttackResult(
            reconstruction=best_reconstruction,
            recovered_feature=target_feature.detach().clone(),
            history=tuple(history),
            best_objective=best_objective,
            best_representation_loss=best_representation_loss,
            best_step=best_step,
            final_objective=final_objective,
            final_representation_loss=final_representation,
            leakage=self.leakage,
        )
