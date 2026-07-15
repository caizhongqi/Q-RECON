from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Literal

import torch
from torch import nn
from torch.nn import functional as F

GradientMatchMode = Literal["hybrid", "cosine", "l2", "normalized_l2"]
LayerWeighting = Literal["tensor", "parameter"]


def _task_loss(prediction: torch.Tensor, target: torch.Tensor, task: str) -> torch.Tensor:
    if task == "classification":
        return F.cross_entropy(prediction, target.long())
    if task == "forecasting":
        return F.mse_loss(prediction, target)
    raise ValueError(f"unknown task: {task}")


def leak_gradients(
    model: nn.Module,
    x: torch.Tensor,
    target: torch.Tensor,
    task: str,
) -> tuple[torch.Tensor, ...]:
    loss = _task_loss(model(x), target, task)
    gradients = torch.autograd.grad(loss, tuple(model.parameters()))
    return tuple(gradient.detach().clone() for gradient in gradients)


def gradient_matching_loss(
    candidate: tuple[torch.Tensor, ...],
    observed: tuple[torch.Tensor, ...],
    *,
    mode: GradientMatchMode = "hybrid",
    layer_weighting: LayerWeighting = "tensor",
    epsilon: float = 1e-12,
) -> torch.Tensor:
    """Compare candidate and released parameter gradients.

    ``hybrid`` preserves the historical Q-RECON objective: layer-normalized L2
    plus a small global cosine term. ``parameter`` weighting prevents a one-element
    bias tensor from receiving the same aggregate weight as a large matrix. The
    legacy behavior remains the default through ``layer_weighting='tensor'``.
    """

    if len(candidate) != len(observed) or not candidate:
        raise ValueError("candidate and observed gradients must have equal non-zero length")
    if mode not in ("hybrid", "cosine", "l2", "normalized_l2"):
        raise ValueError(f"unknown gradient matching mode: {mode}")
    if layer_weighting not in ("tensor", "parameter"):
        raise ValueError(f"unknown layer weighting: {layer_weighting}")
    if not math.isfinite(float(epsilon)) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")

    device = candidate[0].device
    relative = torch.zeros((), device=device)
    relative_weight = 0.0
    squared_error = torch.zeros((), device=device)
    cosine_num = torch.zeros((), device=device)
    candidate_norm = torch.zeros((), device=device)
    observed_norm = torch.zeros((), device=device)

    for left, right in zip(candidate, observed):
        if left.shape != right.shape:
            raise ValueError("candidate and observed gradient tensors must have equal shapes")
        weight = float(left.numel()) if layer_weighting == "parameter" else 1.0
        difference = (left - right).square().sum()
        right_energy = right.square().sum()
        relative = relative + weight * difference / (right_energy + epsilon)
        relative_weight += weight
        squared_error = squared_error + difference
        cosine_num = cosine_num + (left * right).sum()
        candidate_norm = candidate_norm + left.square().sum()
        observed_norm = observed_norm + right_energy

    normalized = relative / max(relative_weight, 1.0)
    global_l2 = squared_error / (observed_norm + epsilon)
    cosine_loss = 1.0 - cosine_num / (
        candidate_norm.sqrt() * observed_norm.sqrt() + epsilon
    )
    if mode == "cosine":
        return cosine_loss
    if mode == "l2":
        return global_l2
    if mode == "normalized_l2":
        return normalized
    return normalized + 0.1 * cosine_loss


def _regularizer(x: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "image":
        vertical = (x[..., 1:, :] - x[..., :-1, :]).abs().mean()
        horizontal = (x[..., :, 1:] - x[..., :, :-1]).abs().mean()
        return vertical + horizontal
    if mode == "timeseries":
        if x.ndim < 2 or x.shape[1] <= 1:
            return torch.zeros((), device=x.device, dtype=x.dtype)
        # Q-RECON uses [batch, time] and [batch, time, channels]. The temporal
        # axis is therefore dimension one in both univariate and multivariate
        # experiments; regularizing the last axis would incorrectly smooth
        # channels in a multivariate PatchTST/iTransformer attack.
        return (x[:, 1:, ...] - x[:, :-1, ...]).square().mean()
    return torch.zeros((), device=x.device, dtype=x.dtype)


@dataclass
class AttackResult:
    reconstruction: torch.Tensor
    reconstructed_target: torch.Tensor
    history: list[dict[str, float]]
    best_objective: float = math.inf
    best_gradient_match: float = math.inf
    best_step: int = 0
    final_objective: float = math.inf
    final_gradient_match: float = math.inf


class GradientInversionAttack:
    def __init__(
        self,
        model: nn.Module,
        observed_gradients: tuple[torch.Tensor, ...],
        prior: nn.Module,
        task: str,
        mode: str,
        known_target: torch.Tensor | None,
        target_shape: tuple[int, ...],
        steps: int = 300,
        learning_rate: float = 0.05,
        regularization: float = 1e-3,
        optimizer_name: str = "adam",
        match_mode: GradientMatchMode = "hybrid",
        layer_weighting: LayerWeighting = "tensor",
        gradient_clip_norm: float | None = None,
        record_every: int | None = None,
        callback: Callable[[int, torch.Tensor], None] | None = None,
    ):
        if steps <= 0:
            raise ValueError("steps must be positive")
        if not math.isfinite(float(learning_rate)) or learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(float(regularization)) or regularization < 0.0:
            raise ValueError("regularization must be finite and non-negative")
        if gradient_clip_norm is not None and (
            not math.isfinite(float(gradient_clip_norm)) or gradient_clip_norm <= 0.0
        ):
            raise ValueError("gradient_clip_norm must be finite and positive when provided")
        if record_every is not None and record_every <= 0:
            raise ValueError("record_every must be positive when provided")

        self.model = model
        self.observed = observed_gradients
        self.prior = prior
        self.task = task
        self.mode = mode
        self.known_target = known_target
        self.target_shape = target_shape
        self.steps = int(steps)
        self.learning_rate = float(learning_rate)
        self.regularization = float(regularization)
        self.optimizer_name = str(optimizer_name).lower()
        self.match_mode = match_mode
        self.layer_weighting = layer_weighting
        self.gradient_clip_norm = gradient_clip_norm
        self.record_every = record_every
        self.callback = callback

    def run(self) -> AttackResult:
        parameters = list(self.prior.parameters())
        if not parameters:
            raise ValueError("the reconstruction prior must expose trainable parameters")
        target_parameter: nn.Parameter | None = None
        if self.known_target is None:
            if self.task == "classification":
                raise ValueError(
                    "classification attacks currently require a known/inferred label"
                )
            target_parameter = nn.Parameter(torch.zeros(self.target_shape))
            parameters.append(target_parameter)
        history: list[dict[str, float]] = []
        record_every = self.record_every or max(1, self.steps // 20)

        best_objective = math.inf
        best_match = math.inf
        best_step = 0
        best_reconstruction: torch.Tensor | None = None
        best_target: torch.Tensor | None = None

        def objective_and_parts() -> tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ]:
            x_hat = self.prior()
            target_hat = (
                self.known_target if target_parameter is None else target_parameter
            )
            dummy_loss = _task_loss(self.model(x_hat), target_hat, self.task)
            dummy_gradients = torch.autograd.grad(
                dummy_loss, tuple(self.model.parameters()), create_graph=True
            )
            match = gradient_matching_loss(
                dummy_gradients,
                self.observed,
                mode=self.match_mode,
                layer_weighting=self.layer_weighting,
            )
            prior_penalty = _regularizer(x_hat, self.mode)
            objective = match + self.regularization * prior_penalty
            return objective, match, prior_penalty, x_hat, target_hat

        def clear_model_gradients() -> None:
            for parameter in self.model.parameters():
                parameter.grad = None

        def update_best(
            step: int,
            objective: torch.Tensor,
            match: torch.Tensor,
            x_hat: torch.Tensor,
            target_hat: torch.Tensor,
        ) -> None:
            nonlocal best_objective, best_match, best_step
            nonlocal best_reconstruction, best_target
            objective_value = float(objective.detach())
            match_value = float(match.detach())
            if not math.isfinite(objective_value) or not math.isfinite(match_value):
                raise FloatingPointError("gradient inversion produced a non-finite objective")
            if objective_value < best_objective:
                best_objective = objective_value
                best_match = match_value
                best_step = int(step)
                best_reconstruction = x_hat.detach().clone()
                best_target = target_hat.detach().clone()

        def record(
            step: int,
            objective: torch.Tensor,
            match: torch.Tensor,
            prior_penalty: torch.Tensor,
            x_hat: torch.Tensor,
        ) -> None:
            history.append(
                {
                    "step": float(step),
                    "objective": float(objective.detach()),
                    "gradient_match": float(match.detach()),
                    "regularizer": float(prior_penalty.detach()),
                    "best_objective": float(best_objective),
                    "best_gradient_match": float(best_match),
                }
            )
            if self.callback is not None:
                self.callback(step, x_hat.detach())

        if self.optimizer_name == "lbfgs":
            optimizer = torch.optim.LBFGS(
                parameters,
                lr=self.learning_rate,
                max_iter=self.steps,
                history_size=min(100, self.steps),
                line_search_fn="strong_wolfe",
                tolerance_grad=1e-12,
                tolerance_change=1e-15,
            )
            closure_calls = 0

            def closure() -> torch.Tensor:
                nonlocal closure_calls
                optimizer.zero_grad(set_to_none=True)
                clear_model_gradients()
                objective, match, prior_penalty, x_hat, target_hat = objective_and_parts()
                closure_calls += 1
                update_best(closure_calls, objective, match, x_hat, target_hat)
                objective.backward()
                if self.gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(parameters, self.gradient_clip_norm)
                if closure_calls == 1 or closure_calls % record_every == 0:
                    record(closure_calls, objective, match, prior_penalty, x_hat)
                return objective

            optimizer.step(closure)
        elif self.optimizer_name in {"adam", "adamw"}:
            optimizer_class = (
                torch.optim.AdamW if self.optimizer_name == "adamw" else torch.optim.Adam
            )
            optimizer = optimizer_class(parameters, lr=self.learning_rate)
            for step in range(1, self.steps + 1):
                optimizer.zero_grad(set_to_none=True)
                clear_model_gradients()
                objective, match, prior_penalty, x_hat, target_hat = objective_and_parts()
                update_best(step, objective, match, x_hat, target_hat)
                objective.backward()
                if self.gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(parameters, self.gradient_clip_norm)
                optimizer.step()

                if step == 1 or step % record_every == 0 or step == self.steps:
                    record(step, objective, match, prior_penalty, x_hat)
        else:
            raise ValueError(f"unknown optimizer: {self.optimizer_name}")

        clear_model_gradients()
        final_objective_tensor, final_match_tensor, _, final_x, final_target = (
            objective_and_parts()
        )
        final_objective = float(final_objective_tensor.detach())
        final_match = float(final_match_tensor.detach())
        update_best(
            max(best_step, self.steps) + 1,
            final_objective_tensor,
            final_match_tensor,
            final_x,
            final_target,
        )
        clear_model_gradients()

        if best_reconstruction is None or best_target is None:
            raise RuntimeError("gradient inversion did not produce a finite candidate")
        return AttackResult(
            reconstruction=best_reconstruction,
            reconstructed_target=best_target,
            history=history,
            best_objective=best_objective,
            best_gradient_match=best_match,
            best_step=best_step,
            final_objective=final_objective,
            final_gradient_match=final_match,
        )
