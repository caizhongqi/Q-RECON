from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F


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
    candidate: tuple[torch.Tensor, ...], observed: tuple[torch.Tensor, ...]
) -> torch.Tensor:
    relative = torch.zeros((), device=candidate[0].device)
    cosine_num = torch.zeros((), device=candidate[0].device)
    candidate_norm = torch.zeros((), device=candidate[0].device)
    observed_norm = torch.zeros((), device=candidate[0].device)
    for left, right in zip(candidate, observed):
        relative = relative + (left - right).square().sum() / (
            right.square().sum() + 1e-12
        )
        cosine_num = cosine_num + (left * right).sum()
        candidate_norm = candidate_norm + left.square().sum()
        observed_norm = observed_norm + right.square().sum()
    cosine = cosine_num / (
        candidate_norm.sqrt() * observed_norm.sqrt() + 1e-12
    )
    return relative / max(len(candidate), 1) + 0.1 * (1.0 - cosine)


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
    return torch.zeros((), device=x.device)


@dataclass
class AttackResult:
    reconstruction: torch.Tensor
    reconstructed_target: torch.Tensor
    history: list[dict[str, float]]


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
        callback: Callable[[int, torch.Tensor], None] | None = None,
    ):
        self.model = model
        self.observed = observed_gradients
        self.prior = prior
        self.task = task
        self.mode = mode
        self.known_target = known_target
        self.target_shape = target_shape
        self.steps = steps
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.optimizer_name = optimizer_name
        self.callback = callback

    def run(self) -> AttackResult:
        parameters = list(self.prior.parameters())
        target_parameter: nn.Parameter | None = None
        if self.known_target is None:
            if self.task == "classification":
                raise ValueError(
                    "classification attacks currently require a known/inferred label"
                )
            target_parameter = nn.Parameter(torch.zeros(self.target_shape))
            parameters.append(target_parameter)
        history: list[dict[str, float]] = []

        def objective_and_parts() -> tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ]:
            x_hat = self.prior()
            target_hat = (
                self.known_target if target_parameter is None else target_parameter
            )
            dummy_loss = _task_loss(self.model(x_hat), target_hat, self.task)
            dummy_gradients = torch.autograd.grad(
                dummy_loss, tuple(self.model.parameters()), create_graph=True
            )
            match = gradient_matching_loss(dummy_gradients, self.observed)
            prior_penalty = _regularizer(x_hat, self.mode)
            objective = match + self.regularization * prior_penalty
            return objective, match, prior_penalty, x_hat

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
                objective, match, prior_penalty, x_hat = objective_and_parts()
                objective.backward()
                closure_calls += 1
                if closure_calls == 1 or closure_calls % max(
                    1, self.steps // 20
                ) == 0:
                    history.append(
                        {
                            "step": float(closure_calls),
                            "objective": float(objective.detach()),
                            "gradient_match": float(match.detach()),
                            "regularizer": float(prior_penalty.detach()),
                        }
                    )
                    if self.callback is not None:
                        self.callback(closure_calls, x_hat.detach())
                return objective

            optimizer.step(closure)
        elif self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(parameters, lr=self.learning_rate)
            for step in range(self.steps):
                optimizer.zero_grad(set_to_none=True)
                objective, match, prior_penalty, x_hat = objective_and_parts()
                objective.backward()
                optimizer.step()

                if step == 0 or (step + 1) % max(1, self.steps // 20) == 0:
                    record = {
                        "step": float(step + 1),
                        "objective": float(objective.detach()),
                        "gradient_match": float(match.detach()),
                        "regularizer": float(prior_penalty.detach()),
                    }
                    history.append(record)
                    if self.callback is not None:
                        self.callback(step + 1, x_hat.detach())
        else:
            raise ValueError(f"unknown optimizer: {self.optimizer_name}")

        final_target = (
            self.known_target
            if target_parameter is None
            else target_parameter.detach()
        )
        return AttackResult(self.prior().detach(), final_target.detach(), history)
