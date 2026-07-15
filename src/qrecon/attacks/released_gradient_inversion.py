from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from .gradient_inversion import (
    AttackResult,
    GradientMatchMode,
    LayerWeighting,
    _regularizer,
    _task_loss,
    gradient_matching_loss,
)
from .gradient_release import (
    GradientRelease,
    GradientReleaseSpec,
    clip_gradient_tuple,
    quantize_gradient_tuple,
)


@dataclass(frozen=True)
class CandidateReleaseTransformReport:
    visible_parameter_indices: tuple[int, ...]
    clip_norm: float | None
    quantization_bits: int | None
    quantization_scale: float | None
    quantization_straight_through: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "visible_parameter_indices": list(self.visible_parameter_indices),
            "clip_norm": self.clip_norm,
            "quantization_bits": self.quantization_bits,
            "quantization_scale": self.quantization_scale,
            "quantization_straight_through": self.quantization_straight_through,
        }


class ReleasedGradientInversionAttack:
    """Gradient inversion matched to a clipped/quantized/partial release contract.

    Candidate gradients pass through the known deterministic release operations:
    global clipping, public-scale quantization and parameter selection. Gaussian
    noise is not reproduced because its realization is not available to the
    attacker. Quantization uses a declared straight-through estimator by default;
    this is an optimization surrogate and is reported explicitly.
    """

    def __init__(
        self,
        model: nn.Module,
        release: GradientRelease,
        release_spec: GradientReleaseSpec,
        prior: nn.Module,
        *,
        task: str,
        mode: str,
        known_target: torch.Tensor | None,
        target_shape: tuple[int, ...],
        steps: int = 300,
        learning_rate: float = 0.05,
        regularization: float = 1e-3,
        match_mode: GradientMatchMode = "hybrid",
        layer_weighting: LayerWeighting = "parameter",
        gradient_clip_norm: float | None = None,
        quantization_straight_through: bool = True,
        record_every: int | None = None,
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
        if release.visible_parameter_indices != (
            tuple(range(len(tuple(model.parameters()))))
            if release_spec.visible_parameter_indices is None
            else tuple(release_spec.visible_parameter_indices)
        ):
            raise ValueError("release and release_spec visible parameter indices differ")
        if release.quantization_bits != release_spec.quantization_bits:
            raise ValueError("release and release_spec quantization declarations differ")

        self.model = model
        self.release = release
        self.release_spec = release_spec
        self.prior = prior
        self.task = task
        self.mode = mode
        self.known_target = known_target
        self.target_shape = target_shape
        self.steps = int(steps)
        self.learning_rate = float(learning_rate)
        self.regularization = float(regularization)
        self.match_mode = match_mode
        self.layer_weighting = layer_weighting
        self.gradient_clip_norm = gradient_clip_norm
        self.quantization_straight_through = bool(
            quantization_straight_through
        )
        self.record_every = record_every
        self.transform_report = CandidateReleaseTransformReport(
            visible_parameter_indices=release.visible_parameter_indices,
            clip_norm=release_spec.clip_norm,
            quantization_bits=release.quantization_bits,
            quantization_scale=release.quantization_scale,
            quantization_straight_through=self.quantization_straight_through,
        )

    def _candidate_release(
        self,
        gradients: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        transformed, _ = clip_gradient_tuple(
            gradients,
            self.release_spec.clip_norm,
        )
        if self.release.quantization_bits is not None:
            assert self.release.quantization_scale is not None
            transformed, _, _, _ = quantize_gradient_tuple(
                transformed,
                self.release.quantization_bits,
                scale=self.release.quantization_scale,
                straight_through=self.quantization_straight_through,
            )
        return tuple(
            transformed[index] for index in self.release.visible_parameter_indices
        )

    def run(self) -> AttackResult:
        parameters = list(self.prior.parameters())
        if not parameters:
            raise ValueError("the reconstruction prior must expose trainable parameters")
        target_parameter: nn.Parameter | None = None
        if self.known_target is None:
            if self.task == "classification":
                raise ValueError("classification release attacks require a known label")
            target_parameter = nn.Parameter(torch.zeros(self.target_shape))
            parameters.append(target_parameter)

        optimizer = torch.optim.Adam(parameters, lr=self.learning_rate)
        record_every = self.record_every or max(1, self.steps // 20)
        history: list[dict[str, float]] = []
        best_objective = math.inf
        best_match = math.inf
        best_step = 0
        best_reconstruction: torch.Tensor | None = None
        best_target: torch.Tensor | None = None

        def evaluate() -> tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ]:
            candidate = self.prior()
            target = (
                self.known_target if target_parameter is None else target_parameter
            )
            loss = _task_loss(self.model(candidate), target, self.task)
            exact_candidate = torch.autograd.grad(
                loss,
                tuple(self.model.parameters()),
                create_graph=True,
            )
            released_candidate = self._candidate_release(exact_candidate)
            match = gradient_matching_loss(
                released_candidate,
                self.release.gradients,
                mode=self.match_mode,
                layer_weighting=self.layer_weighting,
            )
            prior_penalty = _regularizer(candidate, self.mode)
            objective = match + self.regularization * prior_penalty
            return objective, match, prior_penalty, candidate, target

        for step in range(1, self.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            for parameter in self.model.parameters():
                parameter.grad = None
            objective, match, prior_penalty, candidate, target = evaluate()
            objective_value = float(objective.detach())
            match_value = float(match.detach())
            if not math.isfinite(objective_value):
                raise FloatingPointError("release-aware inversion became non-finite")
            if objective_value < best_objective:
                best_objective = objective_value
                best_match = match_value
                best_step = step
                best_reconstruction = candidate.detach().clone()
                best_target = target.detach().clone()
            objective.backward()
            if self.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(parameters, self.gradient_clip_norm)
            optimizer.step()
            if step == 1 or step % record_every == 0 or step == self.steps:
                history.append(
                    {
                        "step": float(step),
                        "objective": objective_value,
                        "gradient_match": match_value,
                        "regularizer": float(prior_penalty.detach()),
                        "best_objective": best_objective,
                        "best_gradient_match": best_match,
                    }
                )

        for parameter in self.model.parameters():
            parameter.grad = None
        final_objective_tensor, final_match_tensor, _, final_candidate, final_target = (
            evaluate()
        )
        final_objective = float(final_objective_tensor.detach())
        final_match = float(final_match_tensor.detach())
        if final_objective < best_objective:
            best_objective = final_objective
            best_match = final_match
            best_step = self.steps + 1
            best_reconstruction = final_candidate.detach().clone()
            best_target = final_target.detach().clone()
        if best_reconstruction is None or best_target is None:
            raise RuntimeError("release-aware inversion produced no finite iterate")
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
