from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from typing import Literal, Sequence

import torch

from .channel_permutation import (
    ChannelPermutationFibreBound,
    apply_channel_permutation,
    tensor_channel_permutation_fibre_bound,
    validate_channel_permutation,
)

OptimizerName = Literal["sgd", "momentum", "adam", "adamw"]


@dataclass(frozen=True)
class TensorTupleDifference:
    maximum_absolute_difference: float
    relative_l2_difference: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class ChannelPermutationTrainingStep:
    step: int
    loss_absolute_difference: float
    gradient_difference: TensorTupleDifference
    parameter_difference: TensorTupleDifference
    optimizer_state_difference: TensorTupleDifference

    def to_dict(self) -> dict[str, object]:
        return {
            "step": self.step,
            "loss_absolute_difference": self.loss_absolute_difference,
            "gradient_difference": self.gradient_difference.to_dict(),
            "parameter_difference": self.parameter_difference.to_dict(),
            "optimizer_state_difference": self.optimizer_state_difference.to_dict(),
        }


@dataclass(frozen=True)
class ChannelPermutationTrainingTranscriptWitness:
    permutation: tuple[int, ...]
    fibre_bound: ChannelPermutationFibreBound
    optimizer: str
    steps: int
    records: tuple[ChannelPermutationTrainingStep, ...]
    final_model_delta_difference: TensorTupleDifference

    @property
    def maximum_loss_absolute_difference(self) -> float:
        return max((record.loss_absolute_difference for record in self.records), default=0.0)

    @property
    def maximum_gradient_absolute_difference(self) -> float:
        return max(
            (record.gradient_difference.maximum_absolute_difference for record in self.records),
            default=0.0,
        )

    @property
    def maximum_parameter_absolute_difference(self) -> float:
        return max(
            (record.parameter_difference.maximum_absolute_difference for record in self.records),
            default=0.0,
        )

    @property
    def maximum_optimizer_state_absolute_difference(self) -> float:
        return max(
            (
                record.optimizer_state_difference.maximum_absolute_difference
                for record in self.records
            ),
            default=0.0,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "permutation": list(self.permutation),
            "fibre_bound": self.fibre_bound.to_dict(),
            "optimizer": self.optimizer,
            "steps": self.steps,
            "records": [record.to_dict() for record in self.records],
            "final_model_delta_difference": self.final_model_delta_difference.to_dict(),
            "maximum_loss_absolute_difference": self.maximum_loss_absolute_difference,
            "maximum_gradient_absolute_difference": self.maximum_gradient_absolute_difference,
            "maximum_parameter_absolute_difference": self.maximum_parameter_absolute_difference,
            "maximum_optimizer_state_absolute_difference": (
                self.maximum_optimizer_state_absolute_difference
            ),
        }


def _tuple_difference(
    left: Sequence[torch.Tensor], right: Sequence[torch.Tensor]
) -> TensorTupleDifference:
    if len(left) != len(right):
        raise ValueError("tensor tuples must have equal length")
    maximum = 0.0
    numerator = 0.0
    denominator = 0.0
    for first, second in zip(left, right):
        if first.shape != second.shape:
            raise ValueError("corresponding tensors must have equal shapes")
        first_value = first.detach().double()
        second_value = second.detach().double()
        difference = first_value - second_value
        maximum = max(maximum, float(difference.abs().max()))
        numerator += float(difference.square().sum())
        denominator += float(first_value.square().sum())
    relative = math.sqrt(numerator) / max(
        math.sqrt(denominator), torch.finfo(torch.float64).tiny
    )
    return TensorTupleDifference(maximum, relative)


def _optimizer_tensor_state(optimizer: torch.optim.Optimizer) -> tuple[torch.Tensor, ...]:
    """Flatten tensor-valued optimizer state in parameter and key order."""

    values: list[torch.Tensor] = []
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            state = optimizer.state.get(parameter, {})
            for key in sorted(state, key=str):
                value = state[key]
                if torch.is_tensor(value):
                    values.append(value)
    return tuple(values)


def _build_optimizer(
    name: OptimizerName,
    parameters: Sequence[torch.nn.Parameter],
    *,
    learning_rate: float,
    weight_decay: float,
    momentum: float,
) -> torch.optim.Optimizer:
    normalized = str(name).lower()
    if not math.isfinite(float(learning_rate)) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be finite and positive")
    if not math.isfinite(float(weight_decay)) or weight_decay < 0.0:
        raise ValueError("weight_decay must be finite and non-negative")
    if not math.isfinite(float(momentum)) or not 0.0 <= momentum < 1.0:
        raise ValueError("momentum must lie in [0, 1)")
    if normalized == "sgd":
        return torch.optim.SGD(parameters, lr=learning_rate, weight_decay=weight_decay)
    if normalized == "momentum":
        return torch.optim.SGD(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum,
        )
    if normalized == "adam":
        return torch.optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
    if normalized == "adamw":
        return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    raise ValueError("optimizer must be one of: sgd, momentum, adam, adamw")


def channel_permutation_training_transcript_witness(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    permutation: Sequence[int],
    *,
    optimizer: OptimizerName = "adamw",
    steps: int = 5,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    momentum: float = 0.9,
) -> ChannelPermutationTrainingTranscriptWitness:
    """Certify equality of deterministic gradient-based training transcripts.

    If the loss is identical under simultaneous channel permutation for every model
    parameter value, deterministic first-order optimizer trajectories are identical
    by induction from a shared initialization and optimizer state. The resulting
    gradients, momentum/Adam state, checkpoints, and final model delta therefore
    reveal no additional channel-order information beyond the one-step gradient.

    The executable witness uses full-batch MSE and one of SGD, momentum SGD, Adam,
    or AdamW. The model must have deterministic forward/backward execution for the
    declared batch (for example, dropout disabled or common randomness explicitly
    coupled). Small floating-point residuals can occur because channel permutation
    changes reduction order.
    """

    if inputs.ndim != 3 or targets.ndim != 3:
        raise ValueError("inputs and targets must be multivariate three-dimensional tensors")
    if inputs.shape[0] != targets.shape[0] or inputs.shape[-1] != targets.shape[-1]:
        raise ValueError("input and target batch/channel dimensions must match")
    count = int(steps)
    if count <= 0:
        raise ValueError("steps must be positive")
    values = validate_channel_permutation(permutation, int(inputs.shape[-1]))
    permuted_inputs = apply_channel_permutation(inputs, values)
    permuted_targets = apply_channel_permutation(targets, values)

    left_model = copy.deepcopy(model)
    right_model = copy.deepcopy(model)
    left_model.train()
    right_model.train()
    left_parameters = tuple(
        parameter for parameter in left_model.parameters() if parameter.requires_grad
    )
    right_parameters = tuple(
        parameter for parameter in right_model.parameters() if parameter.requires_grad
    )
    if not left_parameters or len(left_parameters) != len(right_parameters):
        raise ValueError("model copies must expose matching trainable parameters")
    initial_parameters = tuple(parameter.detach().clone() for parameter in left_parameters)
    left_optimizer = _build_optimizer(
        optimizer,
        left_parameters,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        momentum=momentum,
    )
    right_optimizer = _build_optimizer(
        optimizer,
        right_parameters,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        momentum=momentum,
    )

    records: list[ChannelPermutationTrainingStep] = []
    for step in range(1, count + 1):
        left_optimizer.zero_grad(set_to_none=True)
        right_optimizer.zero_grad(set_to_none=True)
        left_loss = (left_model(inputs) - targets).square().mean()
        right_loss = (right_model(permuted_inputs) - permuted_targets).square().mean()
        left_loss.backward()
        right_loss.backward()
        left_gradients = tuple(
            torch.zeros_like(parameter)
            if parameter.grad is None
            else parameter.grad.detach().clone()
            for parameter in left_parameters
        )
        right_gradients = tuple(
            torch.zeros_like(parameter)
            if parameter.grad is None
            else parameter.grad.detach().clone()
            for parameter in right_parameters
        )
        gradient_difference = _tuple_difference(left_gradients, right_gradients)
        left_optimizer.step()
        right_optimizer.step()
        parameter_difference = _tuple_difference(left_parameters, right_parameters)
        left_state = _optimizer_tensor_state(left_optimizer)
        right_state = _optimizer_tensor_state(right_optimizer)
        optimizer_difference = _tuple_difference(left_state, right_state)
        records.append(
            ChannelPermutationTrainingStep(
                step=step,
                loss_absolute_difference=float((left_loss - right_loss).detach().abs()),
                gradient_difference=gradient_difference,
                parameter_difference=parameter_difference,
                optimizer_state_difference=optimizer_difference,
            )
        )

    left_delta = tuple(
        parameter.detach() - initial
        for parameter, initial in zip(left_parameters, initial_parameters)
    )
    right_delta = tuple(
        parameter.detach() - initial
        for parameter, initial in zip(right_parameters, initial_parameters)
    )
    return ChannelPermutationTrainingTranscriptWitness(
        permutation=values,
        fibre_bound=tensor_channel_permutation_fibre_bound(inputs, targets),
        optimizer=str(optimizer),
        steps=count,
        records=tuple(records),
        final_model_delta_difference=_tuple_difference(left_delta, right_delta),
    )
