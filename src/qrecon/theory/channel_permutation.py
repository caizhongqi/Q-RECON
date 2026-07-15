from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Hashable, Sequence

import torch


@dataclass(frozen=True)
class ChannelPermutationFibreBound:
    """Exact orbit size and uniform-prior recovery ceiling for labeled channels.

    ``multiplicities`` groups channels whose complete private records, including
    every released target coordinate, are identical. Permuting within an identical
    group does not create a distinct private object. The number of distinct ordered
    objects in the simultaneous channel-permutation orbit is therefore

    ``channels! / product_j multiplicities[j]!``.
    """

    channels: int
    multiplicities: tuple[int, ...]
    orbit_size: int
    uniform_exact_ordered_recovery_ceiling: float

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["multiplicities"] = list(self.multiplicities)
        payload["nontrivial"] = self.orbit_size > 1
        return payload


def channel_permutation_orbit_size(multiplicities: Sequence[int]) -> int:
    """Number of distinct permutations of a multiset of channel records."""

    counts = tuple(int(value) for value in multiplicities)
    if not counts or any(value <= 0 for value in counts):
        raise ValueError("multiplicities must be non-empty positive integers")
    channels = sum(counts)
    denominator = math.prod(math.factorial(value) for value in counts)
    return math.factorial(channels) // denominator


def channel_permutation_fibre_bound(
    channel_signatures: Sequence[Hashable],
) -> ChannelPermutationFibreBound:
    """Build the exact simultaneous channel-permutation orbit bound.

    The signatures must encode every private quantity whose channel identity is
    part of the recovery target, normally the concatenated input history and
    forecast target for each variable. Under a uniform prior on the orbit, no
    classical or quantum estimator observing a permutation-invariant channel can
    recover the original labeled ordering with probability above ``1/orbit_size``.
    """

    signatures = tuple(channel_signatures)
    if not signatures:
        raise ValueError("channel_signatures must be non-empty")
    multiplicities = tuple(sorted(Counter(signatures).values(), reverse=True))
    orbit = channel_permutation_orbit_size(multiplicities)
    return ChannelPermutationFibreBound(
        channels=len(signatures),
        multiplicities=multiplicities,
        orbit_size=orbit,
        uniform_exact_ordered_recovery_ceiling=1.0 / orbit,
    )


def _validate_timeseries_pair(
    inputs: torch.Tensor, targets: torch.Tensor
) -> tuple[int, int]:
    if inputs.ndim != 3 or targets.ndim != 3:
        raise ValueError(
            "channel-permutation analysis requires inputs [batch,time,channels] "
            "and targets [batch,horizon,channels]"
        )
    if inputs.shape[0] != targets.shape[0]:
        raise ValueError("input and target batch sizes must match")
    if inputs.shape[2] != targets.shape[2]:
        raise ValueError("input and target channel counts must match")
    if inputs.shape[2] < 2:
        raise ValueError("at least two channels are required")
    return int(inputs.shape[0]), int(inputs.shape[2])


def validate_channel_permutation(
    permutation: Sequence[int], channels: int
) -> tuple[int, ...]:
    values = tuple(int(value) for value in permutation)
    if len(values) != int(channels):
        raise ValueError(f"expected a permutation of {channels} channels")
    if set(values) != set(range(int(channels))):
        raise ValueError("channel permutation must contain every channel exactly once")
    return values


def apply_channel_permutation(
    tensor: torch.Tensor, permutation: Sequence[int]
) -> torch.Tensor:
    if tensor.ndim < 1:
        raise ValueError("tensor must have a channel axis")
    values = validate_channel_permutation(permutation, int(tensor.shape[-1]))
    indices = torch.tensor(values, dtype=torch.long, device=tensor.device)
    return tensor.index_select(-1, indices)


def _private_channel_signature(
    inputs: torch.Tensor, targets: torch.Tensor, channel: int
) -> bytes:
    history = inputs[..., channel].detach().cpu().contiguous()
    future = targets[..., channel].detach().cpu().contiguous()
    header = (
        f"{history.dtype}:{tuple(history.shape)}:"
        f"{future.dtype}:{tuple(future.shape)}:"
    ).encode("ascii")
    return (
        header
        + history.numpy().tobytes(order="C")
        + future.numpy().tobytes(order="C")
    )


def tensor_channel_permutation_fibre_bound(
    inputs: torch.Tensor, targets: torch.Tensor
) -> ChannelPermutationFibreBound:
    """Exact labeled-channel orbit for a concrete multivariate training batch."""

    _, channels = _validate_timeseries_pair(inputs, targets)
    signatures = tuple(
        _private_channel_signature(inputs, targets, channel)
        for channel in range(channels)
    )
    return channel_permutation_fibre_bound(signatures)


@dataclass(frozen=True)
class ChannelPermutationGradientWitness:
    """Numerical witness for a nonlinear full-gradient collision."""

    permutation: tuple[int, ...]
    fibre_bound: ChannelPermutationFibreBound
    input_displacement_l2: float
    target_displacement_l2: float
    prediction_equivariance_max_abs_error: float
    loss_absolute_difference: float
    gradient_max_abs_difference: float
    gradient_relative_l2_difference: float
    parameter_tensors: int

    @property
    def nontrivial_private_collision(self) -> bool:
        return (
            self.fibre_bound.orbit_size > 1
            and self.input_displacement_l2 > 0.0
            and self.target_displacement_l2 > 0.0
        )

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["permutation"] = list(self.permutation)
        payload["fibre_bound"] = self.fibre_bound.to_dict()
        payload["nontrivial_private_collision"] = self.nontrivial_private_collision
        return payload


def channel_permutation_gradient_witness(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    permutation: Sequence[int],
) -> ChannelPermutationGradientWitness:
    """Audit the channel-permutation full-gradient invariance theorem.

    Suppose a forecasting model is channel-permutation equivariant,

    ``f_theta(X P) = f_theta(X) P``,

    and training uses mean squared error over every output entry. Then

    ``L(theta; X P, Y P) = L(theta; X, Y)``

    is an identity in ``theta``. Differentiating proves equality of every model
    parameter gradient, so the released full gradient cannot identify the original
    labeled channel ordering. The information-theoretic exact-order recovery ceiling
    under a uniform prior on the orbit is ``1 / orbit_size`` for both classical and
    quantum estimators.

    This theorem requires the declared model to contain no channel-indexed learned
    parameters or metadata. In Q-RECON it applies to iTransformer with ``revin=False``
    and to channel-independent PatchTST with a shared head and ``revin=False``. It
    does not apply unchanged to channel-specific heads, channel embeddings, or
    per-channel affine RevIN parameters.
    """

    _, channels = _validate_timeseries_pair(inputs, targets)
    values = validate_channel_permutation(permutation, channels)
    permuted_inputs = apply_channel_permutation(inputs, values)
    permuted_targets = apply_channel_permutation(targets, values)
    parameters = tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    if not parameters:
        raise ValueError("model has no trainable parameters")

    was_training = model.training
    model.eval()
    try:
        prediction = model(inputs)
        permuted_prediction = model(permuted_inputs)
        expected_prediction = apply_channel_permutation(prediction, values)
        if prediction.shape != targets.shape:
            raise ValueError(
                f"model output shape {tuple(prediction.shape)} does not match "
                f"target shape {tuple(targets.shape)}"
            )
        prediction_error = float(
            (permuted_prediction - expected_prediction).detach().abs().max()
        )
        loss = (prediction - targets).square().mean()
        permuted_loss = (permuted_prediction - permuted_targets).square().mean()
        gradients = torch.autograd.grad(
            loss, parameters, allow_unused=True, retain_graph=False
        )
        permuted_gradients = torch.autograd.grad(
            permuted_loss, parameters, allow_unused=True, retain_graph=False
        )
    finally:
        model.train(was_training)

    maximum = 0.0
    squared_difference = 0.0
    squared_reference = 0.0
    for parameter, left, right in zip(parameters, gradients, permuted_gradients):
        left_value = torch.zeros_like(parameter) if left is None else left
        right_value = torch.zeros_like(parameter) if right is None else right
        difference = (left_value - right_value).detach().double()
        maximum = max(maximum, float(difference.abs().max()))
        squared_difference += float(difference.square().sum())
        squared_reference += float(left_value.detach().double().square().sum())
    relative = math.sqrt(squared_difference) / max(
        math.sqrt(squared_reference), torch.finfo(torch.float64).tiny
    )

    return ChannelPermutationGradientWitness(
        permutation=values,
        fibre_bound=tensor_channel_permutation_fibre_bound(inputs, targets),
        input_displacement_l2=float(
            (permuted_inputs - inputs).detach().double().norm()
        ),
        target_displacement_l2=float(
            (permuted_targets - targets).detach().double().norm()
        ),
        prediction_equivariance_max_abs_error=prediction_error,
        loss_absolute_difference=float((loss - permuted_loss).detach().abs()),
        gradient_max_abs_difference=maximum,
        gradient_relative_l2_difference=relative,
        parameter_tensors=len(parameters),
    )
