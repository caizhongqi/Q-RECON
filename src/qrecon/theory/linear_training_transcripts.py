from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

from .known_target_collisions import (
    LinearGradientOracleStatistics,
    LinearGradientOracleValue,
    evaluate_linear_gradient_oracle_from_statistics,
    linear_gradient_oracle_statistics,
)


ArrayLike = np.ndarray | Sequence[Sequence[float]]
OptimizerName = Literal["sgd", "momentum", "adam"]


def _matrix(name: str, value: ArrayLike) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _vector(name: str, value: np.ndarray | Sequence[float], length: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},)")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


@dataclass(frozen=True)
class LinearOptimizerConfig:
    optimizer: OptimizerName = "sgd"
    learning_rate: float = 0.05
    momentum: float = 0.9
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    weight_decay: float = 0.0
    decay_bias: bool = False

    def __post_init__(self) -> None:
        if self.optimizer not in ("sgd", "momentum", "adam"):
            raise ValueError("optimizer must be 'sgd', 'momentum', or 'adam'")
        finite = {
            "learning_rate": self.learning_rate,
            "momentum": self.momentum,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.epsilon,
            "weight_decay": self.weight_decay,
        }
        if not all(math.isfinite(float(value)) for value in finite.values()):
            raise ValueError("optimizer hyperparameters must be finite")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 <= self.momentum < 1.0:
            raise ValueError("momentum must lie in [0, 1)")
        if not 0.0 <= self.beta1 < 1.0 or not 0.0 <= self.beta2 < 1.0:
            raise ValueError("Adam beta values must lie in [0, 1)")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")


@dataclass(frozen=True)
class LinearTrainingSnapshot:
    step: int
    loss: float
    weights: np.ndarray
    bias: np.ndarray
    weight_gradient: np.ndarray
    bias_gradient: np.ndarray
    weight_first_moment: np.ndarray
    bias_first_moment: np.ndarray
    weight_second_moment: np.ndarray
    bias_second_moment: np.ndarray

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        for key in (
            "weights",
            "bias",
            "weight_gradient",
            "bias_gradient",
            "weight_first_moment",
            "bias_first_moment",
            "weight_second_moment",
            "bias_second_moment",
        ):
            result[key] = getattr(self, key).tolist()
        return result


@dataclass(frozen=True)
class LinearTrainingTranscript:
    config: LinearOptimizerConfig
    snapshots: tuple[LinearTrainingSnapshot, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "config": asdict(self.config),
            "snapshots": [snapshot.to_dict() for snapshot in self.snapshots],
        }


def evaluate_linear_loss_from_statistics(
    statistics: LinearGradientOracleStatistics,
    target_squared_norm: float,
    weights: ArrayLike,
    bias: np.ndarray | Sequence[float],
) -> float:
    """Evaluate mean half-squared loss using only oracle sufficient statistics."""

    theta = _matrix("weights", weights)
    expected = (statistics.output_dimension, statistics.input_dimension)
    if theta.shape != expected:
        raise ValueError(f"weights must have shape {expected}")
    b = _vector("bias", bias, statistics.output_dimension)
    target_norm = float(target_squared_norm)
    if not math.isfinite(target_norm) or target_norm < 0.0:
        raise ValueError("target_squared_norm must be finite and non-negative")

    quadratic = float(np.trace(theta @ statistics.input_gram @ theta.T))
    input_bias = 2.0 * float(b @ (theta @ statistics.input_sum))
    bias_square = statistics.batch_size * float(b @ b)
    target_cross = -2.0 * float(np.sum(theta * statistics.target_cross))
    target_bias = -2.0 * float(b @ statistics.target_sum)
    return (
        quadratic
        + input_bias
        + bias_square
        + target_cross
        + target_bias
        + target_norm
    ) / (2.0 * statistics.batch_size)


def simulate_linear_training_from_statistics(
    statistics: LinearGradientOracleStatistics,
    target_squared_norm: float,
    initial_weights: ArrayLike,
    initial_bias: np.ndarray | Sequence[float],
    *,
    steps: int,
    config: LinearOptimizerConfig = LinearOptimizerConfig(),
    additive_weight_noise: Sequence[np.ndarray] | None = None,
    additive_bias_noise: Sequence[np.ndarray] | None = None,
) -> LinearTrainingTranscript:
    """Simulate full-batch SGD, Momentum, or Adam from sufficient statistics.

    Optional additive gradient-noise sequences make the coupling assumption
    explicit. When two hidden batches share the same statistics and receive the
    same external noise sequence, their complete optimizer transcripts coincide.
    If noise is sampled independently of the data from the same law, the
    transcript distributions coincide.
    """

    count = int(steps)
    if count < 0:
        raise ValueError("steps must be non-negative")
    weights = _matrix("initial_weights", initial_weights).copy()
    expected = (statistics.output_dimension, statistics.input_dimension)
    if weights.shape != expected:
        raise ValueError(f"initial_weights must have shape {expected}")
    bias = _vector("initial_bias", initial_bias, statistics.output_dimension).copy()

    if additive_weight_noise is not None and len(additive_weight_noise) != count:
        raise ValueError("additive_weight_noise must contain one matrix per step")
    if additive_bias_noise is not None and len(additive_bias_noise) != count:
        raise ValueError("additive_bias_noise must contain one vector per step")

    weight_first = np.zeros_like(weights)
    bias_first = np.zeros_like(bias)
    weight_second = np.zeros_like(weights)
    bias_second = np.zeros_like(bias)
    snapshots: list[LinearTrainingSnapshot] = []

    for step in range(1, count + 1):
        gradient = evaluate_linear_gradient_oracle_from_statistics(
            statistics, weights, bias
        )
        weight_gradient = gradient.weight_gradient.copy()
        bias_gradient = gradient.bias_gradient.copy()
        if config.weight_decay:
            weight_gradient += config.weight_decay * weights
            if config.decay_bias:
                bias_gradient += config.weight_decay * bias
        if additive_weight_noise is not None:
            perturbation = _matrix(
                "additive weight noise", additive_weight_noise[step - 1]
            )
            if perturbation.shape != weights.shape:
                raise ValueError("additive weight noise has an invalid shape")
            weight_gradient += perturbation
        if additive_bias_noise is not None:
            bias_gradient += _vector(
                "additive bias noise",
                additive_bias_noise[step - 1],
                statistics.output_dimension,
            )

        if config.optimizer == "sgd":
            weight_first = weight_gradient.copy()
            bias_first = bias_gradient.copy()
            weights -= config.learning_rate * weight_gradient
            bias -= config.learning_rate * bias_gradient
        elif config.optimizer == "momentum":
            weight_first = config.momentum * weight_first + weight_gradient
            bias_first = config.momentum * bias_first + bias_gradient
            weights -= config.learning_rate * weight_first
            bias -= config.learning_rate * bias_first
        else:
            weight_first = (
                config.beta1 * weight_first
                + (1.0 - config.beta1) * weight_gradient
            )
            bias_first = (
                config.beta1 * bias_first
                + (1.0 - config.beta1) * bias_gradient
            )
            weight_second = (
                config.beta2 * weight_second
                + (1.0 - config.beta2) * np.square(weight_gradient)
            )
            bias_second = (
                config.beta2 * bias_second
                + (1.0 - config.beta2) * np.square(bias_gradient)
            )
            weight_hat = weight_first / (1.0 - config.beta1**step)
            bias_hat = bias_first / (1.0 - config.beta1**step)
            weight_variance = weight_second / (1.0 - config.beta2**step)
            bias_variance = bias_second / (1.0 - config.beta2**step)
            weights -= config.learning_rate * weight_hat / (
                np.sqrt(weight_variance) + config.epsilon
            )
            bias -= config.learning_rate * bias_hat / (
                np.sqrt(bias_variance) + config.epsilon
            )

        loss = evaluate_linear_loss_from_statistics(
            statistics, target_squared_norm, weights, bias
        )
        snapshots.append(
            LinearTrainingSnapshot(
                step=step,
                loss=loss,
                weights=weights.copy(),
                bias=bias.copy(),
                weight_gradient=weight_gradient.copy(),
                bias_gradient=bias_gradient.copy(),
                weight_first_moment=weight_first.copy(),
                bias_first_moment=bias_first.copy(),
                weight_second_moment=weight_second.copy(),
                bias_second_moment=bias_second.copy(),
            )
        )
    return LinearTrainingTranscript(config=config, snapshots=tuple(snapshots))


def simulate_linear_training(
    inputs: ArrayLike,
    targets: ArrayLike,
    initial_weights: ArrayLike,
    initial_bias: np.ndarray | Sequence[float],
    *,
    steps: int,
    config: LinearOptimizerConfig = LinearOptimizerConfig(),
    additive_weight_noise: Sequence[np.ndarray] | None = None,
    additive_bias_noise: Sequence[np.ndarray] | None = None,
) -> LinearTrainingTranscript:
    x = _matrix("inputs", inputs)
    y = _matrix("targets", targets)
    if x.shape[0] != y.shape[0]:
        raise ValueError("inputs and targets must share a batch size")
    return simulate_linear_training_from_statistics(
        linear_gradient_oracle_statistics(x, y),
        float(np.sum(np.square(y))),
        initial_weights,
        initial_bias,
        steps=steps,
        config=config,
        additive_weight_noise=additive_weight_noise,
        additive_bias_noise=additive_bias_noise,
    )


def maximum_training_transcript_difference(
    left: LinearTrainingTranscript, right: LinearTrainingTranscript
) -> float:
    if left.config != right.config or len(left.snapshots) != len(right.snapshots):
        return math.inf
    maximum = 0.0
    for left_snapshot, right_snapshot in zip(left.snapshots, right.snapshots):
        maximum = max(maximum, abs(left_snapshot.loss - right_snapshot.loss))
        for left_value, right_value in (
            (left_snapshot.weights, right_snapshot.weights),
            (left_snapshot.bias, right_snapshot.bias),
            (left_snapshot.weight_gradient, right_snapshot.weight_gradient),
            (left_snapshot.bias_gradient, right_snapshot.bias_gradient),
            (left_snapshot.weight_first_moment, right_snapshot.weight_first_moment),
            (left_snapshot.bias_first_moment, right_snapshot.bias_first_moment),
            (left_snapshot.weight_second_moment, right_snapshot.weight_second_moment),
            (left_snapshot.bias_second_moment, right_snapshot.bias_second_moment),
        ):
            maximum = max(
                maximum, float(np.max(np.abs(left_value - right_value), initial=0.0))
            )
    return maximum
