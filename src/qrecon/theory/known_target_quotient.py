from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .known_target_collisions import (
    LinearGradientOracleStatistics,
    LinearGradientOracleValue,
    evaluate_linear_gradient_oracle_from_statistics,
    linear_gradient_oracle_statistics,
    target_constraint_matrix,
    target_stabilizer_basis,
)


ArrayLike = np.ndarray | Sequence[Sequence[float]]


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


def _rank(matrix: np.ndarray, tolerance: float) -> int:
    singular = np.linalg.svd(matrix, compute_uv=False)
    return int(np.count_nonzero(singular > tolerance))


def _validate_statistics(
    statistics: LinearGradientOracleStatistics,
    targets: np.ndarray,
    *,
    atol: float,
) -> None:
    if statistics.batch_size != targets.shape[0]:
        raise ValueError("statistics and targets must have the same batch size")
    if statistics.output_dimension != targets.shape[1]:
        raise ValueError("statistics and targets must have the same output dimension")
    d = statistics.input_dimension
    c = statistics.output_dimension
    expected_shapes = (
        ("input_gram", statistics.input_gram, (d, d)),
        ("input_sum", statistics.input_sum, (d,)),
        ("target_cross", statistics.target_cross, (c, d)),
        ("target_sum", statistics.target_sum, (c,)),
    )
    for name, value, shape in expected_shapes:
        array = np.asarray(value, dtype=np.float64)
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError(f"statistics.{name} must be finite with shape {shape}")
    if not np.allclose(statistics.target_sum, targets.sum(axis=0), atol=atol, rtol=0.0):
        raise ValueError("statistics.target_sum is inconsistent with the known targets")


@dataclass(frozen=True)
class LinearGradientOracleProbeRecovery:
    statistics: LinearGradientOracleStatistics
    query_count: int
    symmetry_error: float
    reproduction_error: float

    def to_dict(self) -> dict[str, object]:
        return {
            "statistics": self.statistics.to_dict(),
            "query_count": self.query_count,
            "symmetry_error": self.symmetry_error,
            "reproduction_error": self.reproduction_error,
        }


@dataclass(frozen=True)
class KnownTargetOrbitInvariants:
    constraint_rank: int
    orthogonal_complement_dimension: int
    constrained_component: np.ndarray
    residual_gram: np.ndarray

    def to_dict(self) -> dict[str, object]:
        return {
            "constraint_rank": self.constraint_rank,
            "orthogonal_complement_dimension": self.orthogonal_complement_dimension,
            "constrained_component": self.constrained_component.tolist(),
            "residual_gram": self.residual_gram.tolist(),
        }


def _validated_oracle_value(
    value: LinearGradientOracleValue, output_dimension: int, input_dimension: int
) -> LinearGradientOracleValue:
    weight = _matrix("oracle weight_gradient", value.weight_gradient)
    bias = _vector("oracle bias_gradient", value.bias_gradient, output_dimension)
    expected = (output_dimension, input_dimension)
    if weight.shape != expected:
        raise ValueError(f"oracle weight_gradient must have shape {expected}")
    return LinearGradientOracleValue(weight, bias)


def recover_linear_gradient_oracle_statistics(
    targets: ArrayLike,
    input_dimension: int,
    oracle: Callable[[np.ndarray, np.ndarray], LinearGradientOracleValue],
    *,
    atol: float = 1e-9,
) -> LinearGradientOracleProbeRecovery:
    """Recover the entire known-target oracle from ``d+1`` classical queries.

    The zero-parameter query reveals ``Y^T X``. For each coordinate ``k``, a
    query whose first output row is ``e_k`` reveals row ``k`` of ``X^T X`` and
    coordinate ``k`` of ``X^T 1``. The returned statistics emulate every future
    query without further access to the hidden batch.
    """

    y = _matrix("targets", targets)
    if y.shape[0] == 0 or y.shape[1] == 0:
        raise ValueError("targets must contain at least one sample and output")
    dimension = int(input_dimension)
    if dimension <= 0:
        raise ValueError("input_dimension must be positive")
    tolerance = float(atol)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("atol must be finite and non-negative")

    batch, outputs = y.shape
    zero_weights = np.zeros((outputs, dimension), dtype=np.float64)
    zero_bias = np.zeros(outputs, dtype=np.float64)
    base = _validated_oracle_value(oracle(zero_weights, zero_bias), outputs, dimension)
    target_cross = -batch * base.weight_gradient

    gram = np.empty((dimension, dimension), dtype=np.float64)
    input_sum = np.empty(dimension, dtype=np.float64)
    probes: list[tuple[np.ndarray, LinearGradientOracleValue]] = []
    for coordinate in range(dimension):
        weights = np.zeros_like(zero_weights)
        weights[0, coordinate] = 1.0
        value = _validated_oracle_value(oracle(weights, zero_bias), outputs, dimension)
        gram[coordinate, :] = batch * (
            value.weight_gradient[0, :] - base.weight_gradient[0, :]
        )
        input_sum[coordinate] = batch * (
            value.bias_gradient[0] - base.bias_gradient[0]
        )
        probes.append((weights, value))

    symmetry_error = float(np.max(np.abs(gram - gram.T)))
    if symmetry_error > tolerance:
        raise ArithmeticError(
            "recovered input Gram matrix is not symmetric within tolerance; "
            "the callback is inconsistent with the declared exact oracle"
        )
    gram = 0.5 * (gram + gram.T)
    statistics = LinearGradientOracleStatistics(
        batch_size=batch,
        input_dimension=dimension,
        output_dimension=outputs,
        input_gram=gram,
        input_sum=input_sum,
        target_cross=target_cross,
        target_sum=y.sum(axis=0),
    )

    reproduction_error = 0.0
    transcript = [(zero_weights, base), *probes]
    for weights, observed in transcript:
        expected = evaluate_linear_gradient_oracle_from_statistics(
            statistics, weights, zero_bias
        )
        reproduction_error = max(
            reproduction_error,
            float(np.max(np.abs(expected.weight_gradient - observed.weight_gradient))),
            float(np.max(np.abs(expected.bias_gradient - observed.bias_gradient))),
        )
    if reproduction_error > tolerance:
        raise ArithmeticError(
            "recovered statistics do not reproduce the probe transcript within tolerance"
        )
    return LinearGradientOracleProbeRecovery(
        statistics=statistics,
        query_count=dimension + 1,
        symmetry_error=symmetry_error,
        reproduction_error=reproduction_error,
    )


def known_target_orbit_invariants_from_statistics(
    statistics: LinearGradientOracleStatistics,
    targets: ArrayLike,
    *,
    atol: float = 1e-9,
) -> KnownTargetOrbitInvariants:
    """Recover exactly the quotient information identified by the full oracle."""

    y = _matrix("targets", targets)
    tolerance = float(atol)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("atol must be finite and non-negative")
    _validate_statistics(statistics, y, atol=tolerance)

    constraints = target_constraint_matrix(y)
    constraint_rank = _rank(constraints, tolerance)
    cross = np.vstack((statistics.input_sum[None, :], statistics.target_cross))
    constrained = constraints @ np.linalg.pinv(constraints.T @ constraints) @ cross
    residual_gram = statistics.input_gram - constrained.T @ constrained
    residual_gram = 0.5 * (residual_gram + residual_gram.T)
    smallest = float(np.linalg.eigvalsh(residual_gram).min())
    if smallest < -tolerance:
        raise ValueError(
            "statistics are inconsistent: target-orthogonal residual Gram is not positive semidefinite"
        )
    return KnownTargetOrbitInvariants(
        constraint_rank=constraint_rank,
        orthogonal_complement_dimension=statistics.batch_size - constraint_rank,
        constrained_component=constrained,
        residual_gram=residual_gram,
    )


def known_target_orbit_invariants(
    inputs: ArrayLike,
    targets: ArrayLike,
    *,
    atol: float = 1e-9,
) -> KnownTargetOrbitInvariants:
    return known_target_orbit_invariants_from_statistics(
        linear_gradient_oracle_statistics(inputs, targets), targets, atol=atol
    )


def construct_known_target_orbit_representative(
    statistics: LinearGradientOracleStatistics,
    targets: ArrayLike,
    *,
    atol: float = 1e-9,
) -> np.ndarray:
    """Construct one deterministic real-valued representative of an oracle fibre.

    The component in ``span([1,Y])`` is fixed exactly. A factor of the residual
    Gram matrix is embedded into a deterministic orthonormal basis of the
    target-orthogonal subspace. The representative describes the exact quotient;
    it is not guaranteed to satisfy a separate discrete or box candidate domain.
    """

    tolerance = float(atol)
    invariants = known_target_orbit_invariants_from_statistics(
        statistics, targets, atol=tolerance
    )
    basis = target_stabilizer_basis(targets, tolerance=tolerance)
    eigenvalues, eigenvectors = np.linalg.eigh(invariants.residual_gram)
    order = np.argsort(eigenvalues)[::-1]
    positive = [index for index in order if eigenvalues[index] > tolerance]
    if len(positive) > basis.shape[1]:
        raise ValueError(
            "statistics are inconsistent: residual rank exceeds target-orthogonal dimension"
        )
    coordinates = np.zeros(
        (basis.shape[1], statistics.input_dimension), dtype=np.float64
    )
    if positive:
        vectors = eigenvectors[:, positive].T
        coordinates[: len(positive), :] = (
            np.sqrt(eigenvalues[positive])[:, None] * vectors
        )
    representative = invariants.constrained_component + basis @ coordinates
    reconstructed = linear_gradient_oracle_statistics(representative, targets)
    for name, expected, observed in (
        ("input_gram", statistics.input_gram, reconstructed.input_gram),
        ("input_sum", statistics.input_sum, reconstructed.input_sum),
        ("target_cross", statistics.target_cross, reconstructed.target_cross),
    ):
        if not np.allclose(expected, observed, atol=50 * tolerance, rtol=0.0):
            raise RuntimeError(f"constructed representative does not preserve {name}")
    return representative
