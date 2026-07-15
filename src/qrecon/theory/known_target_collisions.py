from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np


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


def _rank(matrix: np.ndarray, tolerance: float | None = None) -> int:
    singular = np.linalg.svd(matrix, compute_uv=False)
    if singular.size == 0:
        return 0
    if tolerance is None:
        tolerance = max(matrix.shape, default=1) * np.finfo(np.float64).eps * singular[0]
    if tolerance < 0.0 or not math.isfinite(tolerance):
        raise ValueError("tolerance must be finite and non-negative")
    return int(np.count_nonzero(singular > tolerance))


@dataclass(frozen=True)
class LinearGradientOracleStatistics:
    """Sufficient statistics of the full biased-linear MSE gradient oracle."""

    batch_size: int
    input_dimension: int
    output_dimension: int
    input_gram: np.ndarray
    input_sum: np.ndarray
    target_cross: np.ndarray
    target_sum: np.ndarray

    def to_dict(self) -> dict[str, object]:
        return {
            "batch_size": self.batch_size,
            "input_dimension": self.input_dimension,
            "output_dimension": self.output_dimension,
            "input_gram": self.input_gram.tolist(),
            "input_sum": self.input_sum.tolist(),
            "target_cross": self.target_cross.tolist(),
            "target_sum": self.target_sum.tolist(),
        }


@dataclass(frozen=True)
class LinearGradientOracleValue:
    weight_gradient: np.ndarray
    bias_gradient: np.ndarray

    def to_dict(self) -> dict[str, object]:
        return {
            "weight_gradient": self.weight_gradient.tolist(),
            "bias_gradient": self.bias_gradient.tolist(),
        }


@dataclass(frozen=True)
class KnownTargetOrbitReport:
    batch_size: int
    target_constraint_rank: int
    orthogonal_complement_dimension: int
    projected_input_rank: int
    stabilizer_group_dimension: int
    continuous_orbit_dimension: int
    has_nontrivial_collision: bool
    has_continuous_family: bool

    def to_dict(self) -> dict[str, int | bool]:
        return asdict(self)


@dataclass(frozen=True)
class KnownTargetCollisionReport:
    transformation: np.ndarray
    transformed_inputs: np.ndarray
    orthogonality_error: float
    fixed_constraint_error: float
    statistic_error: float
    probe_weight_gradient_error: float
    probe_bias_gradient_error: float
    input_displacement: float
    orbit: KnownTargetOrbitReport

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["transformation"] = self.transformation.tolist()
        payload["transformed_inputs"] = self.transformed_inputs.tolist()
        payload["orbit"] = self.orbit.to_dict()
        return payload


def _validate_inputs_targets(inputs: ArrayLike, targets: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    x = _matrix("inputs", inputs)
    y = _matrix("targets", targets)
    if x.shape[0] == 0:
        raise ValueError("batch must contain at least one sample")
    if x.shape[1] == 0:
        raise ValueError("inputs must contain at least one feature")
    if y.shape[1] == 0:
        raise ValueError("targets must contain at least one output")
    if x.shape[0] != y.shape[0]:
        raise ValueError("inputs and targets must have the same batch size")
    return x, y


def target_constraint_matrix(targets: ArrayLike) -> np.ndarray:
    """Return ``[1, Y]``, the sample-index directions fixed by known targets."""

    y = _matrix("targets", targets)
    if y.shape[0] == 0:
        raise ValueError("targets must contain at least one sample")
    if y.shape[1] == 0:
        raise ValueError("targets must contain at least one output")
    return np.column_stack((np.ones(y.shape[0], dtype=np.float64), y))


def linear_gradient_oracle_statistics(
    inputs: ArrayLike, targets: ArrayLike
) -> LinearGradientOracleStatistics:
    """Compute the complete sufficient statistics of every parameter query.

    For ``f(x)=Theta x+b`` and mean half-squared loss, the full gradient at every
    possible ``(Theta,b)`` depends on ``X`` only through ``X^T X``, ``X^T 1`` and
    ``Y^T X``. ``Y^T 1`` is included for evaluating the bias gradient.
    """

    x, y = _validate_inputs_targets(inputs, targets)
    return LinearGradientOracleStatistics(
        batch_size=x.shape[0],
        input_dimension=x.shape[1],
        output_dimension=y.shape[1],
        input_gram=x.T @ x,
        input_sum=x.sum(axis=0),
        target_cross=y.T @ x,
        target_sum=y.sum(axis=0),
    )


def evaluate_linear_gradient_oracle_from_statistics(
    statistics: LinearGradientOracleStatistics,
    weights: ArrayLike,
    bias: np.ndarray | Sequence[float],
) -> LinearGradientOracleValue:
    """Evaluate the full gradient oracle from its sufficient statistics."""

    theta = _matrix("weights", weights)
    expected = (statistics.output_dimension, statistics.input_dimension)
    if theta.shape != expected:
        raise ValueError(f"weights must have shape {expected}")
    b = _vector("bias", bias, statistics.output_dimension)
    batch = statistics.batch_size
    weight_gradient = (
        theta @ statistics.input_gram
        + b[:, None] * statistics.input_sum[None, :]
        - statistics.target_cross
    ) / batch
    bias_gradient = (
        theta @ statistics.input_sum + batch * b - statistics.target_sum
    ) / batch
    return LinearGradientOracleValue(weight_gradient, bias_gradient)


def evaluate_linear_gradient_oracle(
    inputs: ArrayLike,
    targets: ArrayLike,
    weights: ArrayLike,
    bias: np.ndarray | Sequence[float],
) -> LinearGradientOracleValue:
    """Directly evaluate the biased-linear mean half-squared-loss gradients."""

    x, y = _validate_inputs_targets(inputs, targets)
    theta = _matrix("weights", weights)
    expected = (y.shape[1], x.shape[1])
    if theta.shape != expected:
        raise ValueError(f"weights must have shape {expected}")
    b = _vector("bias", bias, y.shape[1])
    residual = x @ theta.T + b[None, :] - y
    return LinearGradientOracleValue(
        weight_gradient=(residual.T @ x) / x.shape[0],
        bias_gradient=residual.mean(axis=0),
    )


def linear_gradient_oracles_equivalent(
    left_inputs: ArrayLike,
    right_inputs: ArrayLike,
    targets: ArrayLike,
    *,
    atol: float = 1e-10,
) -> bool:
    """Whether two input batches induce the same gradient function for all parameters."""

    left = linear_gradient_oracle_statistics(left_inputs, targets)
    right = linear_gradient_oracle_statistics(right_inputs, targets)
    if left.batch_size != right.batch_size or left.input_dimension != right.input_dimension:
        return False
    return all(
        np.allclose(a, b, atol=atol, rtol=0.0)
        for a, b in (
            (left.input_gram, right.input_gram),
            (left.input_sum, right.input_sum),
            (left.target_cross, right.target_cross),
        )
    )


def target_stabilizer_basis(
    targets: ArrayLike, *, tolerance: float | None = None
) -> np.ndarray:
    """Orthonormal basis of ``span([1,Y])^perp`` in sample-index space."""

    constraints = target_constraint_matrix(targets)
    u, singular, _ = np.linalg.svd(constraints, full_matrices=True)
    if singular.size == 0:
        rank = 0
    else:
        threshold = tolerance
        if threshold is None:
            threshold = max(constraints.shape) * np.finfo(np.float64).eps * singular[0]
        if threshold < 0.0 or not math.isfinite(threshold):
            raise ValueError("tolerance must be finite and non-negative")
        rank = int(np.count_nonzero(singular > threshold))
    return u[:, rank:]


def target_stabilizer_rotation(
    targets: ArrayLike,
    angle: float,
    *,
    axes: tuple[int, int] = (0, 1),
    tolerance: float | None = None,
) -> np.ndarray:
    """Orthogonal sample mixing that fixes the all-ones vector and every target column."""

    theta = float(angle)
    if not math.isfinite(theta):
        raise ValueError("angle must be finite")
    basis = target_stabilizer_basis(targets, tolerance=tolerance)
    complement = basis.shape[1]
    first, second = axes
    if first == second or min(first, second) < 0 or max(first, second) >= complement:
        raise ValueError(
            f"axes must select two distinct directions in a {complement}-dimensional complement"
        )
    local = np.eye(complement, dtype=np.float64)
    cosine = math.cos(theta)
    sine = math.sin(theta)
    local[first, first] = cosine
    local[first, second] = -sine
    local[second, first] = sine
    local[second, second] = cosine
    batch = basis.shape[0]
    return np.eye(batch) + basis @ (local - np.eye(complement)) @ basis.T


def target_stabilizer_reflection(
    targets: ArrayLike,
    *,
    axis: int = 0,
    tolerance: float | None = None,
) -> np.ndarray:
    """Reflection in one target-orthogonal direction, including one-dimensional cases."""

    basis = target_stabilizer_basis(targets, tolerance=tolerance)
    complement = basis.shape[1]
    if axis < 0 or axis >= complement:
        raise ValueError(f"axis must lie in [0, {complement})")
    vector = basis[:, axis : axis + 1]
    return np.eye(basis.shape[0]) - 2.0 * (vector @ vector.T)


def known_target_orbit_report(
    inputs: ArrayLike,
    targets: ArrayLike,
    *,
    tolerance: float | None = None,
) -> KnownTargetOrbitReport:
    """Dimension of the target-stabilizer orbit through the input batch."""

    x, y = _validate_inputs_targets(inputs, targets)
    constraints = target_constraint_matrix(y)
    constraint_rank = _rank(constraints, tolerance)
    basis = target_stabilizer_basis(y, tolerance=tolerance)
    complement = basis.shape[1]
    projected_rank = _rank(basis.T @ x, tolerance) if complement else 0
    group_dimension = complement * (complement - 1) // 2
    orbit_dimension = projected_rank * (2 * complement - projected_rank - 1) // 2
    nontrivial = complement > 0 and projected_rank > 0
    return KnownTargetOrbitReport(
        batch_size=x.shape[0],
        target_constraint_rank=constraint_rank,
        orthogonal_complement_dimension=complement,
        projected_input_rank=projected_rank,
        stabilizer_group_dimension=group_dimension,
        continuous_orbit_dimension=orbit_dimension,
        has_nontrivial_collision=nontrivial,
        has_continuous_family=orbit_dimension > 0,
    )


def recover_target_stabilizing_orthogonal_map(
    source_inputs: ArrayLike,
    target_inputs: ArrayLike,
    targets: ArrayLike,
    *,
    atol: float = 1e-9,
) -> np.ndarray:
    """Construct ``Q`` with ``Q[1,Y,X]=[1,Y,X']`` when the oracle statistics agree.

    Equal gradient-oracle statistics make the two concatenated matrices have the
    same Gram matrix. The routine constructs the induced isometry on their column
    spans and extends it to an orthogonal map on the whole sample-index space.
    """

    source, y = _validate_inputs_targets(source_inputs, targets)
    target, y_again = _validate_inputs_targets(target_inputs, targets)
    if source.shape != target.shape or not np.array_equal(y, y_again):
        raise ValueError("source and target inputs must have equal shape and common targets")
    if not linear_gradient_oracles_equivalent(source, target, y, atol=atol):
        raise ValueError("input batches do not induce the same full gradient oracle")

    constraints = target_constraint_matrix(y)
    left = np.column_stack((constraints, source))
    right = np.column_stack((constraints, target))
    if not np.allclose(left.T @ left, right.T @ right, atol=atol, rtol=0.0):
        raise ValueError("concatenated Gram matrices are inconsistent")

    u_left, singular, vh = np.linalg.svd(left, full_matrices=True)
    rank = _rank(left, atol)
    if rank == 0:
        return np.eye(left.shape[0])
    v = vh.T[:, :rank]
    u_right = right @ v @ np.diag(1.0 / singular[:rank])
    if not np.allclose(u_right.T @ u_right, np.eye(rank), atol=20 * atol, rtol=0.0):
        raise RuntimeError("equal Gram matrices did not induce a numerical isometry")

    _, _, right_null_vh = np.linalg.svd(u_right.T, full_matrices=True)
    right_complement = right_null_vh[rank:].T
    left_complement = u_left[:, rank:]
    transform = u_right @ u_left[:, :rank].T + right_complement @ left_complement.T

    if not np.allclose(transform.T @ transform, np.eye(transform.shape[0]), atol=20 * atol, rtol=0.0):
        raise RuntimeError("failed to construct an orthogonal extension")
    if not np.allclose(transform @ left, right, atol=50 * atol, rtol=0.0):
        raise RuntimeError("orthogonal map does not reproduce the requested batches")
    return transform


def construct_known_target_rotation_collision(
    inputs: ArrayLike,
    targets: ArrayLike,
    weights: ArrayLike,
    bias: np.ndarray | Sequence[float],
    angle: float,
    *,
    axes: tuple[int, int] = (0, 1),
    tolerance: float | None = None,
) -> KnownTargetCollisionReport:
    """Generate and audit one exact known-target collision from a stabilizer rotation."""

    x, y = _validate_inputs_targets(inputs, targets)
    transform = target_stabilizer_rotation(y, angle, axes=axes, tolerance=tolerance)
    transformed = transform @ x
    constraints = target_constraint_matrix(y)
    left_stats = linear_gradient_oracle_statistics(x, y)
    right_stats = linear_gradient_oracle_statistics(transformed, y)
    left_probe = evaluate_linear_gradient_oracle(x, y, weights, bias)
    right_probe = evaluate_linear_gradient_oracle(transformed, y, weights, bias)
    statistic_error = max(
        float(np.max(np.abs(a - b)))
        for a, b in (
            (left_stats.input_gram, right_stats.input_gram),
            (left_stats.input_sum, right_stats.input_sum),
            (left_stats.target_cross, right_stats.target_cross),
        )
    )
    return KnownTargetCollisionReport(
        transformation=transform,
        transformed_inputs=transformed,
        orthogonality_error=float(np.max(np.abs(transform.T @ transform - np.eye(x.shape[0])))),
        fixed_constraint_error=float(np.max(np.abs(transform @ constraints - constraints))),
        statistic_error=statistic_error,
        probe_weight_gradient_error=float(
            np.max(np.abs(left_probe.weight_gradient - right_probe.weight_gradient))
        ),
        probe_bias_gradient_error=float(
            np.max(np.abs(left_probe.bias_gradient - right_probe.bias_gradient))
        ),
        input_displacement=float(np.linalg.norm(transformed - x)),
        orbit=known_target_orbit_report(x, y, tolerance=tolerance),
    )
