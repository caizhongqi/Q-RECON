from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class LinearBatchGradientObservation:
    weight_gradient: np.ndarray
    bias_gradient: np.ndarray


@dataclass(frozen=True)
class BatchCollisionReport:
    batch_size: int
    input_dimension: int
    output_dimension: int
    input_change_frobenius: float
    target_change_frobenius: float
    weight_gradient_error_frobenius: float
    bias_gradient_error_l2: float
    nontrivial: bool

    def to_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)


def _finite_array(name: str, value: np.ndarray, dimensions: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != dimensions:
        raise ValueError(f"{name} must have {dimensions} dimensions")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def linear_squared_loss_gradients(
    theta: np.ndarray,
    bias: np.ndarray,
    inputs: np.ndarray,
    targets: np.ndarray,
) -> LinearBatchGradientObservation:
    """Full-batch gradients for mean half-squared error in a biased linear model.

    The model is ``f(x)=theta @ x + bias`` and the loss is
    ``(1/(2B)) * sum_i ||f(x_i)-y_i||^2``.
    """

    weights = _finite_array("theta", theta, 2)
    offset = _finite_array("bias", bias, 1)
    x = _finite_array("inputs", inputs, 2)
    y = _finite_array("targets", targets, 2)
    batch, input_dimension = x.shape
    output_dimension, theta_input_dimension = weights.shape
    if batch <= 0:
        raise ValueError("inputs must contain at least one sample")
    if theta_input_dimension != input_dimension:
        raise ValueError("theta and input dimensions do not match")
    if offset.shape != (output_dimension,):
        raise ValueError("bias shape does not match theta outputs")
    if y.shape != (batch, output_dimension):
        raise ValueError("target shape must be (batch, output_dimension)")

    residual = x @ weights.T + offset[None, :] - y
    return LinearBatchGradientObservation(
        weight_gradient=(residual.T @ x) / batch,
        bias_gradient=residual.mean(axis=0),
    )


def validate_batch_mixing_matrix(mixing: np.ndarray, *, atol: float = 1e-10) -> np.ndarray:
    """Validate an invertible batch mixing matrix satisfying ``A @ 1 = 1``."""

    matrix = _finite_array("mixing", mixing, 2)
    if matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 2:
        raise ValueError("mixing must be a square matrix of size at least two")
    if not np.allclose(matrix @ np.ones(matrix.shape[0]), 1.0, atol=atol, rtol=0.0):
        raise ValueError("mixing must preserve the all-ones vector")
    if np.linalg.matrix_rank(matrix, tol=atol) != matrix.shape[0]:
        raise ValueError("mixing must be invertible")
    return matrix


def symmetric_pair_mixing(batch_size: int, first: int, second: int, alpha: float) -> np.ndarray:
    """Return a non-permutation row-stochastic two-sample mixing embedded in I.

    The selected 2x2 block is ``[[1-a, a], [a, 1-a]]`` and is invertible for
    ``a != 1/2``. For ``0 < a < 1`` it maps a convex input domain to itself.
    """

    if batch_size < 2:
        raise ValueError("batch_size must be at least two")
    if first == second or first < 0 or second < 0 or first >= batch_size or second >= batch_size:
        raise ValueError("first and second must be distinct valid batch indices")
    value = float(alpha)
    if not np.isfinite(value) or value == 0.5:
        raise ValueError("alpha must be finite and different from 0.5")
    matrix = np.eye(batch_size, dtype=np.float64)
    matrix[first, first] = 1.0 - value
    matrix[first, second] = value
    matrix[second, first] = value
    matrix[second, second] = 1.0 - value
    return validate_batch_mixing_matrix(matrix)


def construct_linear_batch_collision(
    theta: np.ndarray,
    bias: np.ndarray,
    inputs: np.ndarray,
    targets: np.ndarray,
    mixing: np.ndarray,
    *,
    atol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, BatchCollisionReport]:
    """Construct a different batch with exactly the same full aggregate gradient.

    If ``Delta = X theta^T + 1 b^T - Y``, set ``X' = A X`` and
    ``Delta' = A^{-T} Delta`` for an invertible ``A`` satisfying ``A 1 = 1``.
    Choosing ``Y' = X' theta^T + 1 b^T - Delta'`` preserves both the weight and
    bias gradients of mean half-squared error.
    """

    weights = _finite_array("theta", theta, 2)
    offset = _finite_array("bias", bias, 1)
    x = _finite_array("inputs", inputs, 2)
    y = _finite_array("targets", targets, 2)
    matrix = validate_batch_mixing_matrix(mixing, atol=atol)
    if matrix.shape[0] != x.shape[0]:
        raise ValueError("mixing size must equal the batch size")

    original = linear_squared_loss_gradients(weights, offset, x, y)
    residual = x @ weights.T + offset[None, :] - y
    mixed_inputs = matrix @ x
    mixed_residual = np.linalg.solve(matrix.T, residual)
    mixed_targets = mixed_inputs @ weights.T + offset[None, :] - mixed_residual
    transformed = linear_squared_loss_gradients(weights, offset, mixed_inputs, mixed_targets)

    weight_error = float(
        np.linalg.norm(original.weight_gradient - transformed.weight_gradient)
    )
    bias_error = float(np.linalg.norm(original.bias_gradient - transformed.bias_gradient))
    input_change = float(np.linalg.norm(x - mixed_inputs))
    target_change = float(np.linalg.norm(y - mixed_targets))
    report = BatchCollisionReport(
        batch_size=x.shape[0],
        input_dimension=x.shape[1],
        output_dimension=weights.shape[0],
        input_change_frobenius=input_change,
        target_change_frobenius=target_change,
        weight_gradient_error_frobenius=weight_error,
        bias_gradient_error_l2=bias_error,
        nontrivial=input_change > atol or target_change > atol,
    )
    if weight_error > atol or bias_error > atol:
        raise ArithmeticError("constructed batch did not preserve gradients within tolerance")
    return mixed_inputs, mixed_targets, report
