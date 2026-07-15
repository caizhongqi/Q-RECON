from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .known_target_collisions import (
    LinearGradientOracleStatistics,
    LinearGradientOracleValue,
    evaluate_linear_gradient_oracle_from_statistics,
)
from .known_target_quotient import LinearGradientOracleProbeRecovery


ArrayLike = np.ndarray | Sequence[Sequence[float]]


def _matrix(name: str, value: ArrayLike) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validated_value(
    value: LinearGradientOracleValue, output_dimension: int, input_dimension: int
) -> LinearGradientOracleValue:
    weight = np.asarray(value.weight_gradient, dtype=np.float64)
    bias = np.asarray(value.bias_gradient, dtype=np.float64).reshape(-1)
    if weight.shape != (output_dimension, input_dimension):
        raise ValueError(
            "oracle weight_gradient must have shape "
            f"({output_dimension}, {input_dimension})"
        )
    if bias.shape != (output_dimension,):
        raise ValueError(f"oracle bias_gradient must have shape ({output_dimension},)")
    if not np.isfinite(weight).all() or not np.isfinite(bias).all():
        raise ValueError("oracle gradients must contain only finite values")
    return LinearGradientOracleValue(weight, bias)


@dataclass(frozen=True)
class PackedLinearGradientProbePlan:
    input_dimension: int
    output_dimension: int
    weight_queries: tuple[np.ndarray, ...]

    @property
    def query_count(self) -> int:
        return len(self.weight_queries)

    def to_dict(self) -> dict[str, object]:
        return {
            "input_dimension": self.input_dimension,
            "output_dimension": self.output_dimension,
            "query_count": self.query_count,
            "weight_queries": [query.tolist() for query in self.weight_queries],
        }


def build_packed_linear_gradient_probe_plan(
    input_dimension: int, output_dimension: int
) -> PackedLinearGradientProbePlan:
    """Build one zero query plus packed basis probes across output rows."""

    d = int(input_dimension)
    c = int(output_dimension)
    if d <= 0 or c <= 0:
        raise ValueError("input_dimension and output_dimension must be positive")
    queries = [np.zeros((c, d), dtype=np.float64)]
    for round_index in range(math.ceil(d / c)):
        weights = np.zeros((c, d), dtype=np.float64)
        for output_index in range(c):
            coordinate = round_index * c + output_index
            if coordinate >= d:
                break
            weights[output_index, coordinate] = 1.0
        queries.append(weights)
    return PackedLinearGradientProbePlan(d, c, tuple(queries))


def recover_linear_gradient_oracle_statistics_packed(
    targets: ArrayLike,
    input_dimension: int,
    oracle: Callable[[np.ndarray, np.ndarray], LinearGradientOracleValue],
    *,
    atol: float = 1e-9,
) -> LinearGradientOracleProbeRecovery:
    """Recover the exact oracle using ``1 + ceil(d/c)`` classical queries."""

    y = _matrix("targets", targets)
    if y.shape[0] == 0 or y.shape[1] == 0:
        raise ValueError("targets must contain at least one sample and output")
    tolerance = float(atol)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("atol must be finite and non-negative")

    batch, outputs = y.shape
    plan = build_packed_linear_gradient_probe_plan(input_dimension, outputs)
    zero_bias = np.zeros(outputs, dtype=np.float64)
    transcript = tuple(
        _validated_value(oracle(weights, zero_bias), outputs, plan.input_dimension)
        for weights in plan.weight_queries
    )
    base = transcript[0]
    target_cross = -batch * base.weight_gradient
    gram = np.empty((plan.input_dimension, plan.input_dimension), dtype=np.float64)
    input_sum = np.empty(plan.input_dimension, dtype=np.float64)

    for round_index, (weights, value) in enumerate(
        zip(plan.weight_queries[1:], transcript[1:])
    ):
        for output_index in range(outputs):
            coordinate = round_index * outputs + output_index
            if coordinate >= plan.input_dimension:
                break
            if weights[output_index, coordinate] != 1.0:
                raise AssertionError("packed probe plan lost its basis assignment")
            gram[coordinate, :] = batch * (
                value.weight_gradient[output_index, :]
                - base.weight_gradient[output_index, :]
            )
            input_sum[coordinate] = batch * (
                value.bias_gradient[output_index]
                - base.bias_gradient[output_index]
            )

    symmetry_error = float(np.max(np.abs(gram - gram.T)))
    if symmetry_error > tolerance:
        raise ArithmeticError(
            "recovered input Gram matrix is not symmetric within tolerance"
        )
    gram = 0.5 * (gram + gram.T)
    statistics = LinearGradientOracleStatistics(
        batch_size=batch,
        input_dimension=plan.input_dimension,
        output_dimension=outputs,
        input_gram=gram,
        input_sum=input_sum,
        target_cross=target_cross,
        target_sum=y.sum(axis=0),
    )

    reproduction_error = 0.0
    for weights, observed in zip(plan.weight_queries, transcript):
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
            "recovered statistics do not reproduce the packed probe transcript"
        )
    return LinearGradientOracleProbeRecovery(
        statistics=statistics,
        query_count=plan.query_count,
        symmetry_error=symmetry_error,
        reproduction_error=reproduction_error,
    )
