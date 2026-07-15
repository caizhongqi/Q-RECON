from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from .known_target_collisions import LinearGradientOracleStatistics


@dataclass(frozen=True)
class GradientOracleStatisticDistance:
    """Norm differences between two known-target linear-gradient quotients."""

    input_gram_frobenius: float
    input_sum_l2: float
    target_cross_frobenius: float

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            converted = float(value)
            if not math.isfinite(converted) or converted < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, converted)

    @property
    def exactly_equal(self) -> bool:
        return (
            self.input_gram_frobenius == 0.0
            and self.input_sum_l2 == 0.0
            and self.target_cross_frobenius == 0.0
        )

    def to_dict(self) -> dict[str, float | bool]:
        return {**asdict(self), "exactly_equal": self.exactly_equal}


@dataclass(frozen=True)
class UniformGradientQueryDifferenceBound:
    """Uniform Euclidean transcript-mean difference for one oracle query."""

    weight_gradient_frobenius: float
    bias_gradient_l2: float
    combined_l2: float
    max_weight_operator_norm: float
    max_bias_l2: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _finite_non_negative(name: str, value: float) -> float:
    converted = float(value)
    if not math.isfinite(converted) or converted < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return converted


def _validate_statistics_pair(
    left: LinearGradientOracleStatistics,
    right: LinearGradientOracleStatistics,
) -> None:
    shape_left = (
        left.batch_size,
        left.input_dimension,
        left.output_dimension,
    )
    shape_right = (
        right.batch_size,
        right.input_dimension,
        right.output_dimension,
    )
    if shape_left != shape_right:
        raise ValueError("statistics must have equal batch, input, and output dimensions")
    for statistics in (left, right):
        expected = (
            (statistics.input_gram, (statistics.input_dimension, statistics.input_dimension)),
            (statistics.input_sum, (statistics.input_dimension,)),
            (
                statistics.target_cross,
                (statistics.output_dimension, statistics.input_dimension),
            ),
            (statistics.target_sum, (statistics.output_dimension,)),
        )
        for value, shape in expected:
            array = np.asarray(value, dtype=np.float64)
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError("statistics contain a non-finite or malformed array")
    if not np.array_equal(left.target_sum, right.target_sum):
        raise ValueError(
            "known-target stability comparison requires a common target sum"
        )


def gradient_oracle_statistic_distance(
    left: LinearGradientOracleStatistics,
    right: LinearGradientOracleStatistics,
) -> GradientOracleStatisticDistance:
    """Return the quotient-statistic norm differences of two gradient oracles."""

    _validate_statistics_pair(left, right)
    return GradientOracleStatisticDistance(
        input_gram_frobenius=float(
            np.linalg.norm(left.input_gram - right.input_gram, ord="fro")
        ),
        input_sum_l2=float(np.linalg.norm(left.input_sum - right.input_sum)),
        target_cross_frobenius=float(
            np.linalg.norm(left.target_cross - right.target_cross, ord="fro")
        ),
    )


def uniform_gradient_query_difference_bound(
    left: LinearGradientOracleStatistics,
    right: LinearGradientOracleStatistics,
    *,
    max_weight_operator_norm: float,
    max_bias_l2: float,
) -> UniformGradientQueryDifferenceBound:
    """Bound every one-query mean difference on a bounded parameter domain.

    For a common known target batch and mean half-squared loss,

    ``Delta grad_W = (Theta Delta G + b Delta s^T - Delta C) / m``
    ``Delta grad_b = Theta Delta s / m``.

    The returned values follow from submultiplicativity and the triangle
    inequality for every query satisfying ``||Theta||_2 <= R_theta`` and
    ``||b||_2 <= R_b``.
    """

    weight_radius = _finite_non_negative(
        "max_weight_operator_norm", max_weight_operator_norm
    )
    bias_radius = _finite_non_negative("max_bias_l2", max_bias_l2)
    distance = gradient_oracle_statistic_distance(left, right)
    batch = left.batch_size
    weight_bound = (
        weight_radius * distance.input_gram_frobenius
        + bias_radius * distance.input_sum_l2
        + distance.target_cross_frobenius
    ) / batch
    bias_bound = weight_radius * distance.input_sum_l2 / batch
    return UniformGradientQueryDifferenceBound(
        weight_gradient_frobenius=weight_bound,
        bias_gradient_l2=bias_bound,
        combined_l2=math.hypot(weight_bound, bias_bound),
        max_weight_operator_norm=weight_radius,
        max_bias_l2=bias_radius,
    )


def equal_covariance_gaussian_binary_success(
    mean_l2_distance: float,
    noise_std: float,
) -> float:
    """Exact equal-prior success for two Gaussian means with covariance sigma^2 I."""

    distance = _finite_non_negative("mean_l2_distance", mean_l2_distance)
    sigma = float(noise_std)
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("noise_std must be finite and positive")
    # Phi(distance / (2 sigma)) expressed through erf.
    return 0.5 * (1.0 + math.erf(distance / (2.0 * math.sqrt(2.0) * sigma)))


def adaptive_gaussian_transcript_success_upper_bound(
    per_query_l2_difference: float,
    queries: int,
    noise_std: float,
) -> float:
    """Pinsker upper bound for arbitrary adaptive Gaussian-noisy queries.

    Each response is the vectorized full gradient plus independent
    ``N(0, sigma^2 I)`` noise.  If the conditional mean difference is at most
    ``B`` for every allowed query and every history, the chain rule gives
    ``KL <= q B^2 / (2 sigma^2)``. Pinsker and the binary Bayes formula yield
    ``P_success <= 1/2 + sqrt(q) B / (4 sigma)``.
    """

    difference = _finite_non_negative(
        "per_query_l2_difference", per_query_l2_difference
    )
    if queries < 0:
        raise ValueError("queries must be non-negative")
    sigma = float(noise_std)
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("noise_std must be finite and positive")
    return min(1.0, 0.5 + math.sqrt(queries) * difference / (4.0 * sigma))


def necessary_gaussian_queries_for_binary_success(
    target_success: float,
    per_query_l2_difference: float,
    noise_std: float,
) -> int | None:
    """Necessary query count implied by the adaptive Gaussian upper bound.

    ``None`` means that the bound forbids every target above random guessing
    because the two oracle means are exactly equal.
    """

    target = float(target_success)
    if not math.isfinite(target) or target < 0.5 or target > 1.0:
        raise ValueError("target_success must lie in [0.5, 1]")
    difference = _finite_non_negative(
        "per_query_l2_difference", per_query_l2_difference
    )
    sigma = float(noise_std)
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("noise_std must be finite and positive")
    if target <= 0.5:
        return 0
    if difference == 0.0:
        return None
    threshold = (4.0 * sigma * (target - 0.5) / difference) ** 2
    # Floating roundoff at an exact integer must not add a spurious extra query.
    rounded = round(threshold)
    if math.isclose(threshold, rounded, rel_tol=1e-12, abs_tol=1e-12):
        return int(rounded)
    return math.ceil(threshold)
