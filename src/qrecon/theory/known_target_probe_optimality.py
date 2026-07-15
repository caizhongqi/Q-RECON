from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np

from .known_target_collisions import (
    evaluate_linear_gradient_oracle,
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


def exact_known_target_probe_query_count(
    input_dimension: int, output_dimension: int
) -> int:
    """Exact worst-case deterministic query count under the theorem assumptions."""

    d = int(input_dimension)
    c = int(output_dimension)
    if d <= 0 or c <= 0:
        raise ValueError("input_dimension and output_dimension must be positive")
    return 1 + math.ceil(d / c)


@dataclass(frozen=True)
class PhysicalProbeLowerBoundWitness:
    batch_size: int
    input_dimension: int
    output_dimension: int
    query_count: int
    exact_query_lower_bound: int
    difference_span_rank: int
    null_vector: np.ndarray
    sample_direction: np.ndarray
    alternative_inputs: np.ndarray
    target_constraint_error: float
    maximum_query_gradient_error: float
    input_displacement: float

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["null_vector"] = self.null_vector.tolist()
        result["sample_direction"] = self.sample_direction.tolist()
        result["alternative_inputs"] = self.alternative_inputs.tolist()
        return result


def construct_physical_probe_lower_bound_witness(
    targets: ArrayLike,
    weight_queries: Sequence[ArrayLike],
    *,
    atol: float = 1e-9,
) -> PhysicalProbeLowerBoundWitness:
    """Construct two physical batches indistinguishable by too few probes.

    The baseline batch is zero and the alternative is rank one, ``X'=u v^T``.
    ``v`` lies in the nullspace of every within-output-row query difference,
    while ``u`` satisfies ``1^T u=0`` and
    ``Y^T u=||u||^2 Theta_0 v``. Every supplied parameter query then receives
    exactly the same full gradient response for both batches, for arbitrary query
    biases.
    """

    y = _matrix("targets", targets)
    if y.shape[0] == 0 or y.shape[1] == 0:
        raise ValueError("targets must contain at least one sample and output")
    if not weight_queries:
        raise ValueError("weight_queries must be non-empty")
    tolerance = float(atol)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("atol must be finite and non-negative")

    queries = tuple(_matrix("weight query", query) for query in weight_queries)
    outputs, dimension = queries[0].shape
    if outputs != y.shape[1]:
        raise ValueError("query output dimension must match the target dimension")
    if dimension == 0:
        raise ValueError("queries must contain at least one input coordinate")
    if any(query.shape != (outputs, dimension) for query in queries):
        raise ValueError("all weight queries must have the same shape")

    constraints = target_constraint_matrix(y)
    constraint_rank = int(np.linalg.matrix_rank(constraints, tol=tolerance))
    if constraint_rank != outputs + 1:
        raise ValueError("[1,Y] must have full column rank")
    if y.shape[0] < outputs + 2:
        raise ValueError("batch size must be at least output_dimension + 2")

    exact_bound = exact_known_target_probe_query_count(dimension, outputs)
    if len(queries) >= exact_bound:
        raise ValueError(
            f"witness requires fewer than {exact_bound} queries for this shape"
        )

    differences = (
        np.concatenate([query - queries[0] for query in queries[1:]], axis=0)
        if len(queries) > 1
        else np.zeros((0, dimension), dtype=np.float64)
    )
    if differences.size:
        _, singular, vh = np.linalg.svd(differences, full_matrices=True)
        rank = int(np.count_nonzero(singular > tolerance))
        null_vector = vh[rank]
    else:
        rank = 0
        null_vector = np.eye(dimension, dtype=np.float64)[0]
    null_vector = null_vector / np.linalg.norm(null_vector)
    shared_image = queries[0] @ null_vector

    if np.linalg.norm(shared_image) > tolerance:
        desired = np.concatenate(([0.0], shared_image))
        sample_seed = constraints @ np.linalg.solve(
            constraints.T @ constraints, desired
        )
        norm_squared = float(sample_seed @ sample_seed)
        if norm_squared <= tolerance:
            raise RuntimeError("failed to construct a nonzero sample direction")
        sample_direction = sample_seed / norm_squared
    else:
        complement = target_stabilizer_basis(y, tolerance=tolerance)
        if complement.shape[1] == 0:
            raise RuntimeError("target-orthogonal sample direction is unavailable")
        sample_direction = complement[:, 0]

    alternative = np.outer(sample_direction, null_vector)
    baseline = np.zeros_like(alternative)
    target_error = max(
        abs(float(np.ones(y.shape[0]) @ sample_direction)),
        float(
            np.max(
                np.abs(
                    y.T @ sample_direction
                    - float(sample_direction @ sample_direction) * shared_image
                )
            )
        ),
    )
    maximum_error = 0.0
    for query in queries:
        zero_bias = np.zeros(outputs, dtype=np.float64)
        left = evaluate_linear_gradient_oracle(baseline, y, query, zero_bias)
        right = evaluate_linear_gradient_oracle(alternative, y, query, zero_bias)
        maximum_error = max(
            maximum_error,
            float(np.max(np.abs(left.weight_gradient - right.weight_gradient))),
            float(np.max(np.abs(left.bias_gradient - right.bias_gradient))),
        )

    if target_error > 20 * tolerance or maximum_error > 20 * tolerance:
        raise RuntimeError("constructed physical witness failed numerical validation")
    return PhysicalProbeLowerBoundWitness(
        batch_size=y.shape[0],
        input_dimension=dimension,
        output_dimension=outputs,
        query_count=len(queries),
        exact_query_lower_bound=exact_bound,
        difference_span_rank=rank,
        null_vector=null_vector,
        sample_direction=sample_direction,
        alternative_inputs=alternative,
        target_constraint_error=target_error,
        maximum_query_gradient_error=maximum_error,
        input_displacement=float(np.linalg.norm(alternative)),
    )
