import numpy as np
import pytest

from qrecon.theory.known_target_collisions import evaluate_linear_gradient_oracle
from qrecon.theory.known_target_probe_optimality import (
    construct_physical_probe_lower_bound_witness,
    exact_known_target_probe_query_count,
)


def _targets(batch: int = 6) -> np.ndarray:
    grid = np.arange(batch, dtype=float)
    return np.column_stack((grid, grid**2))


def test_exact_query_count_formula():
    assert exact_known_target_probe_query_count(5, 2) == 4
    assert exact_known_target_probe_query_count(6, 3) == 3
    assert exact_known_target_probe_query_count(2, 5) == 2


def test_every_subcritical_query_set_has_a_physical_indistinguishable_batch():
    rng = np.random.default_rng(53)
    targets = _targets()
    queries = tuple(rng.normal(size=(2, 5)) for _ in range(3))
    witness = construct_physical_probe_lower_bound_witness(targets, queries)
    assert witness.exact_query_lower_bound == 4
    assert witness.query_count == 3
    assert witness.input_displacement > 0.0
    assert witness.target_constraint_error < 1e-9
    assert witness.maximum_query_gradient_error < 1e-9

    zero = np.zeros_like(witness.alternative_inputs)
    for weights in queries:
        for bias in (np.zeros(2), rng.normal(size=2)):
            left = evaluate_linear_gradient_oracle(zero, targets, weights, bias)
            right = evaluate_linear_gradient_oracle(
                witness.alternative_inputs, targets, weights, bias
            )
            assert np.allclose(left.weight_gradient, right.weight_gradient)
            assert np.allclose(left.bias_gradient, right.bias_gradient)


def test_zero_shared_image_uses_target_orthogonal_direction():
    targets = _targets()
    queries = (np.zeros((2, 4)), np.zeros((2, 4)))
    witness = construct_physical_probe_lower_bound_witness(targets, queries)
    assert witness.input_displacement > 0.0
    assert np.allclose(targets.T @ witness.sample_direction, 0.0)
    assert np.isclose(np.sum(witness.sample_direction), 0.0)


def test_assumptions_and_non_subcritical_counts_are_rejected():
    targets = _targets()
    with pytest.raises(ValueError, match="fewer"):
        construct_physical_probe_lower_bound_witness(
            targets, tuple(np.zeros((2, 5)) for _ in range(4))
        )
    with pytest.raises(ValueError, match="batch size"):
        construct_physical_probe_lower_bound_witness(
            _targets(3), (np.zeros((2, 5)),)
        )
