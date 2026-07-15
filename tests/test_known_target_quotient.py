import numpy as np
import pytest

from qrecon.theory.known_target_collisions import (
    evaluate_linear_gradient_oracle,
    evaluate_linear_gradient_oracle_from_statistics,
    linear_gradient_oracle_statistics,
    linear_gradient_oracles_equivalent,
    target_constraint_matrix,
)
from qrecon.theory.known_target_quotient import (
    construct_known_target_orbit_representative,
    known_target_orbit_invariants,
    recover_linear_gradient_oracle_statistics,
)


def test_d_plus_one_classical_probes_recover_and_emulate_the_entire_oracle():
    rng = np.random.default_rng(31)
    x = rng.normal(size=(9, 4))
    y = rng.normal(size=(9, 2))

    def oracle(weights, bias):
        return evaluate_linear_gradient_oracle(x, y, weights, bias)

    recovery = recover_linear_gradient_oracle_statistics(y, 4, oracle)
    direct = linear_gradient_oracle_statistics(x, y)
    assert recovery.query_count == 5
    assert recovery.symmetry_error < 1e-12
    assert recovery.reproduction_error < 1e-12
    assert np.allclose(recovery.statistics.input_gram, direct.input_gram)
    assert np.allclose(recovery.statistics.input_sum, direct.input_sum)
    assert np.allclose(recovery.statistics.target_cross, direct.target_cross)

    for _ in range(6):
        theta = rng.normal(size=(2, 4))
        bias = rng.normal(size=2)
        observed = oracle(theta, bias)
        emulated = evaluate_linear_gradient_oracle_from_statistics(
            recovery.statistics, theta, bias
        )
        assert np.allclose(observed.weight_gradient, emulated.weight_gradient)
        assert np.allclose(observed.bias_gradient, emulated.bias_gradient)


def test_quotient_invariants_and_constructed_representative_preserve_the_fibre():
    rng = np.random.default_rng(37)
    x = rng.normal(size=(7, 3))
    y = rng.normal(size=(7, 1))
    statistics = linear_gradient_oracle_statistics(x, y)
    invariants = known_target_orbit_invariants(x, y)
    representative = construct_known_target_orbit_representative(statistics, y)

    assert invariants.orthogonal_complement_dimension == 5
    assert np.allclose(
        invariants.constrained_component.T @ invariants.constrained_component
        + invariants.residual_gram,
        statistics.input_gram,
    )
    assert linear_gradient_oracles_equivalent(x, representative, y, atol=1e-8)
    assert np.allclose(
        target_constraint_matrix(y).T @ representative,
        target_constraint_matrix(y).T @ x,
        atol=1e-8,
    )


def test_inconsistent_callback_is_rejected_instead_of_silently_certified():
    targets = np.arange(4, dtype=float).reshape(-1, 1)

    def invalid_oracle(weights, bias):
        from qrecon.theory.known_target_collisions import LinearGradientOracleValue

        return LinearGradientOracleValue(
            np.zeros_like(weights), np.zeros_like(bias)
        )

    with pytest.raises(ArithmeticError, match="reproduce"):
        recover_linear_gradient_oracle_statistics(targets, 2, invalid_oracle)
