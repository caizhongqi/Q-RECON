import numpy as np
import pytest

from qrecon.theory.known_target_collisions import (
    LinearGradientOracleValue,
    evaluate_linear_gradient_oracle,
    evaluate_linear_gradient_oracle_from_statistics,
    linear_gradient_oracle_statistics,
)
from qrecon.theory.known_target_packed_probes import (
    build_packed_linear_gradient_probe_plan,
    recover_linear_gradient_oracle_statistics_packed,
)


def test_plan_packs_input_basis_directions_across_output_rows():
    plan = build_packed_linear_gradient_probe_plan(7, 3)
    assert plan.query_count == 4
    assert np.count_nonzero(plan.weight_queries[0]) == 0
    stacked = np.concatenate(plan.weight_queries[1:], axis=0)
    assigned = stacked[np.any(stacked != 0.0, axis=1)]
    assert np.array_equal(assigned, np.eye(7))


def test_packed_queries_recover_and_emulate_the_entire_oracle():
    rng = np.random.default_rng(43)
    x = rng.normal(size=(10, 7))
    y = rng.normal(size=(10, 3))

    def oracle(weights, bias):
        return evaluate_linear_gradient_oracle(x, y, weights, bias)

    recovery = recover_linear_gradient_oracle_statistics_packed(y, 7, oracle)
    direct = linear_gradient_oracle_statistics(x, y)
    assert recovery.query_count == 1 + 3
    assert recovery.symmetry_error < 1e-12
    assert recovery.reproduction_error < 1e-12
    assert np.allclose(recovery.statistics.input_gram, direct.input_gram)
    assert np.allclose(recovery.statistics.input_sum, direct.input_sum)
    assert np.allclose(recovery.statistics.target_cross, direct.target_cross)

    for _ in range(8):
        weights = rng.normal(size=(3, 7))
        bias = rng.normal(size=3)
        direct_value = oracle(weights, bias)
        emulated = evaluate_linear_gradient_oracle_from_statistics(
            recovery.statistics, weights, bias
        )
        assert np.allclose(direct_value.weight_gradient, emulated.weight_gradient)
        assert np.allclose(direct_value.bias_gradient, emulated.bias_gradient)


def test_two_queries_suffice_when_output_dimension_covers_all_inputs():
    rng = np.random.default_rng(47)
    x = rng.normal(size=(8, 3))
    y = rng.normal(size=(8, 5))
    recovery = recover_linear_gradient_oracle_statistics_packed(
        y,
        3,
        lambda weights, bias: evaluate_linear_gradient_oracle(
            x, y, weights, bias
        ),
    )
    assert recovery.query_count == 2


def test_inconsistent_packed_transcript_is_rejected():
    targets = np.arange(5, dtype=float).reshape(-1, 1)

    def invalid(weights, bias):
        return LinearGradientOracleValue(
            np.zeros_like(weights), np.zeros_like(bias)
        )

    with pytest.raises(ArithmeticError, match="reproduce"):
        recover_linear_gradient_oracle_statistics_packed(targets, 2, invalid)
