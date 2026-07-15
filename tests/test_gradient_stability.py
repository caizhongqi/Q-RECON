import math

import numpy as np
import pytest

from qrecon.theory import (
    LinearGradientOracleStatistics,
    adaptive_gaussian_transcript_success_upper_bound,
    equal_covariance_gaussian_binary_success,
    evaluate_linear_gradient_oracle_from_statistics,
    gradient_oracle_statistic_distance,
    linear_gradient_oracle_statistics,
    necessary_gaussian_queries_for_binary_success,
    target_stabilizer_rotation,
    uniform_gradient_query_difference_bound,
)


def test_exact_orbit_collision_has_zero_uniform_query_distance():
    rng = np.random.default_rng(41)
    targets = rng.normal(size=(7, 2))
    inputs = rng.normal(size=(7, 3))
    rotation = target_stabilizer_rotation(targets, 0.37)
    transformed = rotation @ inputs

    left = linear_gradient_oracle_statistics(inputs, targets)
    right = linear_gradient_oracle_statistics(transformed, targets)
    distance = gradient_oracle_statistic_distance(left, right)
    bound = uniform_gradient_query_difference_bound(
        left,
        right,
        max_weight_operator_norm=5.0,
        max_bias_l2=3.0,
    )

    assert distance.input_gram_frobenius == pytest.approx(0.0, abs=1e-10)
    assert distance.input_sum_l2 == pytest.approx(0.0, abs=1e-10)
    assert distance.target_cross_frobenius == pytest.approx(0.0, abs=1e-10)
    assert bound.combined_l2 == pytest.approx(0.0, abs=1e-10)
    assert adaptive_gaussian_transcript_success_upper_bound(
        0.0, 10_000, 1.0
    ) == pytest.approx(0.5)
    assert necessary_gaussian_queries_for_binary_success(0.75, 0.0, 1.0) is None


def test_uniform_parameter_domain_bound_dominates_actual_query_difference():
    rng = np.random.default_rng(9)
    targets = rng.normal(size=(6, 2))
    left_inputs = rng.normal(size=(6, 4))
    right_inputs = left_inputs + 0.03 * rng.normal(size=left_inputs.shape)
    left = linear_gradient_oracle_statistics(left_inputs, targets)
    right = linear_gradient_oracle_statistics(right_inputs, targets)

    weight_radius = 2.5
    bias_radius = 1.75
    certificate = uniform_gradient_query_difference_bound(
        left,
        right,
        max_weight_operator_norm=weight_radius,
        max_bias_l2=bias_radius,
    )

    for _ in range(50):
        weights = rng.normal(size=(2, 4))
        operator_norm = np.linalg.norm(weights, ord=2)
        weights *= rng.uniform(0.0, weight_radius) / operator_norm
        bias = rng.normal(size=2)
        bias *= rng.uniform(0.0, bias_radius) / np.linalg.norm(bias)

        left_value = evaluate_linear_gradient_oracle_from_statistics(
            left, weights, bias
        )
        right_value = evaluate_linear_gradient_oracle_from_statistics(
            right, weights, bias
        )
        weight_difference = np.linalg.norm(
            left_value.weight_gradient - right_value.weight_gradient,
            ord="fro",
        )
        bias_difference = np.linalg.norm(
            left_value.bias_gradient - right_value.bias_gradient
        )
        combined = math.hypot(weight_difference, bias_difference)

        assert weight_difference <= certificate.weight_gradient_frobenius + 1e-12
        assert bias_difference <= certificate.bias_gradient_l2 + 1e-12
        assert combined <= certificate.combined_l2 + 1e-12


def test_equal_covariance_gaussian_binary_formula_has_expected_limits():
    assert equal_covariance_gaussian_binary_success(0.0, 2.0) == pytest.approx(0.5)
    # Distance 2 sigma gives Phi(1).
    assert equal_covariance_gaussian_binary_success(4.0, 2.0) == pytest.approx(
        0.8413447460685429
    )
    assert equal_covariance_gaussian_binary_success(1e6, 1.0) == pytest.approx(1.0)


def test_adaptive_gaussian_bound_implies_a_necessary_query_count():
    difference = 0.2
    sigma = 1.0
    target = 0.6
    necessary = necessary_gaussian_queries_for_binary_success(
        target, difference, sigma
    )
    assert necessary == 4
    assert adaptive_gaussian_transcript_success_upper_bound(
        difference, necessary - 1, sigma
    ) < target
    assert adaptive_gaussian_transcript_success_upper_bound(
        difference, necessary, sigma
    ) == pytest.approx(target)
    assert necessary_gaussian_queries_for_binary_success(0.5, difference, sigma) == 0


def test_stability_contract_rejects_mismatched_statistics_and_invalid_inputs():
    left = LinearGradientOracleStatistics(
        batch_size=2,
        input_dimension=1,
        output_dimension=1,
        input_gram=np.array([[1.0]]),
        input_sum=np.array([0.0]),
        target_cross=np.array([[0.0]]),
        target_sum=np.array([0.0]),
    )
    different_target_sum = LinearGradientOracleStatistics(
        batch_size=2,
        input_dimension=1,
        output_dimension=1,
        input_gram=np.array([[1.0]]),
        input_sum=np.array([0.0]),
        target_cross=np.array([[0.0]]),
        target_sum=np.array([1.0]),
    )
    with pytest.raises(ValueError, match="target sum"):
        gradient_oracle_statistic_distance(left, different_target_sum)
    with pytest.raises(ValueError):
        uniform_gradient_query_difference_bound(
            left,
            left,
            max_weight_operator_norm=-1.0,
            max_bias_l2=0.0,
        )
    with pytest.raises(ValueError):
        adaptive_gaussian_transcript_success_upper_bound(1.0, -1, 1.0)
    with pytest.raises(ValueError):
        equal_covariance_gaussian_binary_success(1.0, 0.0)
    with pytest.raises(ValueError):
        necessary_gaussian_queries_for_binary_success(0.49, 1.0, 1.0)
