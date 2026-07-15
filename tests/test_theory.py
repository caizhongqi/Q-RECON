import numpy as np
import pytest

from qrecon.theory import (
    AlgorithmCost,
    all_pairs_epsilon_private_uniform_bound,
    bayes_equivalence_reconstruction_success,
    bayes_reconstruction_success,
    binary_helstrom_success,
    channel_bayes_equivalence_reconstruction_success,
    channel_bayes_reconstruction_success,
    classical_queries_for_success,
    classical_success_without_replacement,
    compare_algorithm_costs,
    conditional_min_entropy_bits,
    grover_queries_for_success,
    grover_success,
    maximum_quantum_query_cost_for_advantage,
    minimum_instances_for_quantum_advantage,
    optimal_standard_grover_iterations,
    oracle_error_success_lower_bound,
    postprocess_channel,
    uniform_fibre_success,
)


def test_deterministic_fibre_bound_and_uniform_corollary():
    observation = {"a": 0, "b": 0, "c": 1}
    prior = {"a": 0.6, "b": 0.1, "c": 0.3}
    assert bayes_reconstruction_success(prior, observation) == pytest.approx(0.9)
    assert uniform_fibre_success(observation) == pytest.approx(2 / 3)
    assert conditional_min_entropy_bits(0.25) == pytest.approx(2.0)

    collapsed_prior = {"a": 0.35, "b": 0.25, "c": 0.4}
    collapsed_observation = {"a": "same", "b": "same", "c": "same"}
    target_class = {"a": "equivalent", "b": "equivalent", "c": "other"}
    assert bayes_reconstruction_success(
        collapsed_prior, collapsed_observation
    ) == pytest.approx(0.4)
    assert bayes_equivalence_reconstruction_success(
        collapsed_prior, collapsed_observation, target_class
    ) == pytest.approx(0.6)


def test_noisy_channel_formula_and_data_processing():
    prior = {0: 0.5, 1: 0.5}
    channel = {
        0: {"left": 0.8, "right": 0.2},
        1: {"left": 0.3, "right": 0.7},
    }
    success = channel_bayes_reconstruction_success(prior, channel)
    assert success == pytest.approx(0.75)

    erase = {
        "left": {"erased": 1.0},
        "right": {"erased": 1.0},
    }
    processed = postprocess_channel(channel, erase)
    assert channel_bayes_reconstruction_success(prior, processed) == pytest.approx(0.5)
    assert channel_bayes_equivalence_reconstruction_success(
        prior, processed, {0: "same", 1: "same"}
    ) == pytest.approx(1.0)


def test_helstrom_bound_for_identical_and_orthogonal_states():
    zero = np.array([[1.0, 0.0], [0.0, 0.0]])
    one = np.array([[0.0, 0.0], [0.0, 1.0]])
    assert binary_helstrom_success(zero, zero) == pytest.approx(0.5)
    assert binary_helstrom_success(zero, zero, prior0=0.7) == pytest.approx(0.7)
    assert binary_helstrom_success(zero, one) == pytest.approx(1.0)


def test_all_pairs_privacy_bound_has_expected_limits():
    assert all_pairs_epsilon_private_uniform_bound(8, 0.0) == pytest.approx(1 / 8)
    assert all_pairs_epsilon_private_uniform_bound(1, 10.0) == 1.0
    assert all_pairs_epsilon_private_uniform_bound(8, 2.0) < 1.0
    assert all_pairs_epsilon_private_uniform_bound(8, 1_000.0) == 1.0


def test_classical_and_grover_exact_small_case():
    assert classical_success_without_replacement(4, 1, 1) == pytest.approx(0.25)
    assert classical_queries_for_success(4, 1, 0.75) == 3
    assert grover_success(4, 1, 1) == pytest.approx(1.0)
    assert grover_queries_for_success(4, 1, 0.99) == 1
    assert grover_queries_for_success(2, 1, 0.9) is None
    assert optimal_standard_grover_iterations(5, 3) == 0


def test_cost_break_even_and_oracle_error_bound():
    classical = AlgorithmCost(
        setup_cost=0.0,
        fixed_instance_cost=2.0,
        queries=100,
        cost_per_query=1.0,
    )
    quantum = AlgorithmCost(
        setup_cost=500.0,
        fixed_instance_cost=5.0,
        queries=10,
        cost_per_query=3.0,
    )

    assert not compare_algorithm_costs(classical, quantum, instances=1).quantum_advantage
    threshold = minimum_instances_for_quantum_advantage(classical, quantum)
    assert threshold == 8
    assert compare_algorithm_costs(classical, quantum, instances=threshold).quantum_advantage
    assert maximum_quantum_query_cost_for_advantage(
        classical, quantum, instances=10
    ) == pytest.approx(4.7)
    assert oracle_error_success_lower_bound(0.95, 10, 0.002) == pytest.approx(0.93)


def test_invalid_probability_inputs_are_rejected():
    with pytest.raises(ValueError):
        bayes_reconstruction_success({"x": -1.0}, {"x": 0})
    with pytest.raises(ValueError):
        channel_bayes_reconstruction_success({0: 1.0}, {0: {"x": 0.9}})
    with pytest.raises(ValueError):
        conditional_min_entropy_bits(0.0)
    with pytest.raises(ValueError):
        oracle_error_success_lower_bound(1.0, -1, 0.0)
