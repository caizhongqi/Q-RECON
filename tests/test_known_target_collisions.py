import numpy as np
import pytest

from qrecon.theory.known_target_collisions import (
    construct_known_target_rotation_collision,
    evaluate_linear_gradient_oracle,
    evaluate_linear_gradient_oracle_from_statistics,
    known_target_orbit_report,
    linear_gradient_oracle_statistics,
    linear_gradient_oracles_equivalent,
    recover_target_stabilizing_orthogonal_map,
    target_constraint_matrix,
    target_stabilizer_basis,
    target_stabilizer_reflection,
    target_stabilizer_rotation,
)


def test_sufficient_statistics_reproduce_direct_gradient_queries():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(7, 3))
    y = rng.normal(size=(7, 2))
    theta = rng.normal(size=(2, 3))
    bias = rng.normal(size=2)
    statistics = linear_gradient_oracle_statistics(x, y)
    direct = evaluate_linear_gradient_oracle(x, y, theta, bias)
    reduced = evaluate_linear_gradient_oracle_from_statistics(statistics, theta, bias)
    assert np.allclose(direct.weight_gradient, reduced.weight_gradient)
    assert np.allclose(direct.bias_gradient, reduced.bias_gradient)


def test_target_stabilizer_rotation_is_orthogonal_and_fixes_known_targets():
    y = np.array([[-1.0], [0.0], [1.0], [2.0], [3.0], [4.0]])
    q = target_stabilizer_rotation(y, 0.37)
    constraints = target_constraint_matrix(y)
    assert np.allclose(q.T @ q, np.eye(6), atol=1e-12)
    assert np.allclose(q @ constraints, constraints, atol=1e-12)
    assert target_stabilizer_basis(y).shape == (6, 4)


def test_rotation_collision_preserves_entire_gradient_oracle_not_one_probe_only():
    rng = np.random.default_rng(11)
    x = rng.normal(size=(8, 3))
    y = rng.normal(size=(8, 2))
    q = target_stabilizer_rotation(y, 0.53)
    transformed = q @ x
    assert linear_gradient_oracles_equivalent(x, transformed, y)
    assert np.linalg.norm(transformed - x) > 1e-4

    for _ in range(10):
        theta = rng.normal(size=(2, 3))
        bias = rng.normal(size=2)
        left = evaluate_linear_gradient_oracle(x, y, theta, bias)
        right = evaluate_linear_gradient_oracle(transformed, y, theta, bias)
        assert np.allclose(left.weight_gradient, right.weight_gradient, atol=1e-11)
        assert np.allclose(left.bias_gradient, right.bias_gradient, atol=1e-11)


def test_collision_report_and_orbit_dimension_formula():
    rng = np.random.default_rng(17)
    x = rng.normal(size=(6, 2))
    y = np.arange(6, dtype=float).reshape(-1, 1)
    theta = rng.normal(size=(1, 2))
    bias = np.array([0.2])
    report = construct_known_target_rotation_collision(x, y, theta, bias, 0.2)
    assert report.orthogonality_error < 1e-12
    assert report.fixed_constraint_error < 1e-12
    assert report.statistic_error < 1e-11
    assert report.probe_weight_gradient_error < 1e-11
    assert report.probe_bias_gradient_error < 1e-11
    assert report.input_displacement > 0.0
    assert report.orbit.orthogonal_complement_dimension == 4
    assert report.orbit.projected_input_rank == 2
    assert report.orbit.continuous_orbit_dimension == 5
    assert report.orbit.has_continuous_family


def test_one_dimensional_complement_has_discrete_reflection_but_no_continuous_orbit():
    y = np.array([[0.0], [1.0], [2.0]])
    x = np.array([[0.1], [0.7], [-0.4]])
    report = known_target_orbit_report(x, y)
    assert report.orthogonal_complement_dimension == 1
    assert report.has_nontrivial_collision
    assert not report.has_continuous_family
    reflected = target_stabilizer_reflection(y) @ x
    assert linear_gradient_oracles_equivalent(x, reflected, y)
    assert not np.allclose(x, reflected)
    with pytest.raises(ValueError, match="two distinct"):
        target_stabilizer_rotation(y, 0.1)


def test_full_rank_constraints_remove_target_stabilizer_ambiguity():
    y = np.array([[0.0], [1.0]])
    x = np.array([[1.0], [2.0]])
    report = known_target_orbit_report(x, y)
    assert report.orthogonal_complement_dimension == 0
    assert not report.has_nontrivial_collision
    with pytest.raises(ValueError):
        target_stabilizer_reflection(y)


def test_complete_characterization_recovers_an_orthogonal_stabilizer_map():
    rng = np.random.default_rng(23)
    x = rng.normal(size=(7, 3))
    y = rng.normal(size=(7, 1))
    q = target_stabilizer_rotation(y, -0.41, axes=(1, 3))
    transformed = q @ x
    recovered = recover_target_stabilizing_orthogonal_map(x, transformed, y)
    constraints = target_constraint_matrix(y)
    assert np.allclose(recovered.T @ recovered, np.eye(7), atol=1e-8)
    assert np.allclose(recovered @ constraints, constraints, atol=1e-8)
    assert np.allclose(recovered @ x, transformed, atol=1e-8)


def test_non_equivalent_statistics_are_rejected_by_map_recovery():
    y = np.array([[0.0], [1.0], [2.0], [3.0]])
    x = np.arange(8, dtype=float).reshape(4, 2)
    altered = x.copy()
    altered[0, 0] += 0.25
    assert not linear_gradient_oracles_equivalent(x, altered, y)
    with pytest.raises(ValueError, match="same full gradient oracle"):
        recover_target_stabilizing_orthogonal_map(x, altered, y)


def test_empty_feature_or_output_dimensions_are_rejected():
    with pytest.raises(ValueError, match="feature"):
        linear_gradient_oracle_statistics(np.empty((3, 0)), np.ones((3, 1)))
    with pytest.raises(ValueError, match="output"):
        linear_gradient_oracle_statistics(np.ones((3, 1)), np.empty((3, 0)))
