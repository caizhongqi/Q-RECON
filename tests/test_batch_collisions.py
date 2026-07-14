import numpy as np
import pytest

from qrecon.theory.batch_collisions import (
    construct_linear_batch_collision,
    linear_squared_loss_gradients,
    symmetric_pair_mixing,
    validate_batch_mixing_matrix,
)


def test_batch_mixing_constructs_exact_nontrivial_gradient_collision():
    rng = np.random.default_rng(12)
    batch, input_dimension, output_dimension = 4, 3, 2
    theta = rng.normal(size=(output_dimension, input_dimension))
    bias = rng.normal(size=output_dimension)
    inputs = rng.normal(size=(batch, input_dimension))
    targets = rng.normal(size=(batch, output_dimension))
    mixing = symmetric_pair_mixing(batch, 0, 1, alpha=0.2)

    mixed_inputs, mixed_targets, report = construct_linear_batch_collision(
        theta, bias, inputs, targets, mixing
    )
    original = linear_squared_loss_gradients(theta, bias, inputs, targets)
    transformed = linear_squared_loss_gradients(theta, bias, mixed_inputs, mixed_targets)

    assert report.nontrivial
    assert report.input_change_frobenius > 0.0
    assert np.allclose(original.weight_gradient, transformed.weight_gradient, atol=1e-10)
    assert np.allclose(original.bias_gradient, transformed.bias_gradient, atol=1e-10)


def test_continuum_of_pair_mixings_produces_distinct_collisions():
    theta = np.array([[1.0, -2.0]])
    bias = np.array([0.25])
    inputs = np.array([[0.0, 1.0], [2.0, -1.0]])
    targets = np.array([[1.0], [-0.5]])

    first = construct_linear_batch_collision(
        theta, bias, inputs, targets, symmetric_pair_mixing(2, 0, 1, 0.1)
    )[0]
    second = construct_linear_batch_collision(
        theta, bias, inputs, targets, symmetric_pair_mixing(2, 0, 1, 0.2)
    )[0]
    assert not np.allclose(first, second)
    assert np.all(first >= np.minimum(inputs[0], inputs[1]))
    assert np.all(first <= np.maximum(inputs[0], inputs[1]))


def test_invalid_mixing_is_rejected():
    with pytest.raises(ValueError):
        validate_batch_mixing_matrix(np.array([[1.0, 1.0], [0.0, 1.0]]))
    with pytest.raises(ValueError):
        symmetric_pair_mixing(2, 0, 1, 0.5)
