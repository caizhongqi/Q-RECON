import numpy as np
import pytest

from qrecon.theory.known_target_collisions import (
    linear_gradient_oracle_statistics,
    target_stabilizer_rotation,
)
from qrecon.theory.linear_training_transcripts import (
    LinearOptimizerConfig,
    evaluate_linear_loss_from_statistics,
    maximum_training_transcript_difference,
    simulate_linear_training,
    simulate_linear_training_from_statistics,
)


def test_loss_is_exactly_reconstructed_from_gradient_oracle_statistics():
    rng = np.random.default_rng(59)
    x = rng.normal(size=(7, 3))
    y = rng.normal(size=(7, 2))
    weights = rng.normal(size=(2, 3))
    bias = rng.normal(size=2)
    residual = x @ weights.T + bias[None, :] - y
    direct = float(np.sum(residual**2) / (2 * x.shape[0]))
    reduced = evaluate_linear_loss_from_statistics(
        linear_gradient_oracle_statistics(x, y),
        float(np.sum(y**2)),
        weights,
        bias,
    )
    assert reduced == pytest.approx(direct)


@pytest.mark.parametrize("optimizer", ["sgd", "momentum", "adam"])
def test_target_stabilizer_fibre_produces_identical_full_training_transcripts(optimizer):
    rng = np.random.default_rng(61)
    x = rng.normal(size=(8, 3))
    y = rng.normal(size=(8, 2))
    transformed = target_stabilizer_rotation(y, 0.31, axes=(0, 2)) @ x
    initial_weights = rng.normal(size=(2, 3))
    initial_bias = rng.normal(size=2)
    weight_noise = tuple(rng.normal(scale=1e-3, size=(2, 3)) for _ in range(12))
    bias_noise = tuple(rng.normal(scale=1e-3, size=2) for _ in range(12))
    config = LinearOptimizerConfig(
        optimizer=optimizer,
        learning_rate=0.03,
        weight_decay=0.01,
        decay_bias=True,
    )
    left = simulate_linear_training(
        x,
        y,
        initial_weights,
        initial_bias,
        steps=12,
        config=config,
        additive_weight_noise=weight_noise,
        additive_bias_noise=bias_noise,
    )
    right = simulate_linear_training(
        transformed,
        y,
        initial_weights,
        initial_bias,
        steps=12,
        config=config,
        additive_weight_noise=weight_noise,
        additive_bias_noise=bias_noise,
    )
    assert maximum_training_transcript_difference(left, right) < 1e-10


def test_statistics_emulator_matches_direct_training_and_optimizer_state():
    rng = np.random.default_rng(67)
    x = rng.normal(size=(9, 4))
    y = rng.normal(size=(9, 1))
    weights = rng.normal(size=(1, 4))
    bias = rng.normal(size=1)
    config = LinearOptimizerConfig(optimizer="adam", learning_rate=0.02)
    direct = simulate_linear_training(
        x, y, weights, bias, steps=9, config=config
    )
    reduced = simulate_linear_training_from_statistics(
        linear_gradient_oracle_statistics(x, y),
        float(np.sum(y**2)),
        weights,
        bias,
        steps=9,
        config=config,
    )
    assert maximum_training_transcript_difference(direct, reduced) == pytest.approx(0.0)


def test_different_oracle_statistics_change_the_training_transcript():
    y = np.arange(6, dtype=float).reshape(-1, 1)
    x = np.linspace(-1.0, 1.0, 12).reshape(6, 2)
    altered = x.copy()
    altered[0, 0] += 0.5
    config = LinearOptimizerConfig(optimizer="momentum")
    left = simulate_linear_training(
        x, y, np.zeros((1, 2)), np.zeros(1), steps=4, config=config
    )
    right = simulate_linear_training(
        altered, y, np.zeros((1, 2)), np.zeros(1), steps=4, config=config
    )
    assert maximum_training_transcript_difference(left, right) > 1e-6


def test_optimizer_and_noise_contract_validation():
    with pytest.raises(ValueError):
        LinearOptimizerConfig(optimizer="unknown")
    statistics = linear_gradient_oracle_statistics(
        np.ones((3, 1)), np.ones((3, 1))
    )
    with pytest.raises(ValueError, match="one matrix per step"):
        simulate_linear_training_from_statistics(
            statistics,
            3.0,
            np.zeros((1, 1)),
            np.zeros(1),
            steps=2,
            additive_weight_noise=(np.zeros((1, 1)),),
        )
