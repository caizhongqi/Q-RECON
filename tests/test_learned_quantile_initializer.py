from __future__ import annotations

import pytest
import torch

from qrecon.attacks import (
    GradientToQuantileNetwork,
    initialize_direct_prior_from_median,
    pinball_quantile_loss,
    quantile_crossing_penalty,
    train_gradient_quantile_network,
)
from qrecon.benchmarks import (
    LearnedQuantileAuxiliaryConfig,
    ModernTimeSeriesAttackManifest,
    run_learned_quantile_ts_inverse_benchmark,
)


def test_pinball_loss_and_crossing_penalty_are_exact():
    predictions = torch.tensor([[[0.0, 1.0, 2.0]]])
    target = torch.tensor([[1.0]])
    loss = pinball_quantile_loss(predictions, target, (0.1, 0.5, 0.9))
    assert loss == pytest.approx((0.1 + 0.0 + 0.1) / 3.0)
    assert quantile_crossing_penalty(predictions) == 0.0
    crossed = torch.tensor([[[2.0, 1.0, 0.0]]])
    assert quantile_crossing_penalty(crossed) == pytest.approx(1.0)


def test_gradient_quantile_network_shapes_and_sorted_inference():
    network = GradientToQuantileNetwork(
        gradient_features=6,
        input_shape=(4,),
        target_shape=(2,),
        hidden_sizes=(8, 4),
        dropout=0.0,
        quantiles=(0.1, 0.5, 0.9),
    )
    network.eval()
    inputs, targets = network.inference(torch.randn(3, 6))
    assert inputs.shape == (3, 4, 3)
    assert targets is not None and targets.shape == (3, 2, 3)
    assert torch.all(inputs[..., 1:] >= inputs[..., :-1])
    assert torch.all(targets[..., 1:] >= targets[..., :-1])


def test_quantile_network_training_and_initialized_prior_are_finite():
    torch.manual_seed(9)
    gradients = torch.randn(12, 5)
    inputs = torch.stack((gradients[:, 0], gradients[:, 1]), dim=1)
    targets = gradients[:, 2:3]
    network, normalizer, report = train_gradient_quantile_network(
        gradients,
        inputs,
        targets,
        hidden_sizes=(8, 4),
        dropout=0.0,
        epochs=2,
        batch_size=4,
        learning_rate=1e-2,
        validation_fraction=0.25,
        seed=11,
    )
    quantiles, _ = network.inference(normalizer.transform(gradients[:1]))
    prior = initialize_direct_prior_from_median(
        quantiles[..., 1],
        mode="timeseries",
        bounded=True,
        jitter_standard_deviation=0.01,
        seed=13,
    )
    assert prior().shape == (1, 2)
    assert torch.isfinite(prior()).all()
    assert report.best_epoch in (1, 2)
    assert report.training_samples == 9
    assert report.validation_samples == 3


def test_auxiliary_config_rejects_private_public_overlap():
    with pytest.raises(ValueError, match="disjoint"):
        LearnedQuantileAuxiliaryConfig(
            victim_training_indices=(0, 1, 2),
            auxiliary_indices=(2, 3, 4, 5, 6, 7, 8, 9),
            minimum_publication_auxiliary_samples=8,
        )


def test_learned_quantile_benchmark_runs_on_tiny_patchtst():
    manifest = ModernTimeSeriesAttackManifest(
        dataset={
            "name": "synthetic_time",
            "max_samples": 12,
            "context": 4,
            "horizon": 1,
        },
        victim={
            "architecture": "patchtst",
            "patch_len": 2,
            "stride": 1,
            "d_model": 2,
            "n_heads": 1,
            "e_layers": 1,
            "d_ff": 4,
            "dropout": 0.0,
            "revin": True,
        },
        training={
            "epochs": 1,
            "batch_size": 2,
            "optimizer": "adam",
            "learning_rate": 1e-3,
        },
        attack={
            "bounded": True,
            "known_target": True,
            "steps": 1,
            "optimizer": "adam",
            "learning_rate": 0.02,
            "gradient_l1_weight": 1.0,
            "quantile_bound_weight": 0.01,
            "gradient_clip_norm": 10.0,
            "record_every": 1,
        },
        victim_seed=17,
        attack_indices=(0,),
        attack_seeds=(101,),
        attack_batch_size=1,
        exact_tolerance=0.1,
        relative_l2_threshold=1.0,
        bootstrap_samples=20,
        minimum_publication_batches=1,
        minimum_publication_attack_seeds=1,
        publication_mode=False,
    )
    auxiliary = LearnedQuantileAuxiliaryConfig(
        victim_training_indices=(0, 1, 2, 3),
        auxiliary_indices=(4, 5, 6, 7, 8, 9, 10, 11),
        hidden_sizes=(8, 4),
        dropout=0.0,
        epochs=1,
        batch_size=4,
        learning_rate=1e-2,
        validation_fraction=0.25,
        jitter_standard_deviation=0.0,
        minimum_publication_auxiliary_samples=8,
    )
    report = run_learned_quantile_ts_inverse_benchmark(manifest, auxiliary)
    assert report.victim_class == "PatchTST"
    assert report.quality_gate.learned_quantile_initializer_present
    assert report.quality_gate.victim_and_auxiliary_splits_disjoint
    assert len(report.attempts) == 1
    assert report.attempts[0].status == "success"
    assert report.initializer_training["training_samples"] == 6
    assert report.initializer_training["validation_samples"] == 2
