from __future__ import annotations

import pytest
import torch
from torch import nn

from qrecon.attacks import (
    TSInverseStyleAttack,
    joint_forecasting_sequence,
    l1_gradient_distance,
    quantile_bound_hinge_penalty,
    temporal_total_variation_l1,
)
from qrecon.benchmarks import (
    ModernTimeSeriesAttackManifest,
    run_ts_inverse_style_benchmark,
)
from qrecon.quantum import DirectPrior


def test_l1_gradient_distance_matches_sum_reduction():
    candidate = (torch.tensor([1.0, -1.0]), torch.tensor([[2.0]]))
    observed = (torch.tensor([0.0, 1.0]), torch.tensor([[-1.0]]))
    assert l1_gradient_distance(candidate, observed) == pytest.approx(6.0)


def test_joint_forecasting_sequence_uses_declared_channel():
    inputs = torch.tensor(
        [[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]]
    )
    targets = torch.tensor([[[4.0, 40.0], [5.0, 50.0]]])
    assert torch.equal(
        joint_forecasting_sequence(inputs, targets, channel=1),
        torch.tensor([[10.0, 20.0, 30.0, 40.0, 50.0]]),
    )


def test_quantile_hinge_and_temporal_variation_are_auditable():
    sequence = torch.tensor([[0.0, 2.0]])
    lower = torch.tensor([[0.0, 0.0]])
    upper = torch.tensor([[1.0, 1.0]])
    assert quantile_bound_hinge_penalty(sequence, lower, upper) == pytest.approx(0.25)
    assert temporal_total_variation_l1(sequence) == pytest.approx(2.0)


def test_ts_inverse_style_attack_runs_with_finite_best_iterate():
    torch.manual_seed(7)
    model = nn.Sequential(
        nn.Linear(4, 3),
        nn.Tanh(),
        nn.Linear(3, 2),
    ).eval()
    true_x = torch.tensor([[0.3, -0.4, 0.2, 0.1]])
    target = torch.tensor([[0.1, -0.2]])
    loss = (model(true_x) - target).square().mean()
    observed = tuple(
        gradient.detach().clone()
        for gradient in torch.autograd.grad(loss, tuple(model.parameters()))
    )
    prior = DirectPrior(tuple(true_x.shape), "timeseries", bounded=False)
    attack = TSInverseStyleAttack(
        model,
        observed,
        prior,
        known_target=target,
        target_shape=tuple(target.shape),
        steps=2,
        learning_rate=0.02,
        trend_weight=1e-4,
        periodicity_weight=1e-4,
        periodicity_period=2,
        low_resolution_weight=1e-4,
        low_resolution_factor=2,
        gradient_clip_norm=10.0,
        record_every=1,
    )
    result = attack.run()
    assert result.reconstruction.shape == true_x.shape
    assert torch.isfinite(result.reconstruction).all()
    assert result.best_objective <= result.final_objective + 1e-12
    assert result.best_gradient_match >= 0.0
    assert result.history
    assert "gradient_l1" in result.history[-1]


def test_ts_inverse_style_benchmark_preserves_provenance_and_failures():
    manifest = ModernTimeSeriesAttackManifest(
        dataset={
            "name": "synthetic_time",
            "max_samples": 4,
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
            "prior": "direct",
            "bounded": True,
            "known_target": True,
            "steps": 1,
            "optimizer": "adam",
            "learning_rate": 0.02,
            "gradient_l1_weight": 1.0,
            "trend_weight": 1e-4,
            "periodicity_weight": 1e-4,
            "periodicity_period": 2,
            "low_resolution_weight": 1e-4,
            "low_resolution_factor": 2,
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
        publication_mode=False,
    )
    report = run_ts_inverse_style_benchmark(manifest)
    assert report.victim_class == "PatchTST"
    assert report.provenance["commit"] == (
        "2015946906a693f836e6418cdeb3b64a3f6f2d6e"
    )
    assert len(report.attempts) == 1
    assert report.attempts[0].status == "success"
    assert report.summary["failed_restart_attempts"] == 0
    assert report.quality_gate.official_objective_components_present
    assert not report.quality_gate.full_ts_inverse_reproduction
