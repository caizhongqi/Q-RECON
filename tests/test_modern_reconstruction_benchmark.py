from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from qrecon.attacks import GradientInversionAttack, leak_gradients
from qrecon.benchmarks import (
    ModernTimeSeriesAttackManifest,
    run_modern_timeseries_reconstruction_benchmark,
)
from qrecon.data import load_multivariate_csv
from qrecon.metrics import permutation_invariant_batch_metrics
from qrecon.quantum import DirectPrior


def test_permutation_invariant_batch_metrics_recovers_reversed_batch():
    reference = torch.tensor([[0.0, 1.0], [10.0, 11.0]])
    estimate = torch.tensor([[10.0, 11.0], [0.0, 1.0]])
    report = permutation_invariant_batch_metrics(
        reference, estimate, mode="timeseries", tolerance=1e-6
    )
    assert report.assignment == (1, 0)
    assert report.assignment_total_mse == pytest.approx(0.0)
    assert report.exact_batch_within_tolerance
    assert report.record_success_rate == pytest.approx(1.0)
    assert report.aligned_metrics["mse"] == pytest.approx(0.0)


def test_gradient_inversion_returns_an_objective_selected_best_iterate():
    torch.manual_seed(7)
    model = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 1)).eval()
    reference = torch.tensor([[0.2, -0.4, 0.7]])
    target = torch.tensor([[0.3]])
    observed = leak_gradients(model, reference, target, "forecasting")
    attack = GradientInversionAttack(
        model=model,
        observed_gradients=observed,
        prior=DirectPrior((1, 3), "timeseries", bounded=False),
        task="forecasting",
        mode="timeseries",
        known_target=target,
        target_shape=tuple(target.shape),
        steps=3,
        learning_rate=0.05,
        regularization=0.0,
        layer_weighting="parameter",
        record_every=1,
    )
    result = attack.run()
    assert math.isfinite(result.best_objective)
    assert math.isfinite(result.final_objective)
    assert result.best_objective <= result.final_objective + 1e-12
    assert result.best_step > 0
    assert result.reconstruction.shape == reference.shape


def test_multivariate_csv_loader_preserves_variables_and_normalizes_context(tmp_path):
    path = tmp_path / "ETTm1.csv"
    path.write_text(
        "date,a,b\n"
        "t0,0,10\n"
        "t1,1,12\n"
        "t2,2,14\n"
        "t3,3,16\n"
        "t4,4,18\n"
        "t5,5,20\n",
        encoding="utf-8",
    )
    dataset = load_multivariate_csv(
        path,
        context=3,
        horizon=1,
        max_windows=2,
        stride=1,
    )
    x, y = dataset.tensors
    assert x.shape == (2, 3, 2)
    assert y.shape == (2, 1, 2)
    assert torch.allclose(x.mean(dim=1), torch.zeros(2, 2), atol=1e-6)


def test_modern_transformer_benchmark_runs_one_complete_attack():
    manifest = ModernTimeSeriesAttackManifest(
        dataset={
            "name": "synthetic_time",
            "max_samples": 4,
            "context": 4,
            "horizon": 1,
        },
        victim={
            "architecture": "transformer",
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
            "known_target": True,
            "steps": 1,
            "learning_rate": 0.02,
            "regularization": 0.0,
            "match_mode": "hybrid",
            "layer_weighting": "parameter",
        },
        victim_seed=11,
        attack_indices=(0,),
        attack_seeds=(101,),
        attack_batch_size=1,
        exact_tolerance=0.1,
        relative_l2_threshold=1.0,
        bootstrap_samples=20,
        publication_mode=False,
    )
    report = run_modern_timeseries_reconstruction_benchmark(manifest)
    assert report.victim_class == "TransformerForecaster"
    assert len(report.manifest_sha256) == 64
    assert len(report.dataset_sha256) == 64
    assert len(report.model_sha256) == 64
    assert len(report.attempts) == 1
    assert report.attempts[0].status == "success"
    assert report.selected_attempt_indices == (0,)
    assert report.summary["selected_successful_batches"] == 1
    assert not report.quality_gate.passed


def test_modern_manifest_rejects_duplicate_restarts():
    with pytest.raises(ValueError, match="attack_seeds must be unique"):
        ModernTimeSeriesAttackManifest(
            dataset={"name": "synthetic_time"},
            victim={"architecture": "patchtst"},
            training={"epochs": 1},
            attack={"steps": 1},
            victim_seed=1,
            attack_indices=(0,),
            attack_seeds=(3, 3),
        )
