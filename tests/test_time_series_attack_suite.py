from __future__ import annotations

import pytest
import torch

from qrecon.attacks import (
    linear_trend_penalty,
    periodicity_penalty,
    resolution_consistency_penalty,
)
from qrecon.benchmarks import (
    ModernAttackVariant,
    ModernTimeSeriesAttackManifest,
    run_modern_timeseries_attack_suite,
    standard_modern_attack_variants,
)
from qrecon.metrics import reconstruction_metrics


def test_time_series_regularizers_have_expected_zero_cases():
    trend = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    periodic = torch.tensor([[1.0, 2.0, 1.0, 2.0]])
    constant = torch.ones(1, 6, 2)
    assert linear_trend_penalty(trend) == pytest.approx(0.0, abs=1e-7)
    assert periodicity_penalty(periodic, 2) == pytest.approx(0.0, abs=1e-7)
    assert resolution_consistency_penalty(constant, 2) == pytest.approx(0.0, abs=1e-7)


def test_time_series_regularizers_backpropagate():
    sequence = torch.randn(2, 8, 3, requires_grad=True)
    penalty = (
        linear_trend_penalty(sequence)
        + periodicity_penalty(sequence, 2)
        + resolution_consistency_penalty(sequence, 2)
    )
    gradient = torch.autograd.grad(penalty, sequence)[0]
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


def test_timeseries_metrics_report_smape_percent():
    reference = torch.tensor([[1.0, 0.0, -1.0]])
    estimate = torch.tensor([[1.0, 0.0, 1.0]])
    metrics = reconstruction_metrics(reference, estimate, mode="timeseries")
    # Per element: 0%, 0%, 200%; mean = 66.666...%.
    assert metrics["smape_percent"] == pytest.approx(200.0 / 3.0)


def test_standard_attack_variants_cover_required_baselines():
    variants = standard_modern_attack_variants(period=2)
    assert {variant.name for variant in variants} == {
        "dlg_l2",
        "invg_cosine",
        "qrecon_hybrid",
        "temporal_prior_hybrid",
    }
    temporal = next(variant for variant in variants if variant.name == "temporal_prior_hybrid")
    assert temporal.overrides["periodicity_period"] == 2


def test_modern_attack_suite_runs_paired_tiny_variants():
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
    variants = (
        ModernAttackVariant(
            "dlg_l2",
            {"match_mode": "l2", "regularization": 0.0},
        ),
        ModernAttackVariant(
            "temporal_prior_hybrid",
            {
                "match_mode": "hybrid",
                "regularization": 0.0,
                "trend_regularization": 1e-3,
                "periodicity_regularization": 1e-3,
                "periodicity_period": 2,
                "low_resolution_regularization": 1e-3,
                "low_resolution_factor": 2,
            },
        ),
    )
    report = run_modern_timeseries_attack_suite(manifest, variants)
    assert report.victim_class == "PatchTST"
    assert len(report.attempts) == 2
    assert all(attempt.status == "success" for attempt in report.attempts)
    assert set(report.variant_summaries) == {"dlg_l2", "temporal_prior_hybrid"}
    assert not report.quality_gate.required_baselines_present
    assert not report.quality_gate.passed


def test_attack_suite_rejects_duplicate_variant_names():
    manifest = ModernTimeSeriesAttackManifest(
        dataset={"name": "synthetic_time"},
        victim={"architecture": "patchtst"},
        training={"epochs": 1},
        attack={"steps": 1},
        victim_seed=1,
        attack_indices=(0,),
        attack_seeds=(1,),
    )
    with pytest.raises(ValueError, match="variant names must be unique"):
        run_modern_timeseries_attack_suite(
            manifest,
            (
                ModernAttackVariant("same", {}),
                ModernAttackVariant("same", {}),
            ),
        )
