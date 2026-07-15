from __future__ import annotations

import pytest
import torch

from qrecon.attacks import GradientReleaseSpec
from qrecon.benchmarks.channel_permutation_release import (
    analyze_channel_permutation_releases,
    run_channel_permutation_release_benchmark,
)
from qrecon.benchmarks.modern_timeseries_reconstruction import (
    ModernTimeSeriesAttackManifest,
)
from qrecon.models import build_forecasting_model


def _model() -> torch.nn.Module:
    torch.manual_seed(71)
    return build_forecasting_model(
        6,
        2,
        3,
        {
            "architecture": "itransformer",
            "d_model": 6,
            "n_heads": 2,
            "e_layers": 1,
            "d_ff": 12,
            "dropout": 0.0,
            "revin": True,
            "revin_affine": False,
        },
    ).eval()


def _variants() -> dict[str, GradientReleaseSpec]:
    return {
        "full": GradientReleaseSpec(),
        "clipped": GradientReleaseSpec(clip_norm=0.5),
        "quantized": GradientReleaseSpec(
            quantization_bits=8,
            quantization_scale=1e-3,
        ),
        "gaussian": GradientReleaseSpec(noise_std=0.05, noise_seed=2026),
        "partial": GradientReleaseSpec(visible_parameter_indices=(0,)),
    }


def test_equal_gradient_fibre_survives_release_postprocessing():
    torch.manual_seed(73)
    inputs = torch.randn(1, 6, 3)
    targets = torch.randn(1, 2, 3)
    point = analyze_channel_permutation_releases(
        _model(),
        inputs,
        targets,
        (2, 0, 1),
        _variants(),
        tolerance=2e-5,
    )
    assert point.orbit_size == 6
    assert point.uniform_exact_ordered_recovery_ceiling == pytest.approx(1.0 / 6.0)
    assert point.raw_gradient_difference.relative_l2_difference < 2e-5
    assert point.all_release_checks_pass
    assert {variant.name for variant in point.variants} == set(_variants())
    assert all(
        variant.shared_randomness_pathwise_match
        and variant.identical_release_distribution_certified
        for variant in point.variants
    )


def test_release_benchmark_requires_parameter_free_revin_for_symmetry_claim():
    manifest = ModernTimeSeriesAttackManifest(
        dataset={
            "name": "synthetic_multivariate_time",
            "max_samples": 4,
            "context": 6,
            "horizon": 2,
            "channels": 3,
        },
        victim={
            "architecture": "itransformer",
            "d_model": 6,
            "n_heads": 2,
            "e_layers": 1,
            "d_ff": 12,
            "dropout": 0.0,
            "revin": True,
            "revin_affine": True,
        },
        training={
            "epochs": 1,
            "batch_size": 2,
            "optimizer": "adam",
            "learning_rate": 1e-3,
        },
        attack={"prior": "direct", "steps": 1, "known_target": True},
        victim_seed=79,
        attack_indices=(0,),
        attack_seeds=(101,),
        minimum_publication_batches=1,
        minimum_publication_attack_seeds=1,
        publication_mode=False,
    )
    with pytest.raises(ValueError, match="revin_affine=false"):
        run_channel_permutation_release_benchmark(manifest, _variants())


def test_synthetic_release_benchmark_records_all_mechanism_classes():
    manifest = ModernTimeSeriesAttackManifest(
        dataset={
            "name": "synthetic_multivariate_time",
            "max_samples": 4,
            "context": 6,
            "horizon": 2,
            "channels": 3,
        },
        victim={
            "architecture": "itransformer",
            "d_model": 6,
            "n_heads": 2,
            "e_layers": 1,
            "d_ff": 12,
            "dropout": 0.0,
            "revin": True,
            "revin_affine": False,
        },
        training={
            "epochs": 1,
            "batch_size": 2,
            "optimizer": "adam",
            "learning_rate": 1e-3,
        },
        attack={"prior": "direct", "steps": 1, "known_target": True},
        victim_seed=83,
        attack_indices=(0, 1),
        attack_seeds=(101,),
        minimum_publication_batches=2,
        minimum_publication_attack_seeds=1,
        publication_mode=False,
    )
    report = run_channel_permutation_release_benchmark(
        manifest,
        _variants(),
        tolerance=2e-5,
    )
    assert len(report.points) == 2
    assert report.quality_gate.every_release_check_passed
    assert report.quality_gate.full_gradient_variant_present
    assert report.quality_gate.clipping_variant_present
    assert report.quality_gate.quantization_variant_present
    assert report.quality_gate.gaussian_noise_variant_present
    assert report.quality_gate.partial_visibility_variant_present
    assert not report.quality_gate.real_dataset
    assert not report.quality_gate.passed
