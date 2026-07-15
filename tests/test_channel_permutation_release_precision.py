from __future__ import annotations

import pytest

from qrecon.attacks import GradientReleaseSpec
from qrecon.benchmarks.channel_permutation_release_precision import (
    run_channel_permutation_release_benchmark_with_dtype,
)
from qrecon.benchmarks.modern_timeseries_reconstruction import (
    ModernTimeSeriesAttackManifest,
)


def _manifest() -> ModernTimeSeriesAttackManifest:
    return ModernTimeSeriesAttackManifest(
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
        attack={"prior": "direct", "steps": 1, "known_target": False},
        victim_seed=97,
        attack_indices=(0, 1),
        attack_seeds=(101,),
        minimum_publication_batches=2,
        minimum_publication_attack_seeds=1,
        publication_mode=False,
    )


def _variants() -> dict[str, GradientReleaseSpec]:
    return {
        "full": GradientReleaseSpec(),
        "clipped": GradientReleaseSpec(clip_norm=0.5),
        "quantized": GradientReleaseSpec(
            quantization_bits=8,
            quantization_scale=1e-3,
        ),
        "gaussian": GradientReleaseSpec(noise_std=0.01, noise_seed=20260715),
        "partial": GradientReleaseSpec(visible_parameter_indices=(0,)),
    }


def test_float64_release_audit_records_precision_and_preserves_fibre():
    report = run_channel_permutation_release_benchmark_with_dtype(
        _manifest(),
        _variants(),
        evaluation_dtype="float64",
        tolerance=1e-10,
    )
    assert report.summary["evaluation_dtype"] == "float64"
    assert report.summary["numerical_tolerance"] == pytest.approx(1e-10)
    assert report.environment["release_evaluation_dtype"] == "float64"
    assert report.quality_gate.every_release_check_passed
    assert all(point.all_release_checks_pass for point in report.points)
    assert all(
        variant.identical_release_distribution_certified
        for point in report.points
        for variant in point.variants
    )


def test_release_audit_rejects_undeclared_precision():
    with pytest.raises(ValueError, match="evaluation_dtype"):
        run_channel_permutation_release_benchmark_with_dtype(
            _manifest(),
            _variants(),
            evaluation_dtype="bfloat16",
        )
