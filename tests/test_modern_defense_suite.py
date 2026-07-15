from __future__ import annotations

import pytest

from qrecon.benchmarks import (
    ModernGradientDefenseVariant,
    ModernTimeSeriesAttackManifest,
    run_modern_timeseries_defense_suite,
    standard_modern_gradient_defenses,
)


def _manifest() -> ModernTimeSeriesAttackManifest:
    return ModernTimeSeriesAttackManifest(
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
            "learning_rate": 0.02,
            "regularization": 0.0,
            "match_mode": "hybrid",
            "layer_weighting": "parameter",
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


def test_standard_defense_matrix_contains_required_conditions():
    variants = standard_modern_gradient_defenses()
    assert {variant.name for variant in variants} == {
        "full_exact",
        "global_clip_1",
        "symmetric_int8",
        "gaussian_noise_1e-3",
        "last_head_only",
    }
    head = next(variant for variant in variants if variant.name == "last_head_only")
    assert head.visible_scope == "last_head"


def test_tiny_defense_suite_runs_all_release_contracts():
    report = run_modern_timeseries_defense_suite(
        _manifest(),
        standard_modern_gradient_defenses(),
    )
    assert report.victim_class == "PatchTST"
    assert len(report.attempts) == 5
    assert all(attempt.status == "success" for attempt in report.attempts)
    assert set(report.variant_summaries) == {
        "full_exact",
        "global_clip_1",
        "symmetric_int8",
        "gaussian_noise_1e-3",
        "last_head_only",
    }
    quantized = next(
        attempt for attempt in report.attempts if attempt.variant == "symmetric_int8"
    )
    assert quantized.release_metadata["quantization_bits"] == 8
    assert quantized.release_metadata["quantization_scale"] is not None
    noisy = next(
        attempt
        for attempt in report.attempts
        if attempt.variant == "gaussian_noise_1e-3"
    )
    assert noisy.release_metadata["noise_std"] == pytest.approx(1e-3)
    partial = next(
        attempt for attempt in report.attempts if attempt.variant == "last_head_only"
    )
    assert partial.release_metadata["visible_parameter_tensors"] == 2
    assert not report.quality_gate.passed


def test_defense_suite_rejects_duplicate_condition_names():
    with pytest.raises(ValueError, match="defense variant names must be unique"):
        run_modern_timeseries_defense_suite(
            _manifest(),
            (
                ModernGradientDefenseVariant("same", {}),
                ModernGradientDefenseVariant("same", {"clip_norm": 1.0}),
            ),
        )


def test_defense_variant_rejects_unknown_visibility_scope():
    with pytest.raises(ValueError, match="visible_scope"):
        ModernGradientDefenseVariant("bad", {}, visible_scope="first_layer")
