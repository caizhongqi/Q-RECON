from __future__ import annotations

from qrecon.benchmarks.final_head_orbit import run_final_head_orbit_benchmark
from qrecon.benchmarks.modern_timeseries_reconstruction import (
    ModernTimeSeriesAttackManifest,
)


def _manifest(architecture: str, channels: int) -> ModernTimeSeriesAttackManifest:
    dataset = (
        {
            "name": "synthetic_time",
            "max_samples": 4,
            "context": 4,
            "horizon": 2,
        }
        if channels == 1
        else {
            "name": "synthetic_multivariate_time",
            "max_samples": 4,
            "context": 4,
            "horizon": 2,
            "channels": channels,
        }
    )
    victim = {
        "architecture": architecture,
        "d_model": 2,
        "n_heads": 1,
        "e_layers": 1,
        "d_ff": 4,
        "dropout": 0.0,
        "revin": True,
    }
    if architecture == "patchtst":
        victim.update({"patch_len": 2, "stride": 1, "padding_patch": True})
    return ModernTimeSeriesAttackManifest(
        dataset=dataset,
        victim=victim,
        training={
            "epochs": 1,
            "batch_size": 2,
            "optimizer": "adam",
            "learning_rate": 1e-3,
        },
        attack={"prior": "direct", "known_target": True, "steps": 1},
        victim_seed=19,
        attack_indices=(0,),
        attack_seeds=(101,),
        attack_batch_size=1,
        bootstrap_samples=20,
        publication_mode=False,
    )


def test_univariate_patchtst_head_has_no_sample_index_orbit():
    report = run_final_head_orbit_benchmark(_manifest("patchtst", 1))
    point = report.points[0]
    assert point.effective_samples == 1
    assert point.orthogonal_complement_dimension == 0
    assert not point.has_nontrivial_collision
    assert point.continuous_orbit_dimension == 0


def test_multivariate_itransformer_head_has_exact_gradient_collision():
    report = run_final_head_orbit_benchmark(_manifest("itransformer", 4))
    point = report.points[0]
    assert point.effective_samples == 4
    assert point.output_dimension == 2
    assert point.orthogonal_complement_dimension >= 1
    assert point.has_nontrivial_collision
    assert point.has_continuous_family
    assert point.continuous_orbit_dimension > 0
    assert point.collision_input_displacement is not None
    assert point.collision_input_displacement > 0.0
    assert point.collision_statistic_error is not None
    assert point.collision_statistic_error < 1e-8
    assert point.collision_actual_weight_gradient_error is not None
    assert point.collision_actual_weight_gradient_error < 1e-8
    assert point.collision_actual_bias_gradient_error is not None
    assert point.collision_actual_bias_gradient_error < 1e-8
