from qrecon.benchmarks.head_release_stability import (
    run_head_release_stability_audit,
)
from qrecon.benchmarks.modern_timeseries_defense_suite import (
    standard_modern_gradient_defenses,
)
from qrecon.benchmarks.modern_timeseries_reconstruction import (
    ModernTimeSeriesAttackManifest,
)


def _manifest() -> ModernTimeSeriesAttackManifest:
    return ModernTimeSeriesAttackManifest(
        dataset={
            "name": "synthetic_time",
            "max_samples": 4,
            "context": 4,
            "horizon": 2,
        },
        victim={
            "architecture": "patchtst",
            "d_model": 2,
            "n_heads": 1,
            "e_layers": 1,
            "d_ff": 4,
            "dropout": 0.0,
            "patch_len": 2,
            "stride": 1,
            "padding_patch": True,
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
        },
        victim_seed=17,
        attack_indices=(0,),
        attack_seeds=(101,),
        attack_batch_size=1,
        bootstrap_samples=20,
        publication_mode=False,
    )


def test_standard_defense_audit_certifies_single_effective_sample():
    report = run_head_release_stability_audit(
        _manifest(),
        standard_modern_gradient_defenses(),
        failure_probability=1e-6,
    )
    assert len(report.points) == 5
    by_name = {point.variant: point for point in report.points}
    assert set(by_name) == {
        "full_exact",
        "global_clip_1",
        "symmetric_int8",
        "gaussian_noise_1e-3",
        "last_head_only",
    }
    assert all(point.head_visible for point in report.points)
    assert all(point.effective_samples == 1 for point in report.points)
    assert by_name["full_exact"].actual_representation_l2_error is not None
    assert by_name["full_exact"].actual_representation_l2_error < 1e-6
    assert by_name["global_clip_1"].common_scale_invariance_l2_error is not None
    assert by_name["global_clip_1"].common_scale_invariance_l2_error < 1e-6
    assert not by_name["symmetric_int8"].quantization_saturated
    certifiable = [point for point in report.points if point.certifiable]
    assert certifiable
    assert all(point.certificate_sound is True for point in certifiable)
    assert report.summary["certificate_sound"]["rate"] == 1.0
