from __future__ import annotations

from qrecon.benchmarks import (
    LearnedQuantileAuxiliaryConfig,
    ModernTimeSeriesAttackManifest,
    run_paired_learned_ts_inverse_benchmark,
)


def test_paired_learned_initializer_uses_same_victim_and_seeds():
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
            "prior": "direct",
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
    report = run_paired_learned_ts_inverse_benchmark(manifest, auxiliary)
    assert report.victim_class == "PatchTST"
    assert len(report.attempts) == 2
    assert {attempt.method for attempt in report.attempts} == {
        "random_l1",
        "learned_quantile_l1",
    }
    assert all(attempt.status == "success" for attempt in report.attempts)
    assert report.paired_summary["fully_paired_batches"] == 1
    assert report.quality_gate.every_method_has_successful_attempt_per_batch
    assert report.quality_gate.split_disjoint
