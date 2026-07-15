from __future__ import annotations

import json

from qrecon.benchmarks import (
    LearnedQuantileAuxiliaryConfig,
    ModernTimeSeriesAttackManifest,
    run_paired_learned_ts_inverse_benchmark,
)


def main() -> None:
    manifest = ModernTimeSeriesAttackManifest(
        dataset={
            "name": "gift_eval",
            "max_samples": 128,
            "context": 16,
            "horizon": 4,
            "streaming": True,
            "split": "train",
            "revision": "30841734ac5cfddbd0c3bad6d09d2b6b32becbb0",
        },
        victim={
            "architecture": "patchtst",
            "patch_len": 4,
            "stride": 2,
            "padding_patch": True,
            "d_model": 4,
            "n_heads": 1,
            "e_layers": 1,
            "d_ff": 8,
            "dropout": 0.0,
            "head_dropout": 0.0,
            "revin": True,
        },
        training={
            "epochs": 1,
            "batch_size": 8,
            "optimizer": "adamw",
            "learning_rate": 1e-3,
            "weight_decay": 1e-3,
        },
        attack={
            "prior": "direct",
            "bounded": True,
            "known_target": True,
            "steps": 100,
            "optimizer": "adam",
            "learning_rate": 0.03,
            "gradient_l1_weight": 1.0,
            "input_total_variation_weight": 0.0,
            "target_total_variation_weight": 0.0,
            "trend_weight": 0.0,
            "periodicity_weight": 0.0,
            "low_resolution_weight": 0.0,
            "quantile_bound_weight": 0.01,
            "gradient_clip_norm": 10.0,
            "record_every": 20,
        },
        victim_seed=43,
        # Samples 0 and 1 were used only for exploratory calibration. The
        # confirmatory set is the disjoint victim-training subset 2..21.
        attack_indices=tuple(range(2, 22)),
        attack_seeds=(101, 103, 107),
        attack_batch_size=1,
        exact_tolerance=0.1,
        relative_l2_threshold=0.5,
        confidence_level=0.95,
        bootstrap_samples=2000,
        bootstrap_seed=20260715,
        minimum_publication_batches=20,
        minimum_publication_attack_seeds=3,
        publication_mode=True,
    )
    auxiliary = LearnedQuantileAuxiliaryConfig(
        # The victim is trained on 0..31. Auxiliary gradients use 32..127 and
        # therefore contain no victim-training record.
        victim_training_indices=tuple(range(32)),
        auxiliary_indices=tuple(range(32, 128)),
        hidden_sizes=(128, 64),
        dropout=0.05,
        quantiles=(0.1, 0.5, 0.9),
        validation_fraction=0.2,
        epochs=75,
        batch_size=16,
        learning_rate=1e-3,
        weight_decay=1e-4,
        crossing_weight=1e-2,
        training_seed=20260715,
        jitter_standard_deviation=0.02,
        minimum_publication_auxiliary_samples=64,
    )
    report = run_paired_learned_ts_inverse_benchmark(manifest, auxiliary)
    payload = report.to_dict()
    payload["confirmatory_split"] = {
        "exploratory_attack_indices": [0, 1],
        "confirmatory_attack_indices": list(range(2, 22)),
        "victim_training_indices": list(range(32)),
        "auxiliary_indices": list(range(32, 128)),
        "all_sets_declared_before_confirmatory_execution": True,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not report.quality_gate.passed:
        raise SystemExit("paired learned-initializer publication gate failed")


if __name__ == "__main__":
    main()
