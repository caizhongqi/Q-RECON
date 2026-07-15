from __future__ import annotations

import json

from qrecon.benchmarks import (
    LearnedQuantileAuxiliaryConfig,
    ModernTimeSeriesAttackManifest,
    run_learned_quantile_ts_inverse_benchmark,
)


def main() -> None:
    """Run a disjoint-auxiliary learned TS-Inverse study on revision-pinned GIFT-Eval.

    The attacked records are part of the victim training split. The gradient-to-
    quantile initializer is trained only on a disjoint auxiliary split, preventing
    direct private-record reuse while preserving the public auxiliary-data threat
    model used by learned inversion attacks.
    """

    manifest = ModernTimeSeriesAttackManifest(
        dataset={
            "name": "gift_eval",
            "max_samples": 112,
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
            "d_model": 8,
            "n_heads": 2,
            "e_layers": 2,
            "d_ff": 16,
            "dropout": 0.0,
            "head_dropout": 0.0,
            "revin": True,
        },
        training={
            "epochs": 3,
            "batch_size": 8,
            "optimizer": "adamw",
            "learning_rate": 1e-3,
            "weight_decay": 1e-3,
        },
        attack={
            "bounded": True,
            "known_target": True,
            "steps": 60,
            "optimizer": "adam",
            "learning_rate": 0.03,
            "gradient_l1_weight": 1.0,
            "input_total_variation_weight": 1e-3,
            "target_total_variation_weight": 0.0,
            "trend_weight": 1e-3,
            "trend_loss": "l1",
            "trend_detach": True,
            "periodicity_weight": 1e-3,
            "periodicity_period": 4,
            "periodicity_loss": "l1",
            "low_resolution_weight": 1e-3,
            "low_resolution_factor": 2,
            "low_resolution_loss": "l1",
            "quantile_bound_weight": 1e-2,
            "gradient_clip_norm": 10.0,
            "record_every": 10,
        },
        victim_seed=43,
        attack_indices=tuple(range(20)),
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
        victim_training_indices=tuple(range(32)),
        auxiliary_indices=tuple(range(32, 112)),
        hidden_sizes=(128, 64),
        dropout=0.05,
        quantiles=(0.1, 0.5, 0.9),
        validation_fraction=0.2,
        epochs=50,
        batch_size=16,
        learning_rate=1e-3,
        weight_decay=1e-4,
        crossing_weight=1e-2,
        training_seed=20260715,
        jitter_standard_deviation=0.01,
        minimum_publication_auxiliary_samples=64,
    )
    report = run_learned_quantile_ts_inverse_benchmark(manifest, auxiliary)
    payload = report.to_dict()
    payload["publication_claim"] = {
        "passed": report.quality_gate.passed,
        "scope": "known-target learned gradient-to-quantile initialization plus TS-Inverse refinement",
        "private_auxiliary_overlap": False,
        "claim_boundary": (
            "This is a real-data, 20-window, three-restart learned-initializer study. "
            "It does not reproduce unknown-target recovery and does not establish "
            "coherent quantum access or quantum advantage."
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not report.quality_gate.passed:
        raise SystemExit("learned TS-Inverse publication quality gate failed")


if __name__ == "__main__":
    main()
