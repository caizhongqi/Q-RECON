from __future__ import annotations

import json

from qrecon.benchmarks import (
    ModernTimeSeriesAttackManifest,
    run_ts_inverse_style_benchmark,
)


def main() -> None:
    manifest = ModernTimeSeriesAttackManifest(
        dataset={
            "name": "gift_eval",
            "max_samples": 24,
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
            "batch_size": 6,
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
            "quantile_bound_weight": 0.0,
            "gradient_clip_norm": 10.0,
            "record_every": 20,
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
    report = run_ts_inverse_style_benchmark(manifest)
    gate = report.quality_gate
    passed_objective_publication_gate = (
        gate.architecture_is_modern
        and gate.real_dataset
        and gate.enough_attack_batches
        and gate.enough_attack_seeds
        and gate.every_batch_has_successful_attempt
        and gate.no_failed_attempts
        and gate.official_objective_components_present
        and gate.publication_mode
    )
    payload = report.to_dict()
    payload["objective_publication_gate"] = {
        "passed": passed_objective_publication_gate,
        "requires_full_learned_ts_inverse_reproduction": False,
        "learned_quantile_model_present": gate.learned_quantile_model_present,
        "claim_boundary": (
            "Passing this gate validates a 20-batch, three-restart reproduction of "
            "the public TS-Inverse optimization objective family. It does not claim "
            "the learned gradient-to-quantile initializer has been reproduced."
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not passed_objective_publication_gate:
        raise SystemExit("TS-Inverse objective publication gate failed")


if __name__ == "__main__":
    main()
