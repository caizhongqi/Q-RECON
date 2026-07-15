from __future__ import annotations

import json

from qrecon.benchmarks import (
    ModernTimeSeriesAttackManifest,
    run_head_release_stability_audit,
    standard_modern_gradient_defenses,
)


def main() -> None:
    manifest = ModernTimeSeriesAttackManifest(
        dataset={
            "name": "gift_eval",
            "max_samples": 6,
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
            "batch_size": 3,
            "optimizer": "adamw",
            "learning_rate": 1e-3,
            "weight_decay": 1e-3,
        },
        attack={
            "prior": "direct",
            "bounded": True,
            "known_target": True,
            "steps": 1,
        },
        victim_seed=43,
        attack_indices=(0, 1),
        attack_seeds=(101,),
        attack_batch_size=1,
        exact_tolerance=0.1,
        relative_l2_threshold=0.5,
        bootstrap_samples=200,
        publication_mode=False,
    )
    report = run_head_release_stability_audit(
        manifest,
        standard_modern_gradient_defenses(),
        failure_probability=0.01,
    )
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
