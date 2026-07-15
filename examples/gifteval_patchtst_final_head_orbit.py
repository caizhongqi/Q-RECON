from __future__ import annotations

import json

from qrecon.benchmarks import (
    ModernTimeSeriesAttackManifest,
    run_final_head_orbit_benchmark,
)


def main() -> None:
    manifest = ModernTimeSeriesAttackManifest(
        dataset={
            "name": "gift_eval",
            "max_samples": 32,
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
            "epochs": 2,
            "batch_size": 8,
            "optimizer": "adamw",
            "learning_rate": 1e-3,
            "weight_decay": 1e-3,
        },
        attack={"prior": "direct", "known_target": True, "steps": 1},
        victim_seed=43,
        attack_indices=tuple(range(20)),
        attack_seeds=(101, 103, 107),
        attack_batch_size=1,
        confidence_level=0.95,
        bootstrap_samples=2000,
        bootstrap_seed=1729,
        minimum_publication_batches=20,
        minimum_publication_attack_seeds=3,
        publication_mode=True,
    )
    report = run_final_head_orbit_benchmark(manifest)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
