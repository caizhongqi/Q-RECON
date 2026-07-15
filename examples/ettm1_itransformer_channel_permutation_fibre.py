from __future__ import annotations

import json

from qrecon.benchmarks import (
    ModernTimeSeriesAttackManifest,
    run_channel_permutation_fibre_benchmark,
)

ETTM1_SHA256 = "6ce1759b1a18e3328421d5d75fadcb316c449fcd7cec32820c8dafda71986c9e"


def main() -> None:
    manifest = ModernTimeSeriesAttackManifest(
        dataset={
            "name": "multivariate_csv",
            "path": "data/ETT-small/ETTm1.csv",
            "expected_file_sha256": ETTM1_SHA256,
            "max_samples": 32,
            "context": 16,
            "horizon": 4,
            "stride": 4,
            "columns": ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"],
        },
        victim={
            "architecture": "itransformer",
            "d_model": 4,
            "n_heads": 1,
            "e_layers": 1,
            "d_ff": 8,
            "dropout": 0.0,
            # Per-channel affine RevIN parameters attach identities to channel
            # positions. They are disabled for the permutation-equivariant theorem.
            "revin": False,
        },
        training={
            "epochs": 2,
            "batch_size": 8,
            "optimizer": "adamw",
            "learning_rate": 1e-3,
            "weight_decay": 1e-3,
        },
        attack={"prior": "direct", "known_target": False, "steps": 1},
        victim_seed=47,
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
    report = run_channel_permutation_fibre_benchmark(manifest, tolerance=1e-5)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
