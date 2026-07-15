from __future__ import annotations

import json

from qrecon.benchmarks import (
    ModernTimeSeriesAttackManifest,
    run_head_representation_reconstruction_benchmark,
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
            "head_steps": 500,
            "head_learning_rate": 0.03,
            "head_regularization": 1e-4,
            "gradient_clip_norm": 10.0,
            "record_every": 25,
        },
        victim_seed=43,
        attack_indices=(0, 1),
        attack_seeds=(101, 103),
        attack_batch_size=1,
        exact_tolerance=0.1,
        relative_l2_threshold=0.5,
        bootstrap_samples=500,
        publication_mode=False,
    )
    report = run_head_representation_reconstruction_benchmark(manifest)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
