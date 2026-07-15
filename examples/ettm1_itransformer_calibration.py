from __future__ import annotations

import json

from qrecon.benchmarks import (
    ModernTimeSeriesAttackManifest,
    run_modern_timeseries_reconstruction_benchmark,
)


def main() -> None:
    manifest = ModernTimeSeriesAttackManifest(
        dataset={
            "name": "multivariate_csv",
            "path": "data/ETT-small/ETTm1.csv",
            "max_samples": 6,
            "context": 16,
            "horizon": 4,
            "stride": 4,
            "start": 0,
            "columns": ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"],
        },
        victim={
            "architecture": "itransformer",
            "d_model": 4,
            "n_heads": 1,
            "e_layers": 1,
            "d_ff": 8,
            "dropout": 0.0,
            "activation": "gelu",
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
            "steps": 5,
            "optimizer": "adam",
            "learning_rate": 0.02,
            "regularization": 1e-4,
            "match_mode": "hybrid",
            "layer_weighting": "parameter",
            "gradient_clip_norm": 10.0,
            "record_every": 1,
        },
        victim_seed=47,
        attack_indices=(0, 1),
        attack_seeds=(101,),
        attack_batch_size=1,
        exact_tolerance=0.1,
        relative_l2_threshold=0.5,
        bootstrap_samples=200,
        publication_mode=False,
    )
    report = run_modern_timeseries_reconstruction_benchmark(manifest)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
