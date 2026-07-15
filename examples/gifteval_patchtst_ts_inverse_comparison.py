from __future__ import annotations

import json

from qrecon.benchmarks import (
    ModernTimeSeriesAttackManifest,
    run_ts_inverse_style_benchmark,
)


def _manifest(*, temporal: bool) -> ModernTimeSeriesAttackManifest:
    return ModernTimeSeriesAttackManifest(
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
            "steps": 100,
            "optimizer": "adam",
            "learning_rate": 0.03,
            "gradient_l1_weight": 1.0,
            "input_total_variation_weight": 0.0,
            "target_total_variation_weight": 0.0,
            "trend_weight": 1e-3 if temporal else 0.0,
            "trend_loss": "l1",
            "trend_detach": True,
            "periodicity_weight": 1e-3 if temporal else 0.0,
            "periodicity_period": 4,
            "periodicity_loss": "l1",
            "low_resolution_weight": 1e-3 if temporal else 0.0,
            "low_resolution_factor": 2,
            "low_resolution_loss": "l1",
            "quantile_bound_weight": 0.0,
            "gradient_clip_norm": 10.0,
            "record_every": 10,
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


def main() -> None:
    l1_only = run_ts_inverse_style_benchmark(_manifest(temporal=False))
    temporal = run_ts_inverse_style_benchmark(_manifest(temporal=True))
    print(
        json.dumps(
            {
                "schema_version": "qrecon.ts-inverse-style-comparison.v1",
                "l1_only": l1_only.to_dict(),
                "l1_temporal": temporal.to_dict(),
                "claim_boundary": (
                    "Both runs reproduce the public optimization objective family. "
                    "Neither includes the learned TS-Inverse gradient-to-quantile model."
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
