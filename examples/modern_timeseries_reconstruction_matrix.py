from __future__ import annotations

import json

from qrecon.benchmarks import (
    ModernTimeSeriesAttackManifest,
    run_modern_timeseries_reconstruction_benchmark,
)


def _manifest(architecture: str) -> ModernTimeSeriesAttackManifest:
    multivariate = architecture == "itransformer"
    dataset: dict[str, object]
    if multivariate:
        dataset = {
            "name": "synthetic_multivariate_time",
            "max_samples": 8,
            "context": 6,
            "horizon": 2,
            "channels": 3,
        }
    else:
        dataset = {
            "name": "synthetic_time",
            "max_samples": 8,
            "context": 6,
            "horizon": 2,
        }

    victim: dict[str, object] = {
        "architecture": architecture,
        "d_model": 4,
        "n_heads": 1,
        "e_layers": 1,
        "d_ff": 8,
        "dropout": 0.0,
        "activation": "gelu",
        "revin": True,
    }
    if architecture == "patchtst":
        victim.update(
            {
                "patch_len": 2,
                "stride": 2,
                "padding_patch": True,
                "head_dropout": 0.0,
            }
        )

    return ModernTimeSeriesAttackManifest(
        dataset=dataset,
        victim=victim,
        training={
            "epochs": 1,
            "batch_size": 4,
            "optimizer": "adamw",
            "learning_rate": 1e-3,
            "weight_decay": 1e-3,
        },
        attack={
            "method": "gradient_matching",
            "prior": "direct",
            "bounded": True,
            "known_target": True,
            "steps": 3,
            "optimizer": "adam",
            "learning_rate": 0.03,
            "regularization": 1e-4,
            "match_mode": "hybrid",
            "layer_weighting": "parameter",
            "gradient_clip_norm": 10.0,
            "record_every": 1,
        },
        victim_seed=29,
        attack_indices=(0,),
        attack_seeds=(101, 103),
        attack_batch_size=1,
        exact_tolerance=0.1,
        relative_l2_threshold=0.5,
        bootstrap_samples=200,
        minimum_publication_batches=20,
        minimum_publication_attack_seeds=3,
        publication_mode=False,
    )


def main() -> None:
    reports = {
        architecture: run_modern_timeseries_reconstruction_benchmark(
            _manifest(architecture)
        ).to_dict()
        for architecture in ("transformer", "patchtst", "itransformer")
    }
    print(json.dumps(reports, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
