from __future__ import annotations

import json

from qrecon.attacks import GradientReleaseSpec
from qrecon.benchmarks import (
    ModernTimeSeriesAttackManifest,
    run_channel_permutation_release_benchmark,
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
            "d_model": 8,
            "n_heads": 2,
            "e_layers": 2,
            "d_ff": 16,
            "dropout": 0.0,
            # Preserve practical instance normalization without attaching learned
            # parameters to absolute channel positions.
            "revin": True,
            "revin_affine": False,
        },
        training={
            "epochs": 3,
            "batch_size": 8,
            "optimizer": "adamw",
            "learning_rate": 1e-3,
            "weight_decay": 1e-3,
        },
        attack={
            "prior": "direct",
            # Ordered targets are included in the private object. Supplying their
            # channel labels publicly would change the observation model.
            "known_target": False,
            "steps": 1,
            "learning_rate": 0.01,
        },
        victim_seed=47,
        attack_indices=tuple(range(20)),
        attack_seeds=(101,),
        attack_batch_size=1,
        exact_tolerance=1e-5,
        relative_l2_threshold=1e-5,
        confidence_level=0.95,
        bootstrap_samples=2000,
        bootstrap_seed=20260715,
        minimum_publication_batches=20,
        minimum_publication_attack_seeds=1,
        publication_mode=True,
    )
    variants = {
        "full_exact": GradientReleaseSpec(),
        "global_clip_0p5": GradientReleaseSpec(clip_norm=0.5),
        "fixed_8bit_quantization": GradientReleaseSpec(
            quantization_bits=8,
            quantization_scale=1e-3,
        ),
        "gaussian_noise_0p01": GradientReleaseSpec(
            noise_std=0.01,
            noise_seed=20260715,
        ),
        "first_parameter_only": GradientReleaseSpec(
            visible_parameter_indices=(0,),
        ),
        "combined_release": GradientReleaseSpec(
            clip_norm=0.5,
            noise_std=0.01,
            noise_seed=20260717,
            quantization_bits=8,
            quantization_scale=1e-3,
            visible_parameter_indices=(0,),
        ),
    }
    report = run_channel_permutation_release_benchmark(
        manifest,
        variants,
        tolerance=2e-5,
    )
    payload = report.to_dict()
    payload["private_target_contract"] = (
        "Ordered histories and their ordered future targets are private. All release "
        "mechanisms are applied after the identical full-gradient channel, so data "
        "processing preserves the same orbit for classical and quantum estimators."
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not report.quality_gate.passed:
        raise SystemExit("ETTm1 channel-permutation release-closure gate failed")


if __name__ == "__main__":
    main()
