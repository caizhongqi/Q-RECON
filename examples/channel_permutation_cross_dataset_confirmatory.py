from __future__ import annotations

import json

from qrecon.attacks import GradientReleaseSpec
from qrecon.benchmarks import (
    ModernTimeSeriesAttackManifest,
    run_channel_permutation_fibre_benchmark,
    run_channel_permutation_release_benchmark,
)

DATASETS = (
    (
        "ettm1",
        "data/ETT-small/ETTm1.csv",
        "6ce1759b1a18e3328421d5d75fadcb316c449fcd7cec32820c8dafda71986c9e",
    ),
    (
        "ettm2",
        "data/ETT-small/ETTm2.csv",
        "db973ca252c6410a30d0469b13d696cf919648d0f3fd588c60f03fdbdbadd1fd",
    ),
    (
        "etth1",
        "data/ETT-small/ETTh1.csv",
        "f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066",
    ),
)


def _manifest(path: str, sha256: str, seed: int) -> ModernTimeSeriesAttackManifest:
    return ModernTimeSeriesAttackManifest(
        dataset={
            "name": "multivariate_csv",
            "path": path,
            "expected_file_sha256": sha256,
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
            "known_target": False,
            "steps": 1,
            "learning_rate": 0.01,
        },
        victim_seed=seed,
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


def main() -> None:
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
    reports: dict[str, object] = {}
    global_pass = True
    for offset, (name, path, sha256) in enumerate(DATASETS):
        manifest = _manifest(path, sha256, 47 + 2 * offset)
        fibre = run_channel_permutation_fibre_benchmark(manifest, tolerance=2e-5)
        releases = run_channel_permutation_release_benchmark(
            manifest,
            variants,
            tolerance=2e-5,
        )
        identity = {
            "dataset_sha256_match": fibre.dataset_sha256 == releases.dataset_sha256,
            "model_sha256_match": fibre.model_sha256 == releases.model_sha256,
            "victim_class_match": fibre.victim_class == releases.victim_class,
        }
        passed = (
            fibre.quality_gate.passed
            and releases.quality_gate.passed
            and all(identity.values())
        )
        global_pass = global_pass and passed
        reports[name] = {
            "fibre": fibre.to_dict(),
            "releases": releases.to_dict(),
            "cross_report_identity": identity,
            "passed": passed,
        }

    payload = {
        "schema_version": "qrecon.channel-permutation-cross-dataset.v1",
        "datasets": reports,
        "quality_gate": {
            "three_immutable_real_datasets": True,
            "twenty_windows_per_dataset": True,
            "full_gradient_generator_complete_fibres": all(
                report["fibre"]["quality_gate"]["passed"]
                for report in reports.values()
            ),
            "release_closure_across_all_mechanisms": all(
                report["releases"]["quality_gate"]["passed"]
                for report in reports.values()
            ),
            "passed": global_pass,
        },
        "theorem": (
            "Anonymous-channel equivariant forecasting losses induce a full-gradient "
            "channel-permutation fibre. Deterministic gradient postprocessing and "
            "data-independent randomized release kernels preserve that fibre; the "
            "uniform exact labeled-order recovery ceiling is the inverse orbit size."
        ),
        "claim_boundary": (
            "The benchmark treats channel identities and ordered targets as private. "
            "Public semantic channel labels, channel-indexed parameters, or recovery "
            "defined only modulo channel permutation are different threat models."
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not global_pass:
        raise SystemExit("cross-dataset channel-permutation quality gate failed")


if __name__ == "__main__":
    main()
