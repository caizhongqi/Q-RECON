from __future__ import annotations

import hashlib
import json
from pathlib import Path

from qrecon.attacks import GradientReleaseSpec
from qrecon.benchmarks import (
    ModernTimeSeriesAttackManifest,
    benchmark_environment_manifest,
    run_channel_permutation_fibre_benchmark,
    run_channel_side_information_benchmark,
)
from qrecon.benchmarks.channel_permutation_release_precision import (
    run_channel_permutation_release_benchmark_with_dtype,
)

DATASET_PATH = "data/UCI-appliances/energydata_complete.csv"
DATASET_SHA256 = "2820bf712ad0275cb18b85a05250926100d8e65ebb9f4d2d016ca91ea152a25d"
SOURCE_ARCHIVE_SHA256 = (
    "2fccf354445d886e7917620b0195db1f3e3e34d5a067a93b844694a4c561255a"
)
DATASET_DOI = "10.24432/C5VC8G"
DATASET_LICENSE = "CC BY 4.0"
CHANNELS = ("T1", "T2", "T3", "T4", "T5", "T6", "T7")

ARCHITECTURES = (
    {
        "name": "itransformer",
        "config": {
            "architecture": "itransformer",
            "d_model": 8,
            "n_heads": 2,
            "e_layers": 2,
            "d_ff": 16,
            "dropout": 0.0,
            "revin": True,
            "revin_affine": False,
        },
    },
    {
        "name": "patchtst_shared_head",
        "config": {
            "architecture": "patchtst",
            "patch_len": 4,
            "stride": 2,
            "padding_patch": True,
            "d_model": 8,
            "n_heads": 2,
            "e_layers": 2,
            "d_ff": 16,
            "dropout": 0.0,
            "head_dropout": 0.0,
            "revin": True,
            "revin_affine": False,
            "individual_head": False,
        },
    },
)

RELEASE_VARIANTS = {
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
}


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest(victim: dict[str, object], seed: int) -> ModernTimeSeriesAttackManifest:
    return ModernTimeSeriesAttackManifest(
        dataset={
            "name": "multivariate_csv",
            "path": DATASET_PATH,
            "expected_file_sha256": DATASET_SHA256,
            "max_samples": 64,
            "context": 16,
            "horizon": 4,
            "stride": 8,
            "columns": list(CHANNELS),
        },
        victim=dict(victim),
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
        attack_indices=tuple(range(40, 60)),
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
    observed_sha256 = _file_sha256(DATASET_PATH)
    if observed_sha256 != DATASET_SHA256:
        raise RuntimeError(
            f"UCI CSV SHA256 mismatch: expected {DATASET_SHA256}, "
            f"observed {observed_sha256}"
        )

    architecture_reports: dict[str, object] = {}
    architecture_pass = True
    for index, architecture in enumerate(ARCHITECTURES):
        manifest = _manifest(architecture["config"], seed=211 + index)
        fibre = run_channel_permutation_fibre_benchmark(manifest, tolerance=2e-5)
        release = run_channel_permutation_release_benchmark_with_dtype(
            manifest,
            RELEASE_VARIANTS,
            evaluation_dtype="float64",
            tolerance=1e-10,
        )
        passed = fibre.quality_gate.passed and release.quality_gate.passed
        architecture_pass = architecture_pass and passed
        architecture_reports[str(architecture["name"])] = {
            "fibre": fibre.to_dict(),
            "release_closure": release.to_dict(),
            "quality_gate_passed": passed,
        }

    side_manifest = _manifest(ARCHITECTURES[0]["config"], seed=223)
    side_information = run_channel_side_information_benchmark(
        side_manifest,
        calibration_indices=tuple(range(20)),
        evaluation_indices=tuple(range(20, 40)),
        permutation_seeds=(101, 103, 107, 109, 113),
        spectral_bins=4,
        variance_floor=1e-4,
        minimum_calibration_windows=20,
        minimum_permutation_trials=100,
    )

    passed = architecture_pass and side_information.quality_gate.passed
    payload = {
        "schema_version": "qrecon.uci-appliances-channel-permutation.v1",
        "source": {
            "dataset": "UCI Appliances Energy Prediction",
            "doi": DATASET_DOI,
            "license": DATASET_LICENSE,
            "archive_sha256": SOURCE_ARCHIVE_SHA256,
            "csv_sha256": observed_sha256,
            "selected_channels": list(CHANNELS),
        },
        "experimental_contract": {
            "context": 16,
            "horizon": 4,
            "stride": 8,
            "fibre_release_windows": list(range(40, 60)),
            "public_calibration_windows": list(range(20)),
            "private_side_information_windows": list(range(20, 40)),
            "side_information_permutations_per_window": 5,
            "release_evaluation_dtype": "float64",
            "release_tolerance": 1e-10,
        },
        "architectures": architecture_reports,
        "side_information": side_information.to_dict(),
        "quality_gate": {
            "independent_non_ett_real_dataset": True,
            "immutable_source_hashes": True,
            "two_modern_architectures": len(architecture_reports) == 2,
            "forty_primary_fibre_release_points": True,
            "one_hundred_side_information_trials": True,
            "all_architecture_gates_passed": architecture_pass,
            "side_information_gate_passed": side_information.quality_gate.passed,
            "passed": passed,
        },
        "environment": benchmark_environment_manifest(),
        "conclusion": (
            "The anonymous-channel gradient fibre and release closure are evaluated on "
            "an independent UCI building-sensor dataset for iTransformer and shared-head "
            "PatchTST. Public labeled calibration is measured separately and may shrink "
            "the residual anonymity in practice."
        ),
        "claim_boundary": (
            "The theorem applies when semantic channel identities are private and the "
            "victim is channel-permutation equivariant. The side-information matcher is "
            "declared rather than Bayes-optimal. This experiment does not establish a "
            "positive quantum speedup or privacy when full channel labels are public."
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit("UCI appliances publication quality gate failed")


if __name__ == "__main__":
    main()
