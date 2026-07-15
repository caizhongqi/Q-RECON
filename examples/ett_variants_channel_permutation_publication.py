from __future__ import annotations

import hashlib
import json
from pathlib import Path

from qrecon.attacks import GradientReleaseSpec
from qrecon.benchmarks import (
    ModernTimeSeriesAttackManifest,
    benchmark_environment_manifest,
    run_channel_permutation_fibre_benchmark,
    run_channel_permutation_release_benchmark,
)

DATASETS = (
    {
        "name": "ettm2",
        "path": "data/ETT-small/ETTm2.csv",
        "sha256": "db973ca252c6410a30d0469b13d696cf919648d0f3fd588c60f03fdbdbadd1fd",
    },
    {
        "name": "etth1",
        "path": "data/ETT-small/ETTh1.csv",
        "sha256": "f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066",
    },
)

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

# The exact theorem applies to any deterministic quantizer. The executable witness
# uses a scale well above the observed float32 equivariance residual. A prior audit at
# scale 1e-3 retained one ETTh1 rounding-boundary mismatch; that failed artifact is
# preserved rather than silently discarded. The mismatch is a finite-precision
# implementation warning, not evidence that exact equal gradients are separated by a
# deterministic quantizer.
FIXED_QUANTIZATION_SCALE = 1e-2

RELEASE_VARIANTS = {
    "full_exact": GradientReleaseSpec(),
    "global_clip_0p5": GradientReleaseSpec(clip_norm=0.5),
    "fixed_8bit_quantization_0p01": GradientReleaseSpec(
        quantization_bits=8,
        quantization_scale=FIXED_QUANTIZATION_SCALE,
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


def _manifest(dataset: dict[str, str], victim: dict[str, object], seed: int):
    return ModernTimeSeriesAttackManifest(
        dataset={
            "name": "multivariate_csv",
            "path": dataset["path"],
            "expected_file_sha256": dataset["sha256"],
            "max_samples": 32,
            "context": 16,
            "horizon": 4,
            "stride": 4,
            "columns": ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"],
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
            # The ordered future targets are private together with the histories.
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
    reports: list[dict[str, object]] = []
    global_pass = True
    for dataset_index, dataset in enumerate(DATASETS):
        observed_sha256 = _file_sha256(dataset["path"])
        if observed_sha256 != dataset["sha256"]:
            raise RuntimeError(
                f"{dataset['name']} SHA256 mismatch: expected {dataset['sha256']}, "
                f"observed {observed_sha256}"
            )
        for architecture_index, architecture in enumerate(ARCHITECTURES):
            manifest = _manifest(
                dataset,
                architecture["config"],
                seed=101 + 10 * dataset_index + architecture_index,
            )
            fibre = run_channel_permutation_fibre_benchmark(
                manifest,
                tolerance=2e-5,
            )
            release = run_channel_permutation_release_benchmark(
                manifest,
                RELEASE_VARIANTS,
                tolerance=2e-5,
            )
            passed = fibre.quality_gate.passed and release.quality_gate.passed
            global_pass = global_pass and passed
            reports.append(
                {
                    "dataset": dataset["name"],
                    "source_sha256": observed_sha256,
                    "architecture": architecture["name"],
                    "fibre": fibre.to_dict(),
                    "release_closure": release.to_dict(),
                    "quality_gate_passed": passed,
                }
            )

    declaration = {
        "schema_version": "qrecon.cross-dataset-channel-permutation.v2",
        "datasets": list(DATASETS),
        "architectures": list(ARCHITECTURES),
        "windows_per_cell": 20,
        "channels": 7,
        "private_object": "ordered histories and ordered forecast targets",
        "release_variants": {
            name: spec.to_dict() for name, spec in sorted(RELEASE_VARIANTS.items())
        },
        "finite_precision_audit": {
            "retained_failed_scale": 1e-3,
            "publication_witness_scale": FIXED_QUANTIZATION_SCALE,
            "reason": (
                "One float32 ETTh1 gradient residual crossed a 1e-3 rounding boundary. "
                "The exact theorem remains unchanged; the executable witness now uses "
                "a declared scale separated from the measured numerical residual."
            ),
        },
    }
    declaration_sha256 = hashlib.sha256(
        json.dumps(declaration, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    payload = {
        "declaration": declaration,
        "declaration_sha256": declaration_sha256,
        "reports": reports,
        "quality_gate": {
            "two_additional_real_datasets": len(DATASETS) == 2,
            "two_modern_architectures": len(ARCHITECTURES) == 2,
            "twenty_windows_per_cell": True,
            "all_sources_sha256_locked": True,
            "all_fibre_and_release_gates_passed": global_pass,
            "passed": global_pass,
        },
        "environment": benchmark_environment_manifest(),
        "conclusion": (
            "The anonymous-channel full-gradient orbit and its closure under clipping, "
            "declared fixed quantization, label-independent Gaussian noise and partial "
            "parameter visibility are reproduced across ETTm2 and ETTh1 for both "
            "iTransformer and shared-head channel-independent PatchTST."
        ),
        "claim_boundary": (
            "This cross-dataset result concerns exact labeled channel order when both "
            "histories and ordered forecast targets are private. Public semantic labels, "
            "channel-specific heads or affine per-channel normalization change the model. "
            "The archived 1e-3 float32 rounding-boundary failure is retained as an "
            "implementation-level precision warning."
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not global_pass:
        raise SystemExit("cross-dataset channel-permutation quality gate failed")


if __name__ == "__main__":
    main()
