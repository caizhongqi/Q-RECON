from __future__ import annotations

import hashlib
import json
from pathlib import Path

from qrecon.benchmarks import (
    ModernTimeSeriesAttackManifest,
    benchmark_environment_manifest,
    run_channel_permutation_fibre_benchmark,
)


DATASET_PATH = "data/UCI-appliances/energydata_complete.csv"
DATASET_SHA256 = "2820bf712ad0275cb18b85a05250926100d8e65ebb9f4d2d016ca91ea152a25d"
CHANNELS = ("T1", "T2", "T3", "T4", "T5", "T6", "T7")
BATCH_SIZES = (1, 2, 4, 8)
BATCH_STARTS = tuple(range(0, 50, 5))

ARCHITECTURES = (
    {
        "name": "itransformer",
        "seed": 901,
        "config": {
            "architecture": "itransformer",
            "d_model": 8,
            "n_heads": 2,
            "e_layers": 1,
            "d_ff": 16,
            "dropout": 0.0,
            "revin": True,
            "revin_affine": False,
        },
    },
    {
        "name": "patchtst",
        "seed": 907,
        "config": {
            "architecture": "patchtst",
            "patch_len": 8,
            "stride": 4,
            "padding_patch": True,
            "d_model": 8,
            "n_heads": 2,
            "e_layers": 1,
            "d_ff": 16,
            "dropout": 0.0,
            "head_dropout": 0.0,
            "revin": True,
            "revin_affine": False,
            "individual_head": False,
        },
    },
)


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest(
    victim: dict[str, object],
    *,
    victim_seed: int,
    batch_size: int,
) -> ModernTimeSeriesAttackManifest:
    return ModernTimeSeriesAttackManifest(
        dataset={
            "name": "multivariate_csv",
            "path": DATASET_PATH,
            "expected_file_sha256": DATASET_SHA256,
            "max_samples": 64,
            "context": 32,
            "horizon": 8,
            "stride": 8,
            "columns": list(CHANNELS),
        },
        victim=dict(victim),
        training={
            "epochs": 1,
            "batch_size": 8,
            "optimizer": "adamw",
            "learning_rate": 5e-4,
            "weight_decay": 1e-3,
        },
        attack={
            "prior": "direct",
            "known_target": False,
            "steps": 1,
            "learning_rate": 0.01,
        },
        victim_seed=int(victim_seed),
        attack_indices=BATCH_STARTS,
        attack_seeds=(101,),
        attack_batch_size=int(batch_size),
        exact_tolerance=1e-5,
        relative_l2_threshold=1e-5,
        confidence_level=0.95,
        bootstrap_samples=2000,
        bootstrap_seed=20260715,
        minimum_publication_batches=len(BATCH_STARTS),
        minimum_publication_attack_seeds=1,
        publication_mode=True,
    )


def main() -> None:
    observed_sha256 = _sha256(DATASET_PATH)
    if observed_sha256 != DATASET_SHA256:
        raise RuntimeError(
            f"UCI SHA256 mismatch: expected {DATASET_SHA256}, observed {observed_sha256}"
        )

    reports: list[dict[str, object]] = []
    maximum_gradient_error = 0.0
    maximum_output_error = 0.0
    all_passed = True
    for architecture in ARCHITECTURES:
        for batch_size in BATCH_SIZES:
            report = run_channel_permutation_fibre_benchmark(
                _manifest(
                    architecture["config"],
                    victim_seed=int(architecture["seed"]),
                    batch_size=batch_size,
                ),
                tolerance=1e-5,
            )
            all_passed = all_passed and report.quality_gate.passed
            maximum_gradient_error = max(
                maximum_gradient_error,
                max(point.maximum_gradient_invariance_error for point in report.points),
            )
            maximum_output_error = max(
                maximum_output_error,
                max(point.maximum_output_equivariance_error for point in report.points),
            )
            reports.append(
                {
                    "architecture": architecture["name"],
                    "batch_size": batch_size,
                    "report": report.to_dict(),
                    "quality_gate_passed": report.quality_gate.passed,
                }
            )

    observed_points = sum(
        int(item["report"]["summary"]["points"]) for item in reports
    )
    all_orbits_5040 = all(
        all(int(point["orbit_size"]) == 5040 for point in item["report"]["points"])
        for item in reports
    )
    declaration = {
        "schema_version": "qrecon.uci-batch-size-channel-fibre.v1",
        "dataset_sha256": observed_sha256,
        "architectures": [item["name"] for item in ARCHITECTURES],
        "batch_sizes": list(BATCH_SIZES),
        "batch_starts": list(BATCH_STARTS),
        "context": 32,
        "horizon": 8,
        "channels": len(CHANNELS),
        "points_per_cell": len(BATCH_STARTS),
        "expected_total_points": (
            len(ARCHITECTURES) * len(BATCH_SIZES) * len(BATCH_STARTS)
        ),
    }
    declaration_sha256 = hashlib.sha256(
        json.dumps(declaration, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    quality_gate = {
        "immutable_non_ett_source": observed_sha256 == DATASET_SHA256,
        "two_modern_architectures": len(ARCHITECTURES) == 2,
        "predeclared_batch_sizes_1_2_4_8": BATCH_SIZES == (1, 2, 4, 8),
        "ten_batches_per_cell": all(
            int(item["report"]["summary"]["points"]) == 10 for item in reports
        ),
        "eighty_primary_fibre_points": observed_points == 80,
        "all_orbits_equal_5040": all_orbits_5040,
        "all_fibre_quality_gates_passed": all_passed,
    }
    quality_gate["passed"] = all(quality_gate.values())
    payload = {
        "declaration": declaration,
        "declaration_sha256": declaration_sha256,
        "reports": reports,
        "maximum_output_equivariance_error": maximum_output_error,
        "maximum_gradient_invariance_error": maximum_gradient_error,
        "quality_gate": quality_gate,
        "environment": benchmark_environment_manifest(),
        "conclusion": (
            "Simultaneous channel permutation remains one exact gradient fibre for "
            "batch sizes 1, 2, 4 and 8 on both anonymous-channel modern victims."
        ),
        "claim_boundary": (
            "This is a numerical witness for an exact algebraic batch-size-independent "
            "theorem. It does not imply semantic privacy when labels or channel-indexed "
            "parameters are public, and it does not establish quantum advantage."
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not quality_gate["passed"]:
        raise SystemExit("UCI batch-size fibre quality gate failed")


if __name__ == "__main__":
    main()
