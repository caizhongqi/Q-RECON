from __future__ import annotations

import hashlib
import json
from pathlib import Path

from qrecon.benchmarks import (
    ModernTimeSeriesAttackManifest,
    benchmark_environment_manifest,
    run_channel_permutation_fibre_benchmark,
)


DATASET = {
    "name": "uci_appliances",
    "path": "data/UCI-appliances/energydata_complete.csv",
    "sha256": "2820bf712ad0275cb18b85a05250926100d8e65ebb9f4d2d016ca91ea152a25d",
    "columns": ["T1", "T2", "T3", "T4", "T5", "T6", "T7"],
}

ARCHITECTURES = (
    {
        "name": "itransformer_d32_l2",
        "config": {
            "architecture": "itransformer",
            "d_model": 32,
            "n_heads": 4,
            "e_layers": 2,
            "d_ff": 64,
            "dropout": 0.0,
            "revin": True,
            "revin_affine": False,
        },
    },
    {
        "name": "patchtst_d32_l3",
        "config": {
            "architecture": "patchtst",
            "patch_len": 16,
            "stride": 8,
            "padding_patch": True,
            "d_model": 32,
            "n_heads": 4,
            "e_layers": 3,
            "d_ff": 64,
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


def _manifest(victim: dict[str, object], seed: int) -> ModernTimeSeriesAttackManifest:
    return ModernTimeSeriesAttackManifest(
        dataset={
            "name": "multivariate_csv",
            "path": DATASET["path"],
            "expected_file_sha256": DATASET["sha256"],
            "max_samples": 64,
            "context": 96,
            "horizon": 24,
            "stride": 24,
            "columns": list(DATASET["columns"]),
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
        victim_seed=seed,
        attack_indices=tuple(range(10)),
        attack_seeds=(101,),
        attack_batch_size=1,
        exact_tolerance=1e-5,
        relative_l2_threshold=1e-5,
        confidence_level=0.95,
        bootstrap_samples=2000,
        bootstrap_seed=20260715,
        minimum_publication_batches=10,
        minimum_publication_attack_seeds=1,
        publication_mode=True,
    )


def main() -> None:
    observed = _sha256(DATASET["path"])
    if observed != DATASET["sha256"]:
        raise RuntimeError(
            "UCI Appliances SHA256 mismatch: "
            f"expected {DATASET['sha256']}, observed {observed}"
        )

    reports: list[dict[str, object]] = []
    all_passed = True
    for architecture_index, architecture in enumerate(ARCHITECTURES):
        report = run_channel_permutation_fibre_benchmark(
            _manifest(
                architecture["config"],
                seed=811 + architecture_index,
            ),
            # The theorem is exact. This tolerance only audits the larger float32
            # implementation at context 96 / horizon 24 / d_model 32.
            tolerance=5e-5,
        )
        all_passed = all_passed and report.quality_gate.passed
        reports.append(
            {
                "dataset": DATASET["name"],
                "source_sha256": observed,
                "architecture": architecture["name"],
                "victim_config": architecture["config"],
                "fibre": report.to_dict(),
                "quality_gate_passed": report.quality_gate.passed,
            }
        )

    declaration = {
        "schema_version": "qrecon.uci-large-model-channel-permutation-stress.v1",
        "dataset": DATASET,
        "architectures": list(ARCHITECTURES),
        "context": 96,
        "horizon": 24,
        "windows_per_cell": 10,
        "cells": len(ARCHITECTURES),
        "primary_gradient_fibre_points": 20,
        "channels": 7,
        "private_object": "ordered histories and ordered forecast targets",
        "numerical_tolerance": 5e-5,
    }
    declaration_sha256 = hashlib.sha256(
        json.dumps(declaration, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    payload = {
        "declaration": declaration,
        "declaration_sha256": declaration_sha256,
        "reports": reports,
        "quality_gate": {
            "independent_non_ett_real_dataset": True,
            "two_modern_architectures": len(ARCHITECTURES) == 2,
            "long_context_and_horizon": True,
            "ten_windows_per_cell": all(
                report["fibre"]["summary"]["points"] == 10 for report in reports
            ),
            "source_sha256_locked": observed == DATASET["sha256"],
            "all_fibre_gates_passed": all_passed,
            "twenty_primary_fibre_points": sum(
                int(report["fibre"]["summary"]["points"]) for report in reports
            )
            == 20,
            "passed": all_passed,
        },
        "environment": benchmark_environment_manifest(),
        "conclusion": (
            "The anonymous-channel gradient fibre is reproduced on an independent "
            "UCI building-sensor dataset at context 96, horizon 24 and d_model 32 "
            "for a two-layer iTransformer and a three-layer shared-head PatchTST."
        ),
        "claim_boundary": (
            "This is a long-context numerical stress certificate for an exact "
            "information-theoretic theorem. It is not a pinned-hardware runtime "
            "benchmark, a universal semantic-privacy guarantee, or a positive "
            "quantum-speedup result."
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["quality_gate"]["passed"]:
        raise SystemExit("UCI large-model channel-permutation stress gate failed")


if __name__ == "__main__":
    main()
