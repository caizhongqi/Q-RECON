from __future__ import annotations

import hashlib
import json
from pathlib import Path

from qrecon.benchmarks import (
    ModernTimeSeriesAttackManifest,
    benchmark_environment_manifest,
    run_channel_permutation_fibre_benchmark,
)


DATASETS = (
    {
        "name": "ettm1",
        "path": "data/ETT-small/ETTm1.csv",
        "sha256": "6ce1759b1a18e3328421d5d75fadcb316c449fcd7cec32820c8dafda71986c9e",
    },
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


def _manifest(dataset: dict[str, str], victim: dict[str, object], seed: int):
    return ModernTimeSeriesAttackManifest(
        dataset={
            "name": "multivariate_csv",
            "path": dataset["path"],
            "expected_file_sha256": dataset["sha256"],
            "max_samples": 64,
            "context": 96,
            "horizon": 24,
            "stride": 24,
            "columns": ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"],
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
    reports: list[dict[str, object]] = []
    all_passed = True
    for dataset_index, dataset in enumerate(DATASETS):
        observed = _sha256(dataset["path"])
        if observed != dataset["sha256"]:
            raise RuntimeError(
                f"{dataset['name']} SHA256 mismatch: expected {dataset['sha256']}, observed {observed}"
            )
        for architecture_index, architecture in enumerate(ARCHITECTURES):
            report = run_channel_permutation_fibre_benchmark(
                _manifest(
                    dataset,
                    architecture["config"],
                    seed=701 + 10 * dataset_index + architecture_index,
                ),
                # The theorem is exact. This tolerance only certifies floating-point
                # execution of the generator identities at the larger declared scale.
                tolerance=5e-5,
            )
            all_passed = all_passed and report.quality_gate.passed
            reports.append(
                {
                    "dataset": dataset["name"],
                    "source_sha256": observed,
                    "architecture": architecture["name"],
                    "victim_config": architecture["config"],
                    "fibre": report.to_dict(),
                    "quality_gate_passed": report.quality_gate.passed,
                }
            )

    declaration = {
        "schema_version": "qrecon.large-model-channel-permutation-stress.v1",
        "datasets": list(DATASETS),
        "architectures": list(ARCHITECTURES),
        "context": 96,
        "horizon": 24,
        "windows_per_cell": 10,
        "cells": len(DATASETS) * len(ARCHITECTURES),
        "primary_gradient_fibre_points": 60,
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
            "three_real_datasets": len(DATASETS) == 3,
            "two_modern_architectures": len(ARCHITECTURES) == 2,
            "long_context_and_horizon": True,
            "ten_windows_per_cell": all(
                report["fibre"]["summary"]["points"] == 10 for report in reports
            ),
            "all_sources_sha256_locked": True,
            "all_fibre_gates_passed": all_passed,
            "passed": all_passed,
        },
        "environment": benchmark_environment_manifest(),
        "conclusion": (
            "The exact anonymous-channel gradient fibre is reproduced at context 96, "
            "horizon 24 and d_model 32 across ETTm1, ETTm2 and ETTh1 for both a "
            "two-layer iTransformer and a three-layer shared-head PatchTST."
        ),
        "claim_boundary": (
            "This is a numerical stress certificate for an exact theorem, not a runtime "
            "benchmark or a claim that semantic channel labels are unavailable in every deployment."
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["quality_gate"]["passed"]:
        raise SystemExit("large-model channel-permutation stress gate failed")


if __name__ == "__main__":
    main()
