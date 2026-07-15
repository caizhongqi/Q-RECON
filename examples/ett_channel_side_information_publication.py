from __future__ import annotations

import json

from qrecon.benchmarks import (
    ModernTimeSeriesAttackManifest,
    run_channel_side_information_benchmark,
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


def _manifest(path: str, sha256: str) -> ModernTimeSeriesAttackManifest:
    return ModernTimeSeriesAttackManifest(
        dataset={
            "name": "multivariate_csv",
            "path": path,
            "expected_file_sha256": sha256,
            "max_samples": 64,
            "context": 16,
            "horizon": 4,
            "stride": 4,
            "columns": ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"],
        },
        victim={
            "architecture": "itransformer",
            "d_model": 8,
            "n_heads": 2,
            "e_layers": 1,
            "d_ff": 16,
            "dropout": 0.0,
            "revin": True,
            "revin_affine": False,
        },
        training={"epochs": 0, "batch_size": 8, "optimizer": "adam"},
        attack={"prior": "direct", "steps": 1, "known_target": False},
        victim_seed=47,
        attack_indices=tuple(range(20, 40)),
        attack_seeds=(101, 103, 107, 109, 113),
        attack_batch_size=1,
        exact_tolerance=1e-5,
        relative_l2_threshold=1e-5,
        confidence_level=0.95,
        bootstrap_samples=2000,
        bootstrap_seed=20260715,
        minimum_publication_batches=20,
        minimum_publication_attack_seeds=5,
        publication_mode=True,
    )


def main() -> None:
    calibration_indices = tuple(range(20))
    evaluation_indices = tuple(range(20, 40))
    permutation_seeds = (101, 103, 107, 109, 113)
    reports: dict[str, object] = {}
    all_passed = True
    for name, path, sha256 in DATASETS:
        report = run_channel_side_information_benchmark(
            _manifest(path, sha256),
            calibration_indices=calibration_indices,
            evaluation_indices=evaluation_indices,
            permutation_seeds=permutation_seeds,
            spectral_bins=4,
            variance_floor=1e-4,
            minimum_calibration_windows=20,
            minimum_permutation_trials=100,
        )
        reports[name] = report.to_dict()
        all_passed = all_passed and report.quality_gate.passed

    payload = {
        "schema_version": "qrecon.ett-channel-side-information.v1",
        "calibration_contract": {
            "public_labeled_windows": list(calibration_indices),
            "private_evaluation_windows": list(evaluation_indices),
            "permutations_per_window": len(permutation_seeds),
            "attacker_receives_exact_orbit_representative": True,
            "target_is_exact_semantic_channel_assignment": True,
        },
        "datasets": reports,
        "quality_gate": {
            "three_immutable_real_datasets": True,
            "disjoint_calibration_and_evaluation": True,
            "one_hundred_trials_per_dataset": True,
            "all_dataset_gates_passed": all_passed,
            "passed": all_passed,
        },
        "interpretation": (
            "The 1/orbit no-side-information ceiling is conditional. Public labeled "
            "calibration data can statistically identify recovered anonymous channels; "
            "the measured matcher success quantifies how much this explicit side "
            "information shrinks the privacy guarantee."
        ),
        "claim_boundary": (
            "The matcher is a declared diagonal-Gaussian temporal feature model, not "
            "the Bayes-optimal side-information attacker. Failure of this matcher does "
            "not prove that semantic channel identity remains hidden from all external "
            "priors."
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not all_passed:
        raise SystemExit("ETT public side-information quality gate failed")


if __name__ == "__main__":
    main()
