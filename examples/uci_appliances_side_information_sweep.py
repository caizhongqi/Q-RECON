from __future__ import annotations

import json

from qrecon.benchmarks import (
    ModernTimeSeriesAttackManifest,
    benchmark_environment_manifest,
    run_channel_side_information_benchmark,
)

DATASET_PATH = "data/UCI-appliances/energydata_complete.csv"
DATASET_SHA256 = "2820bf712ad0275cb18b85a05250926100d8e65ebb9f4d2d016ca91ea152a25d"
CHANNELS = ("T1", "T2", "T3", "T4", "T5", "T6", "T7")
CALIBRATION_SIZES = (1, 2, 5, 10, 20, 40, 60)
EVALUATION_INDICES = tuple(range(64, 84))
PERMUTATION_SEEDS = (101, 103, 107, 109, 113)


def _manifest() -> ModernTimeSeriesAttackManifest:
    return ModernTimeSeriesAttackManifest(
        dataset={
            "name": "multivariate_csv",
            "path": DATASET_PATH,
            "expected_file_sha256": DATASET_SHA256,
            "max_samples": 96,
            "context": 16,
            "horizon": 4,
            "stride": 8,
            "columns": list(CHANNELS),
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
        victim_seed=227,
        attack_indices=EVALUATION_INDICES,
        attack_seeds=PERMUTATION_SEEDS,
        attack_batch_size=1,
        exact_tolerance=1e-5,
        relative_l2_threshold=1e-5,
        confidence_level=0.95,
        bootstrap_samples=2000,
        bootstrap_seed=20260715,
        minimum_publication_batches=len(EVALUATION_INDICES),
        minimum_publication_attack_seeds=len(PERMUTATION_SEEDS),
        publication_mode=True,
    )


def main() -> None:
    manifest = _manifest()
    reports: dict[str, object] = {}
    all_passed = True
    exact_rates: list[float] = []
    channel_accuracies: list[float] = []

    for size in CALIBRATION_SIZES:
        report = run_channel_side_information_benchmark(
            manifest,
            calibration_indices=tuple(range(size)),
            evaluation_indices=EVALUATION_INDICES,
            permutation_seeds=PERMUTATION_SEEDS,
            spectral_bins=4,
            variance_floor=1e-4,
            minimum_calibration_windows=size,
            minimum_permutation_trials=(
                len(EVALUATION_INDICES) * len(PERMUTATION_SEEDS)
            ),
        )
        payload = report.to_dict()
        reports[str(size)] = payload
        all_passed = all_passed and report.quality_gate.passed
        exact_rates.append(
            float(payload["summary"]["exact_labeled_order_recovery"]["rate"])
        )
        channel_accuracies.append(
            float(payload["summary"]["channel_accuracy"]["mean"])
        )

    total_trials = (
        len(CALIBRATION_SIZES)
        * len(EVALUATION_INDICES)
        * len(PERMUTATION_SEEDS)
    )
    payload = {
        "schema_version": "qrecon.uci-side-information-sweep.v1",
        "dataset": {
            "name": "UCI Appliances Energy Prediction",
            "csv_sha256": DATASET_SHA256,
            "selected_channels": list(CHANNELS),
        },
        "contract": {
            "calibration_sizes": list(CALIBRATION_SIZES),
            "evaluation_indices": list(EVALUATION_INDICES),
            "permutation_seeds": list(PERMUTATION_SEEDS),
            "attacker_receives_exact_orbit_representative": True,
            "evaluation_set_fixed_across_all_calibration_sizes": True,
            "target": "exact semantic channel order",
            "no_side_information_uniform_ceiling": 1.0 / 5040.0,
        },
        "reports": reports,
        "curve": [
            {
                "calibration_windows": size,
                "exact_order_success_rate": exact_rate,
                "mean_channel_accuracy": accuracy,
            }
            for size, exact_rate, accuracy in zip(
                CALIBRATION_SIZES, exact_rates, channel_accuracies
            )
        ],
        "quality_gate": {
            "immutable_real_source": True,
            "predeclared_calibration_sizes": True,
            "fixed_disjoint_private_evaluation_set": True,
            "seven_calibration_levels": len(CALIBRATION_SIZES) == 7,
            "seven_hundred_total_trials": total_trials == 700,
            "all_level_gates_passed": all_passed,
            "passed": all_passed,
        },
        "environment": benchmark_environment_manifest(),
        "interpretation": (
            "The curve measures one declared side-information attacker while holding the "
            "private evaluation windows and hidden permutation seeds fixed. Increasing "
            "public calibration can reduce practical anonymity, but observed failures are "
            "not a universal lower bound against Bayes-optimal external priors."
        ),
        "claim_boundary": (
            "The analytical 1/5040 ceiling applies only with no semantic side information. "
            "Each calibration level defines a different observation model. Full public "
            "channel identities collapse this permutation ambiguity regardless of the "
            "empirical matcher curve."
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not all_passed:
        raise SystemExit("UCI side-information sweep quality gate failed")


if __name__ == "__main__":
    main()
