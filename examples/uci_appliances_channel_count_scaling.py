from __future__ import annotations

import argparse
import json
import math

from qrecon.benchmarks import (
    benchmark_environment_manifest,
    summarize_proportion,
    summarize_scalar,
)
from qrecon.experiment import _build_model, _load_dataset, _seed_everything, _train
from qrecon.theory import channel_permutation_gradient_witness

DATASET_PATH = "data/UCI-appliances/energydata_complete.csv"
DATASET_SHA256 = "2820bf712ad0275cb18b85a05250926100d8e65ebb9f4d2d016ca91ea152a25d"
CHANNEL_SETS: dict[int, tuple[str, ...]] = {
    3: ("T1", "T2", "T3"),
    5: ("T1", "T2", "T3", "T4", "T5"),
    7: ("T1", "T2", "T3", "T4", "T5", "T6", "T7"),
    9: ("T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9"),
    10: ("T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T_out"),
}
ARCHITECTURES: dict[str, dict[str, object]] = {
    "itransformer": {
        "architecture": "itransformer",
        "d_model": 8,
        "n_heads": 2,
        "e_layers": 1,
        "d_ff": 16,
        "dropout": 0.0,
        "revin": True,
        "revin_affine": False,
    },
    "patchtst": {
        "architecture": "patchtst",
        "patch_len": 4,
        "stride": 2,
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
}
ARCHITECTURE_SEEDS = {"itransformer": 281, "patchtst": 283}
SAMPLE_INDICES = tuple(range(40, 50))
TOLERANCE = 1e-5


def _scalar(values: list[float], label: str) -> dict[str, object]:
    return summarize_scalar(
        values,
        confidence_level=0.95,
        bootstrap_samples=2000,
        bootstrap_seed=20260715,
        label=f"uci-channel-count:{label}",
    ).to_dict()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=tuple(ARCHITECTURES), required=True)
    args = parser.parse_args()
    architecture = str(args.architecture)
    architecture_reports: list[dict[str, object]] = []
    all_passed = True

    for channel_count, columns in CHANNEL_SETS.items():
        dataset_config = {
            "name": "multivariate_csv",
            "path": DATASET_PATH,
            "expected_file_sha256": DATASET_SHA256,
            "max_samples": 64,
            "context": 16,
            "horizon": 4,
            "stride": 8,
            "columns": list(columns),
        }
        victim_config = dict(ARCHITECTURES[architecture])
        victim_seed = ARCHITECTURE_SEEDS[architecture] + channel_count
        _seed_everything(victim_seed)
        dataset, task, mode = _load_dataset(
            {"seed": victim_seed, "dataset": dataset_config}
        )
        if task != "forecasting" or mode != "timeseries":
            raise RuntimeError("channel-count scaling requires multivariate forecasting")
        training = {
            "epochs": 2,
            "batch_size": 8,
            "optimizer": "adamw",
            "learning_rate": 1e-3,
            "weight_decay": 1e-3,
        }
        model = _build_model(dataset, task, victim_config)
        _train(model, dataset, task, training)
        inputs, targets = dataset.tensors
        permutation = tuple(range(1, channel_count)) + (0,)
        witnesses = tuple(
            channel_permutation_gradient_witness(
                model,
                inputs[index : index + 1].clone(),
                targets[index : index + 1].clone(),
                permutation,
            )
            for index in SAMPLE_INDICES
        )
        expected_orbit = math.factorial(channel_count)
        collision_flags = [
            witness.nontrivial_private_collision
            and witness.prediction_equivariance_max_abs_error <= TOLERANCE
            and witness.loss_absolute_difference <= TOLERANCE
            and witness.gradient_max_abs_difference <= TOLERANCE
            and witness.gradient_relative_l2_difference <= TOLERANCE
            and witness.fibre_bound.orbit_size == expected_orbit
            and math.isclose(
                witness.fibre_bound.uniform_exact_ordered_recovery_ceiling,
                1.0 / expected_orbit,
                rel_tol=1e-12,
                abs_tol=0.0,
            )
            for witness in witnesses
        ]
        cell_pass = all(collision_flags)
        all_passed = all_passed and cell_pass
        architecture_reports.append(
            {
                "channel_count": channel_count,
                "columns": list(columns),
                "victim_seed": victim_seed,
                "trainable_parameters": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
                "permutation": list(permutation),
                "expected_orbit_size": expected_orbit,
                "uniform_exact_order_success_ceiling": 1.0 / expected_orbit,
                "points": [witness.to_dict() for witness in witnesses],
                "summary": {
                    "certified_collision": summarize_proportion(
                        sum(collision_flags),
                        len(collision_flags),
                        confidence_level=0.95,
                    ).to_dict(),
                    "prediction_equivariance_max_abs_error": _scalar(
                        [
                            witness.prediction_equivariance_max_abs_error
                            for witness in witnesses
                        ],
                        f"{architecture}:{channel_count}:prediction",
                    ),
                    "loss_absolute_difference": _scalar(
                        [witness.loss_absolute_difference for witness in witnesses],
                        f"{architecture}:{channel_count}:loss",
                    ),
                    "gradient_max_abs_difference": _scalar(
                        [witness.gradient_max_abs_difference for witness in witnesses],
                        f"{architecture}:{channel_count}:gradient-max",
                    ),
                    "gradient_relative_l2_difference": _scalar(
                        [
                            witness.gradient_relative_l2_difference
                            for witness in witnesses
                        ],
                        f"{architecture}:{channel_count}:gradient-relative",
                    ),
                },
                "quality_gate_passed": cell_pass,
            }
        )

    payload = {
        "schema_version": "qrecon.uci-channel-count-scaling.v1",
        "dataset": {
            "name": "UCI Appliances Energy Prediction",
            "path": DATASET_PATH,
            "csv_sha256": DATASET_SHA256,
            "context": 16,
            "horizon": 4,
            "stride": 8,
        },
        "architecture": architecture,
        "victim": ARCHITECTURES[architecture],
        "sample_indices": list(SAMPLE_INDICES),
        "cells": architecture_reports,
        "quality_gate": {
            "immutable_non_ett_source": True,
            "five_predeclared_channel_scales": len(CHANNEL_SETS) == 5,
            "ten_windows_per_scale": len(SAMPLE_INDICES) == 10,
            "fifty_total_fibre_points": len(CHANNEL_SETS) * len(SAMPLE_INDICES) == 50,
            "all_cells_match_factorial_orbit": all_passed,
            "passed": all_passed,
        },
        "environment": benchmark_environment_manifest(),
        "conclusion": (
            "For anonymous, channel-permutation-equivariant victims, the observed real-data "
            "fibre grows from 3! through 10!, and the uniform exact semantic-order ceiling "
            "falls from 1/6 to 1/3,628,800 while the full-gradient witness remains invariant."
        ),
        "claim_boundary": (
            "The factorial ceiling is conditional on private channel order and distinct "
            "channel signatures. It is not an empirical hardness estimate for content "
            "recovery, nor does it apply after public semantic labels or channel-indexed "
            "parameters remove the symmetry."
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not all_passed:
        raise SystemExit("UCI channel-count scaling quality gate failed")


if __name__ == "__main__":
    main()
