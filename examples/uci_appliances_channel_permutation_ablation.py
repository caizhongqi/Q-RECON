from __future__ import annotations

import hashlib
import json

from qrecon.benchmarks import (
    benchmark_environment_manifest,
    summarize_proportion,
    summarize_scalar,
)
from qrecon.experiment import _build_model, _load_dataset, _seed_everything, _train
from qrecon.theory import channel_permutation_gradient_witness

DATASET_PATH = "data/UCI-appliances/energydata_complete.csv"
DATASET_SHA256 = "2820bf712ad0275cb18b85a05250926100d8e65ebb9f4d2d016ca91ea152a25d"
CHANNELS = ("T1", "T2", "T3", "T4", "T5", "T6", "T7")


def _scalar(values: list[float], label: str) -> dict[str, object]:
    return summarize_scalar(
        values,
        confidence_level=0.95,
        bootstrap_samples=2000,
        bootstrap_seed=20260715,
        label=f"uci-channel-ablation:{label}",
    ).to_dict()


def main() -> None:
    dataset_config = {
        "name": "multivariate_csv",
        "path": DATASET_PATH,
        "expected_file_sha256": DATASET_SHA256,
        "max_samples": 64,
        "context": 16,
        "horizon": 4,
        "stride": 8,
        "columns": list(CHANNELS),
    }
    training_config = {
        "epochs": 3,
        "batch_size": 8,
        "optimizer": "adamw",
        "learning_rate": 1e-3,
        "weight_decay": 1e-3,
    }
    variants = (
        {
            "name": "itransformer_shared_nonaffine_revin",
            "expected_equivariant": True,
            "seed": 241,
            "victim": {
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
            "name": "patchtst_shared_nonaffine_revin",
            "expected_equivariant": True,
            "seed": 251,
            "victim": {
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
        {
            "name": "itransformer_affine_revin_control",
            "expected_equivariant": False,
            "seed": 257,
            "victim": {
                "architecture": "itransformer",
                "d_model": 8,
                "n_heads": 2,
                "e_layers": 2,
                "d_ff": 16,
                "dropout": 0.0,
                "revin": True,
                "revin_affine": True,
            },
        },
        {
            "name": "patchtst_channel_specific_head_control",
            "expected_equivariant": False,
            "seed": 263,
            "victim": {
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
                "individual_head": True,
            },
        },
    )

    dataset, task, mode = _load_dataset({"seed": 241, "dataset": dataset_config})
    if task != "forecasting" or mode != "timeseries":
        raise RuntimeError("UCI symmetry ablation requires multivariate forecasting")
    inputs, targets = dataset.tensors
    channels = int(inputs.shape[-1])
    permutation = tuple(range(1, channels)) + (0,)
    sample_indices = tuple(range(40, 60))
    variant_reports: list[dict[str, object]] = []
    global_pass = True

    for declaration in variants:
        seed = int(declaration["seed"])
        _seed_everything(seed)
        model = _build_model(dataset, task, dict(declaration["victim"]))
        _train(model, dataset, task, training_config)
        witnesses = tuple(
            channel_permutation_gradient_witness(
                model,
                inputs[index : index + 1].clone(),
                targets[index : index + 1].clone(),
                permutation,
            )
            for index in sample_indices
        )
        expected_equivariant = bool(declaration["expected_equivariant"])
        collision_flags = [
            witness.nontrivial_private_collision
            and witness.prediction_equivariance_max_abs_error <= 1e-5
            and witness.loss_absolute_difference <= 1e-5
            and witness.gradient_max_abs_difference <= 1e-5
            and witness.gradient_relative_l2_difference <= 1e-5
            for witness in witnesses
        ]
        broken_flags = [
            witness.prediction_equivariance_max_abs_error > 1e-4
            and witness.gradient_relative_l2_difference > 1e-4
            for witness in witnesses
        ]
        variant_pass = (
            all(collision_flags) if expected_equivariant else all(broken_flags)
        )
        global_pass = global_pass and variant_pass
        variant_reports.append(
            {
                "name": declaration["name"],
                "expected_equivariant": expected_equivariant,
                "victim": declaration["victim"],
                "victim_class": type(model).__name__,
                "trainable_parameters": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
                "points": [witness.to_dict() for witness in witnesses],
                "summary": {
                    "certified_collision": summarize_proportion(
                        sum(collision_flags),
                        len(collision_flags),
                        confidence_level=0.95,
                    ).to_dict(),
                    "certified_symmetry_break": summarize_proportion(
                        sum(broken_flags),
                        len(broken_flags),
                        confidence_level=0.95,
                    ).to_dict(),
                    "prediction_equivariance_max_abs_error": _scalar(
                        [
                            witness.prediction_equivariance_max_abs_error
                            for witness in witnesses
                        ],
                        f"{declaration['name']}:prediction",
                    ),
                    "loss_absolute_difference": _scalar(
                        [witness.loss_absolute_difference for witness in witnesses],
                        f"{declaration['name']}:loss",
                    ),
                    "gradient_max_abs_difference": _scalar(
                        [witness.gradient_max_abs_difference for witness in witnesses],
                        f"{declaration['name']}:gradient-max",
                    ),
                    "gradient_relative_l2_difference": _scalar(
                        [
                            witness.gradient_relative_l2_difference
                            for witness in witnesses
                        ],
                        f"{declaration['name']}:gradient-relative",
                    ),
                    "orbit_size": _scalar(
                        [float(witness.fibre_bound.orbit_size) for witness in witnesses],
                        f"{declaration['name']}:orbit-size",
                    ),
                },
                "quality_gate_passed": variant_pass,
            }
        )

    manifest = {
        "schema_version": "qrecon.uci-channel-permutation-ablation.v1",
        "dataset": dataset_config,
        "training": training_config,
        "variants": list(variants),
        "sample_indices": list(sample_indices),
        "permutation": list(permutation),
        "collision_tolerance": 1e-5,
        "symmetry_break_threshold": 1e-4,
    }
    manifest_sha256 = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    payload = {
        "manifest": manifest,
        "manifest_sha256": manifest_sha256,
        "variants": variant_reports,
        "quality_gate": {
            "independent_real_multivariate_dataset": True,
            "immutable_source_hash_declared": True,
            "twenty_windows_per_variant": True,
            "two_equivariant_architectures": True,
            "two_explicit_symmetry_breaking_controls": True,
            "all_expected_outcomes_observed": global_pass,
            "passed": global_pass,
        },
        "environment": benchmark_environment_manifest(),
        "conclusion": (
            "Shared anonymous-channel iTransformer and PatchTST retain exact "
            "full-gradient collisions on the independent UCI source, while affine "
            "per-channel RevIN parameters and channel-specific PatchTST heads break "
            "the declared channel-permutation symmetry."
        ),
        "claim_boundary": (
            "The controls show that the impossibility follows from the declared "
            "architectural symmetry rather than all multivariate Transformers. Public "
            "semantic channel identities and channel-indexed learned parameters define "
            "different observation models that require separate fibre analysis."
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not global_pass:
        raise SystemExit("UCI channel-permutation ablation quality gate failed")


if __name__ == "__main__":
    main()
