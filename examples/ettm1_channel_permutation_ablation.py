from __future__ import annotations

import hashlib
import json

from qrecon.benchmarks import (
    benchmark_environment_manifest,
    summarize_proportion,
    summarize_scalar,
)
from qrecon.experiment import _build_model, _load_dataset, _seed_everything, _train
from qrecon.models import RevIN
from qrecon.theory import channel_permutation_gradient_witness

ETTM1_SHA256 = "6ce1759b1a18e3328421d5d75fadcb316c449fcd7cec32820c8dafda71986c9e"


def _scalar(values: list[float], label: str) -> dict[str, object]:
    return summarize_scalar(
        values,
        confidence_level=0.95,
        bootstrap_samples=2000,
        bootstrap_seed=20260715,
        label=f"channel-ablation:{label}",
    ).to_dict()


def main() -> None:
    dataset_config = {
        "name": "multivariate_csv",
        "path": "data/ETT-small/ETTm1.csv",
        "expected_file_sha256": ETTM1_SHA256,
        "max_samples": 32,
        "context": 16,
        "horizon": 4,
        "stride": 4,
        "columns": ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"],
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
            "seed": 47,
            "victim": {
                "architecture": "itransformer",
                "d_model": 8,
                "n_heads": 2,
                "e_layers": 2,
                "d_ff": 16,
                "dropout": 0.0,
                "revin": False,
            },
            "install_nonaffine_revin": True,
        },
        {
            "name": "patchtst_shared_nonaffine_revin",
            "expected_equivariant": True,
            "seed": 53,
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
                "revin": False,
                "individual_head": False,
            },
            "install_nonaffine_revin": True,
        },
        {
            "name": "patchtst_channel_specific_head",
            "expected_equivariant": False,
            "seed": 59,
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
                "revin": False,
                "individual_head": True,
            },
            "install_nonaffine_revin": True,
        },
    )

    dataset, task, mode = _load_dataset({"seed": 47, "dataset": dataset_config})
    if task != "forecasting" or mode != "timeseries":
        raise RuntimeError("channel-symmetry ablation requires multivariate forecasting")
    inputs, targets = dataset.tensors
    channels = int(inputs.shape[-1])
    permutation = tuple(range(1, channels)) + (0,)
    variant_reports: list[dict[str, object]] = []
    global_pass = True

    for declaration in variants:
        seed = int(declaration["seed"])
        _seed_everything(seed)
        model = _build_model(dataset, task, dict(declaration["victim"]))
        if bool(declaration["install_nonaffine_revin"]):
            model.revin = RevIN(channels, affine=False)
        _train(model, dataset, task, training_config)
        witnesses = tuple(
            channel_permutation_gradient_witness(
                model,
                inputs[index : index + 1].clone(),
                targets[index : index + 1].clone(),
                permutation,
            )
            for index in range(20)
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
                "nonaffine_revin_installed": bool(
                    declaration["install_nonaffine_revin"]
                ),
                "victim_class": type(model).__name__,
                "trainable_parameters": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
                "points": [witness.to_dict() for witness in witnesses],
                "summary": {
                    "certified_collision": summarize_proportion(
                        sum(collision_flags), len(collision_flags), confidence_level=0.95
                    ).to_dict(),
                    "certified_symmetry_break": summarize_proportion(
                        sum(broken_flags), len(broken_flags), confidence_level=0.95
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
                    "uniform_exact_labeled_recovery_ceiling": _scalar(
                        [
                            witness.fibre_bound.uniform_exact_ordered_recovery_ceiling
                            for witness in witnesses
                        ],
                        f"{declaration['name']}:ceiling",
                    ),
                },
                "quality_gate_passed": variant_pass,
            }
        )

    manifest = {
        "schema_version": "qrecon.channel-permutation-ablation.v1",
        "dataset": dataset_config,
        "training": training_config,
        "variants": list(variants),
        "sample_indices": list(range(20)),
        "permutation": list(permutation),
    }
    manifest_sha256 = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    payload = {
        "manifest": manifest,
        "manifest_sha256": manifest_sha256,
        "variants": variant_reports,
        "quality_gate": {
            "real_multivariate_dataset": True,
            "immutable_source_hash_declared": True,
            "twenty_windows_per_variant": True,
            "two_equivariant_architectures": True,
            "explicit_symmetry_breaking_control": True,
            "passed": global_pass,
        },
        "environment": benchmark_environment_manifest(),
        "conclusion": (
            "Shared anonymous-channel iTransformer and PatchTST retain exact "
            "full-gradient collisions even with non-affine per-record normalization, "
            "whereas channel-specific PatchTST heads break the collision."
        ),
        "claim_boundary": (
            "The control demonstrates that the impossibility is caused by a declared "
            "architectural symmetry, not by a universal property of all multivariate "
            "Transformers. Public semantic channel identities or channel-indexed "
            "parameters require a different fibre analysis."
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not global_pass:
        raise SystemExit("ETTm1 channel-permutation ablation gate failed")


if __name__ == "__main__":
    main()
