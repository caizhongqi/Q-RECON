from __future__ import annotations

import hashlib
import json

from qrecon.benchmarks import (
    benchmark_environment_manifest,
    summarize_proportion,
    summarize_scalar,
)
from qrecon.experiment import _build_model, _load_dataset, _seed_everything, _train
from qrecon.theory import channel_permutation_training_transcript_witness

ETTM1_SHA256 = "6ce1759b1a18e3328421d5d75fadcb316c449fcd7cec32820c8dafda71986c9e"


def _scalar(values: list[float], label: str) -> dict[str, object]:
    return summarize_scalar(
        values,
        confidence_level=0.95,
        bootstrap_samples=2000,
        bootstrap_seed=20260715,
        label=f"training-transcript:{label}",
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
            "name": "anonymous_itransformer",
            "expect_same_transcript": True,
            "seed": 83,
            "victim": {
                "architecture": "itransformer",
                "d_model": 8,
                "n_heads": 2,
                "e_layers": 2,
                "d_ff": 16,
                "dropout": 0.0,
                "revin": False,
            },
        },
        {
            "name": "channel_specific_patchtst_head",
            "expect_same_transcript": False,
            "seed": 89,
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
        },
    )

    dataset, task, mode = _load_dataset({"seed": 83, "dataset": dataset_config})
    if task != "forecasting" or mode != "timeseries":
        raise RuntimeError("training-transcript study requires forecasting data")
    inputs, targets = dataset.tensors
    channels = int(inputs.shape[-1])
    permutation = tuple(range(1, channels)) + (0,)
    reports: list[dict[str, object]] = []
    global_pass = True

    for declaration in variants:
        _seed_everything(int(declaration["seed"]))
        model = _build_model(dataset, task, dict(declaration["victim"]))
        _train(model, dataset, task, training_config)
        witnesses = tuple(
            channel_permutation_training_transcript_witness(
                model,
                inputs[index : index + 1].clone(),
                targets[index : index + 1].clone(),
                permutation,
                optimizer="adamw",
                steps=3,
                learning_rate=1e-3,
                weight_decay=1e-3,
            )
            for index in range(20)
        )
        expected = bool(declaration["expect_same_transcript"])
        identical = [
            witness.maximum_loss_absolute_difference <= 1e-5
            and witness.maximum_gradient_absolute_difference <= 1e-5
            and witness.maximum_parameter_absolute_difference <= 1e-5
            and witness.maximum_optimizer_state_absolute_difference <= 1e-5
            and witness.final_model_delta_difference.relative_l2_difference <= 1e-4
            for witness in witnesses
        ]
        broken = [
            witness.maximum_loss_absolute_difference > 1e-5
            and witness.maximum_gradient_absolute_difference > 1e-4
            and witness.final_model_delta_difference.relative_l2_difference > 1e-4
            for witness in witnesses
        ]
        passed = all(identical) if expected else all(broken)
        global_pass = global_pass and passed
        reports.append(
            {
                "name": declaration["name"],
                "expect_same_transcript": expected,
                "victim": declaration["victim"],
                "victim_class": type(model).__name__,
                "trainable_parameters": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
                "points": [witness.to_dict() for witness in witnesses],
                "summary": {
                    "identical_transcript": summarize_proportion(
                        sum(identical), len(identical), confidence_level=0.95
                    ).to_dict(),
                    "symmetry_broken": summarize_proportion(
                        sum(broken), len(broken), confidence_level=0.95
                    ).to_dict(),
                    "maximum_loss_absolute_difference": _scalar(
                        [w.maximum_loss_absolute_difference for w in witnesses],
                        f"{declaration['name']}:loss",
                    ),
                    "maximum_gradient_absolute_difference": _scalar(
                        [w.maximum_gradient_absolute_difference for w in witnesses],
                        f"{declaration['name']}:gradient",
                    ),
                    "maximum_parameter_absolute_difference": _scalar(
                        [w.maximum_parameter_absolute_difference for w in witnesses],
                        f"{declaration['name']}:parameters",
                    ),
                    "maximum_optimizer_state_absolute_difference": _scalar(
                        [
                            w.maximum_optimizer_state_absolute_difference
                            for w in witnesses
                        ],
                        f"{declaration['name']}:optimizer-state",
                    ),
                    "final_model_delta_relative_l2_difference": _scalar(
                        [
                            w.final_model_delta_difference.relative_l2_difference
                            for w in witnesses
                        ],
                        f"{declaration['name']}:model-delta",
                    ),
                    "uniform_exact_labeled_recovery_ceiling": _scalar(
                        [
                            w.fibre_bound.uniform_exact_ordered_recovery_ceiling
                            for w in witnesses
                        ],
                        f"{declaration['name']}:ceiling",
                    ),
                },
                "quality_gate_passed": passed,
            }
        )

    manifest = {
        "schema_version": "qrecon.channel-permutation-training-transcript.v1",
        "dataset": dataset_config,
        "victim_training": training_config,
        "local_release_optimizer": {
            "name": "adamw",
            "steps": 3,
            "learning_rate": 1e-3,
            "weight_decay": 1e-3,
        },
        "sample_indices": list(range(20)),
        "permutation": list(permutation),
        "variants": list(variants),
    }
    manifest_sha256 = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    payload = {
        "manifest": manifest,
        "manifest_sha256": manifest_sha256,
        "variants": reports,
        "quality_gate": {
            "real_multivariate_dataset": True,
            "immutable_source_hash_declared": True,
            "twenty_windows_per_variant": True,
            "three_adamw_local_steps": True,
            "anonymous_model_collision": bool(reports[0]["quality_gate_passed"]),
            "channel_specific_control_breaks_collision": bool(
                reports[1]["quality_gate_passed"]
            ),
            "passed": global_pass,
        },
        "environment": benchmark_environment_manifest(),
        "theorem": (
            "A loss identity that holds for every parameter value implies identical "
            "deterministic first-order optimizer trajectories by induction. Therefore "
            "gradients, AdamW state, checkpoints and final model deltas preserve the "
            "same channel-permutation fibre."
        ),
        "claim_boundary": (
            "The executable study uses deterministic full-batch MSE and three AdamW "
            "steps. The theorem extends under shared label-independent randomness; "
            "channel metadata or channel-indexed parameters break the premise."
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not global_pass:
        raise SystemExit("ETTm1 training-transcript permutation gate failed")


if __name__ == "__main__":
    main()
