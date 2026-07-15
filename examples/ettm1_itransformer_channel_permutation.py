from __future__ import annotations

import hashlib
import json

from qrecon.benchmarks import (
    benchmark_environment_manifest,
    summarize_proportion,
    summarize_scalar,
)
from qrecon.experiment import _build_model, _load_dataset, _seed_everything, _train
from qrecon.theory.channel_permutation import channel_permutation_gradient_witness

ETTM1_SHA256 = "6ce1759b1a18e3328421d5d75fadcb316c449fcd7cec32820c8dafda71986c9e"


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
    victim_config = {
        "architecture": "itransformer",
        "d_model": 8,
        "n_heads": 2,
        "e_layers": 2,
        "d_ff": 16,
        "dropout": 0.0,
        # The theorem requires no channel-indexed learned parameters. The core
        # iTransformer remains permutation equivariant; per-channel affine RevIN
        # would encode channel identity in the parameter vector and is disabled.
        "revin": False,
    }
    training_config = {
        "epochs": 3,
        "batch_size": 8,
        "optimizer": "adamw",
        "learning_rate": 1e-3,
        "weight_decay": 1e-3,
    }

    _seed_everything(47)
    dataset, task, mode = _load_dataset({"seed": 47, "dataset": dataset_config})
    if task != "forecasting" or mode != "timeseries":
        raise RuntimeError("ETTm1 channel-permutation study requires forecasting data")
    inputs, targets = dataset.tensors
    model = _build_model(dataset, task, victim_config)
    _train(model, dataset, task, training_config)

    channels = int(inputs.shape[-1])
    permutation = tuple(range(1, channels)) + (0,)
    witnesses = tuple(
        channel_permutation_gradient_witness(
            model,
            inputs[index : index + 1].clone(),
            targets[index : index + 1].clone(),
            permutation,
        )
        for index in range(20)
    )

    prediction_errors = [
        witness.prediction_equivariance_max_abs_error for witness in witnesses
    ]
    loss_errors = [witness.loss_absolute_difference for witness in witnesses]
    gradient_errors = [witness.gradient_max_abs_difference for witness in witnesses]
    relative_gradient_errors = [
        witness.gradient_relative_l2_difference for witness in witnesses
    ]
    orbit_sizes = [float(witness.fibre_bound.orbit_size) for witness in witnesses]
    ceilings = [
        witness.fibre_bound.uniform_exact_ordered_recovery_ceiling
        for witness in witnesses
    ]
    nontrivial = sum(witness.nontrivial_private_collision for witness in witnesses)
    certified = sum(
        witness.nontrivial_private_collision
        and witness.prediction_equivariance_max_abs_error <= 1e-5
        and witness.loss_absolute_difference <= 1e-5
        and witness.gradient_max_abs_difference <= 1e-5
        and witness.gradient_relative_l2_difference <= 1e-5
        for witness in witnesses
    )
    passed = (
        len(witnesses) == 20
        and nontrivial == len(witnesses)
        and certified == len(witnesses)
        and str(victim_config["architecture"]).lower() == "itransformer"
        and victim_config["revin"] is False
    )

    manifest_payload = {
        "schema_version": "qrecon.channel-permutation-gradient.v1",
        "dataset": dataset_config,
        "victim": victim_config,
        "training": training_config,
        "sample_indices": list(range(20)),
        "permutation": list(permutation),
    }
    manifest_sha256 = hashlib.sha256(
        json.dumps(manifest_payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    payload = {
        "manifest": manifest_payload,
        "manifest_sha256": manifest_sha256,
        "victim_class": type(model).__name__,
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "points": [witness.to_dict() for witness in witnesses],
        "summary": {
            "nontrivial_private_collision": summarize_proportion(
                nontrivial, len(witnesses), confidence_level=0.95
            ).to_dict(),
            "certified_full_gradient_collision": summarize_proportion(
                certified, len(witnesses), confidence_level=0.95
            ).to_dict(),
            "prediction_equivariance_max_abs_error": summarize_scalar(
                prediction_errors,
                confidence_level=0.95,
                bootstrap_samples=2000,
                bootstrap_seed=1729,
                label="channel-permutation:prediction-error",
            ).to_dict(),
            "loss_absolute_difference": summarize_scalar(
                loss_errors,
                confidence_level=0.95,
                bootstrap_samples=2000,
                bootstrap_seed=1729,
                label="channel-permutation:loss-error",
            ).to_dict(),
            "gradient_max_abs_difference": summarize_scalar(
                gradient_errors,
                confidence_level=0.95,
                bootstrap_samples=2000,
                bootstrap_seed=1729,
                label="channel-permutation:gradient-max-error",
            ).to_dict(),
            "gradient_relative_l2_difference": summarize_scalar(
                relative_gradient_errors,
                confidence_level=0.95,
                bootstrap_samples=2000,
                bootstrap_seed=1729,
                label="channel-permutation:gradient-relative-error",
            ).to_dict(),
            "orbit_size": summarize_scalar(
                orbit_sizes,
                confidence_level=0.95,
                bootstrap_samples=2000,
                bootstrap_seed=1729,
                label="channel-permutation:orbit-size",
            ).to_dict(),
            "uniform_exact_labeled_recovery_ceiling": summarize_scalar(
                ceilings,
                confidence_level=0.95,
                bootstrap_samples=2000,
                bootstrap_seed=1729,
                label="channel-permutation:recovery-ceiling",
            ).to_dict(),
        },
        "quality_gate": {
            "real_multivariate_dataset": True,
            "immutable_source_hash_declared": True,
            "twenty_independent_windows": len(witnesses) == 20,
            "channel_identity_parameters_absent": victim_config["revin"] is False,
            "every_window_has_nontrivial_orbit": nontrivial == len(witnesses),
            "every_window_matches_full_gradient": certified == len(witnesses),
            "passed": passed,
        },
        "environment": benchmark_environment_manifest(),
        "theorem": (
            "For every channel-permutation-equivariant forecasting model and MSE "
            "loss, synchronously permuting input and target channels leaves the loss "
            "identical as a function of all parameters; therefore every full model "
            "gradient is identical. Under a uniform prior on a generic C-channel "
            "orbit, exact labeled-channel recovery is bounded by 1/C! for both "
            "classical and quantum estimators."
        ),
        "claim_boundary": (
            "This is an input-level nonlinear non-identifiability result for an "
            "anonymous-channel iTransformer. Channel embeddings, channel-specific "
            "heads, affine per-channel RevIN parameters, or trusted semantic side "
            "information can break the symmetry and require a separate analysis."
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit("ETTm1 iTransformer channel-permutation gate failed")


if __name__ == "__main__":
    main()
