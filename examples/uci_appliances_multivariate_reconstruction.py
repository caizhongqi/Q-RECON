from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch

from qrecon.attacks import GradientInversionAttack, leak_gradients
from qrecon.benchmarks import (
    benchmark_environment_manifest,
    summarize_proportion,
    summarize_scalar,
)
from qrecon.experiment import _build_model, _load_dataset, _prior, _seed_everything, _train
from qrecon.metrics import (
    permutation_invariant_channel_metrics,
    reconstruction_metrics,
)

DATASET_PATH = "data/UCI-appliances/energydata_complete.csv"
DATASET_SHA256 = "2820bf712ad0275cb18b85a05250926100d8e65ebb9f4d2d016ca91ea152a25d"
CHANNELS = ("T1", "T2", "T3", "T4", "T5", "T6", "T7")
ATTACK_INDICES = tuple(range(40, 60))
RESTART_SEEDS = (101, 103, 107)
EXACT_TOLERANCE = 0.1
RELATIVE_L2_THRESHOLD = 0.5

ARCHITECTURES: dict[str, dict[str, object]] = {
    "itransformer": {
        "architecture": "itransformer",
        "d_model": 4,
        "n_heads": 1,
        "e_layers": 1,
        "d_ff": 8,
        "dropout": 0.0,
        "revin": True,
        "revin_affine": False,
    },
    "patchtst": {
        "architecture": "patchtst",
        "patch_len": 4,
        "stride": 2,
        "padding_patch": True,
        "d_model": 4,
        "n_heads": 1,
        "e_layers": 1,
        "d_ff": 8,
        "dropout": 0.0,
        "head_dropout": 0.0,
        "revin": True,
        "revin_affine": False,
        "individual_head": False,
    },
}
ARCHITECTURE_SEEDS = {"itransformer": 271, "patchtst": 277}


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha256(*tensors: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _model_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _scalar(values: list[float], label: str) -> dict[str, object] | None:
    if not values:
        return None
    return summarize_scalar(
        values,
        confidence_level=0.95,
        bootstrap_samples=2000,
        bootstrap_seed=20260715,
        label=f"uci-multivariate-reconstruction:{label}",
    ).to_dict()


def _ordered_exact(reference: torch.Tensor, estimate: torch.Tensor) -> bool:
    return bool((reference.detach() - estimate.detach()).abs().max() <= EXACT_TOLERANCE)


def _run_attempt(
    *,
    model: torch.nn.Module,
    true_x: torch.Tensor,
    true_target: torch.Tensor,
    batch_index: int,
    restart_seed: int,
) -> dict[str, object]:
    started = time.perf_counter()
    try:
        _seed_everything(restart_seed)
        observed = leak_gradients(model, true_x, true_target, "forecasting")
        prior = _prior(
            tuple(true_x.shape),
            "timeseries",
            {"prior": "direct", "bounded": True},
        )
        attack = GradientInversionAttack(
            model=model,
            observed_gradients=observed,
            prior=prior,
            task="forecasting",
            mode="timeseries",
            known_target=None,
            target_shape=tuple(true_target.shape),
            steps=25,
            learning_rate=0.04,
            regularization=1e-4,
            optimizer_name="adam",
            match_mode="hybrid",
            layer_weighting="parameter",
            gradient_clip_norm=10.0,
            record_every=25,
        )
        result = attack.run()
        reconstructed_x = result.reconstruction.detach()
        reconstructed_target = result.reconstructed_target.detach()
        complete_reference = torch.cat((true_x, true_target), dim=1)
        complete_estimate = torch.cat((reconstructed_x, reconstructed_target), dim=1)
        channel_alignment = permutation_invariant_channel_metrics(
            complete_reference,
            complete_estimate,
            tolerance=EXACT_TOLERANCE,
        )
        assignment = list(channel_alignment.assignment)
        aligned_x = reconstructed_x[..., assignment]
        aligned_target = reconstructed_target[..., assignment]
        aligned_complete = torch.cat((aligned_x, aligned_target), dim=1)
        ordered_complete_metrics = reconstruction_metrics(
            complete_reference, complete_estimate, mode="timeseries"
        )
        aligned_complete_metrics = reconstruction_metrics(
            complete_reference, aligned_complete, mode="timeseries"
        )
        return {
            "status": "success",
            "batch_index": batch_index,
            "restart_seed": restart_seed,
            "seconds": time.perf_counter() - started,
            "best_objective": result.best_objective,
            "best_gradient_match": result.best_gradient_match,
            "best_step": result.best_step,
            "final_objective": result.final_objective,
            "final_gradient_match": result.final_gradient_match,
            "ordered_complete_metrics": ordered_complete_metrics,
            "aligned_complete_metrics": aligned_complete_metrics,
            "ordered_input_metrics": reconstruction_metrics(
                true_x, reconstructed_x, mode="timeseries"
            ),
            "aligned_input_metrics": reconstruction_metrics(
                true_x, aligned_x, mode="timeseries"
            ),
            "ordered_target_metrics": reconstruction_metrics(
                true_target, reconstructed_target, mode="timeseries"
            ),
            "aligned_target_metrics": reconstruction_metrics(
                true_target, aligned_target, mode="timeseries"
            ),
            "channel_alignment": channel_alignment.to_dict(),
            "ordered_exact_within_tolerance": _ordered_exact(
                complete_reference, complete_estimate
            ),
            "aligned_exact_within_tolerance": _ordered_exact(
                complete_reference, aligned_complete
            ),
        }
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        return {
            "status": "failed",
            "batch_index": batch_index,
            "restart_seed": restart_seed,
            "seconds": time.perf_counter() - started,
            "error_type": type(exc).__name__,
            "error_message_sha256": hashlib.sha256(
                message.encode("utf-8")
            ).hexdigest(),
        }


def _metric_series(
    selected: list[dict[str, object]], section: str, metric: str
) -> list[float]:
    result: list[float] = []
    for attempt in selected:
        payload = attempt.get(section)
        if isinstance(payload, dict) and metric in payload:
            result.append(float(payload[metric]))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=tuple(ARCHITECTURES), required=True)
    args = parser.parse_args()
    architecture = str(args.architecture)
    torch.set_num_threads(max(1, min(2, torch.get_num_threads())))

    observed_file_sha256 = _file_sha256(DATASET_PATH)
    if observed_file_sha256 != DATASET_SHA256:
        raise RuntimeError(
            f"UCI CSV SHA256 mismatch: expected {DATASET_SHA256}, "
            f"observed {observed_file_sha256}"
        )

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
        "epochs": 2,
        "batch_size": 8,
        "optimizer": "adamw",
        "learning_rate": 1e-3,
        "weight_decay": 1e-3,
    }
    victim_config = dict(ARCHITECTURES[architecture])
    victim_seed = ARCHITECTURE_SEEDS[architecture]

    _seed_everything(victim_seed)
    dataset, task, mode = _load_dataset(
        {"seed": victim_seed, "dataset": dataset_config}
    )
    if task != "forecasting" or mode != "timeseries":
        raise RuntimeError("multivariate reconstruction requires forecasting data")
    inputs, targets = dataset.tensors
    model = _build_model(dataset, task, victim_config)
    training_started = time.perf_counter()
    _train(model, dataset, task, training_config)
    training_seconds = time.perf_counter() - training_started

    attempts: list[dict[str, object]] = []
    selected: list[dict[str, object]] = []
    selected_attempt_indices: list[int] = []
    for batch_index in ATTACK_INDICES:
        true_x = inputs[batch_index : batch_index + 1].clone()
        true_target = targets[batch_index : batch_index + 1].clone()
        group_indices: list[int] = []
        for restart_seed in RESTART_SEEDS:
            attempt = _run_attempt(
                model=model,
                true_x=true_x,
                true_target=true_target,
                batch_index=batch_index,
                restart_seed=restart_seed,
            )
            group_indices.append(len(attempts))
            attempts.append(attempt)
        successful = [
            index
            for index in group_indices
            if attempts[index]["status"] == "success"
            and attempts[index].get("best_objective") is not None
        ]
        if successful:
            chosen = min(
                successful,
                key=lambda index: (
                    float(attempts[index]["best_objective"]),
                    float(attempts[index]["best_gradient_match"]),
                    int(attempts[index]["restart_seed"]),
                ),
            )
            selected_attempt_indices.append(chosen)
            selected.append(attempts[chosen])

    failed_attempts = sum(item["status"] != "success" for item in attempts)
    ordered_mse = _metric_series(selected, "ordered_complete_metrics", "mse")
    aligned_mse = _metric_series(selected, "aligned_complete_metrics", "mse")
    ordered_relative = _metric_series(
        selected, "ordered_complete_metrics", "relative_l2_error"
    )
    aligned_relative = _metric_series(
        selected, "aligned_complete_metrics", "relative_l2_error"
    )
    ordered_correlation = _metric_series(
        selected, "ordered_complete_metrics", "correlation"
    )
    aligned_correlation = _metric_series(
        selected, "aligned_complete_metrics", "correlation"
    )
    alignment_improved = sum(
        aligned + 1e-12 < ordered for ordered, aligned in zip(ordered_mse, aligned_mse)
    )
    nonidentity_assignments = sum(
        not bool(item["channel_alignment"]["identity_assignment"])
        for item in selected
        if isinstance(item.get("channel_alignment"), dict)
    )
    ordered_exact = sum(
        bool(item["ordered_exact_within_tolerance"]) for item in selected
    )
    aligned_exact = sum(
        bool(item["aligned_exact_within_tolerance"]) for item in selected
    )
    ordered_relative_success = sum(
        value <= RELATIVE_L2_THRESHOLD for value in ordered_relative
    )
    aligned_relative_success = sum(
        value <= RELATIVE_L2_THRESHOLD for value in aligned_relative
    )

    selected_count = len(selected)
    completed = len(attempts) - failed_attempts
    quality_gate_passed = (
        observed_file_sha256 == DATASET_SHA256
        and len(ATTACK_INDICES) == 20
        and len(RESTART_SEEDS) == 3
        and len(attempts) == 60
        and failed_attempts == 0
        and selected_count == 20
    )
    payload = {
        "schema_version": "qrecon.uci-multivariate-reconstruction.v1",
        "dataset": dataset_config,
        "dataset_file_sha256": observed_file_sha256,
        "dataset_tensor_sha256": _tensor_sha256(inputs, targets),
        "victim": victim_config,
        "victim_class": type(model).__name__,
        "victim_seed": victim_seed,
        "model_sha256": _model_sha256(model),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "training": training_config,
        "training_seconds": training_seconds,
        "attack_contract": {
            "indices": list(ATTACK_INDICES),
            "restart_seeds": list(RESTART_SEEDS),
            "known_target": False,
            "selection_rule": "minimum released-gradient objective",
            "steps": 25,
            "match_mode": "hybrid",
            "layer_weighting": "parameter",
            "exact_tolerance": EXACT_TOLERANCE,
            "relative_l2_threshold": RELATIVE_L2_THRESHOLD,
            "evaluation_channel_assignment_uses_private_reference": True,
        },
        "attempts": attempts,
        "selected_attempt_indices": selected_attempt_indices,
        "summary": {
            "restart_completion": summarize_proportion(
                completed, len(attempts), confidence_level=0.95
            ).to_dict(),
            "selected_batches": selected_count,
            "ordered_exact_within_0p1": summarize_proportion(
                ordered_exact, max(1, len(ATTACK_INDICES)), confidence_level=0.95
            ).to_dict(),
            "channel_aligned_exact_within_0p1": summarize_proportion(
                aligned_exact, max(1, len(ATTACK_INDICES)), confidence_level=0.95
            ).to_dict(),
            "ordered_relative_l2_le_0p5": summarize_proportion(
                ordered_relative_success,
                max(1, len(ATTACK_INDICES)),
                confidence_level=0.95,
            ).to_dict(),
            "channel_aligned_relative_l2_le_0p5": summarize_proportion(
                aligned_relative_success,
                max(1, len(ATTACK_INDICES)),
                confidence_level=0.95,
            ).to_dict(),
            "alignment_improved_mse": summarize_proportion(
                alignment_improved,
                max(1, selected_count),
                confidence_level=0.95,
            ).to_dict(),
            "selected_nonidentity_channel_assignment": summarize_proportion(
                nonidentity_assignments,
                max(1, selected_count),
                confidence_level=0.95,
            ).to_dict(),
            "ordered_mse": _scalar(ordered_mse, f"{architecture}:ordered-mse"),
            "channel_aligned_mse": _scalar(
                aligned_mse, f"{architecture}:aligned-mse"
            ),
            "ordered_relative_l2": _scalar(
                ordered_relative, f"{architecture}:ordered-relative"
            ),
            "channel_aligned_relative_l2": _scalar(
                aligned_relative, f"{architecture}:aligned-relative"
            ),
            "ordered_correlation": _scalar(
                ordered_correlation, f"{architecture}:ordered-correlation"
            ),
            "channel_aligned_correlation": _scalar(
                aligned_correlation, f"{architecture}:aligned-correlation"
            ),
        },
        "quality_gate": {
            "immutable_non_ett_source": True,
            "modern_anonymous_channel_architecture": True,
            "twenty_private_records": len(ATTACK_INDICES) == 20,
            "three_restarts_per_record": len(RESTART_SEEDS) == 3,
            "all_sixty_attempts_completed": failed_attempts == 0,
            "every_record_has_selected_attempt": selected_count == 20,
            "passed": quality_gate_passed,
        },
        "environment": benchmark_environment_manifest(),
        "interpretation": (
            "Ordered and channel-aligned scores answer different recovery targets. The "
            "attack is selected only by the released-gradient objective. The private "
            "reference is used after selection solely to evaluate the best global "
            "channel assignment."
        ),
        "claim_boundary": (
            "A lower channel-aligned error demonstrates numerical content leakage modulo "
            "the anonymous-channel orbit; it does not recover semantic channel labels. "
            "Poor optimization does not strengthen the exact information-theoretic "
            "ceiling, and this classical white-box experiment does not imply quantum "
            "advantage or a coherent compiler for the complete Transformer stack."
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not quality_gate_passed:
        raise SystemExit("UCI multivariate reconstruction quality gate failed")


if __name__ == "__main__":
    main()
