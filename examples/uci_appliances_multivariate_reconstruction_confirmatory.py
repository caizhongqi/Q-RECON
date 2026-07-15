from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path
from types import ModuleType

import torch

from qrecon.attacks import GradientInversionAttack, leak_gradients
from qrecon.benchmarks import (
    benchmark_environment_manifest,
    summarize_proportion,
    summarize_scalar,
)
from qrecon.experiment import _build_model, _load_dataset, _prior, _seed_everything, _train
from qrecon.metrics import permutation_invariant_channel_metrics, reconstruction_metrics


def _load_base_module() -> ModuleType:
    path = Path(__file__).with_name("uci_appliances_multivariate_reconstruction.py")
    spec = importlib.util.spec_from_file_location("qrecon_uci_reconstruction_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load base reconstruction module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = _load_base_module()
DATASET_PATH = BASE.DATASET_PATH
DATASET_SHA256 = BASE.DATASET_SHA256
CHANNELS = BASE.CHANNELS
ARCHITECTURES = BASE.ARCHITECTURES
ARCHITECTURE_SEEDS = {"itransformer": 307, "patchtst": 311}
ATTACK_INDICES = tuple(range(40, 50))
RESTART_SEEDS = (131, 137, 139)
EXACT_TOLERANCE = 0.1
RELATIVE_L2_THRESHOLD = 0.5

VARIANTS: tuple[dict[str, object], ...] = (
    {
        "name": "adam_hybrid_150",
        "steps": 150,
        "learning_rate": 0.03,
        "optimizer_name": "adam",
        "match_mode": "hybrid",
        "regularization": 1e-4,
    },
    {
        "name": "adam_cosine_150",
        "steps": 150,
        "learning_rate": 0.03,
        "optimizer_name": "adam",
        "match_mode": "cosine",
        "regularization": 0.0,
    },
    {
        "name": "adam_temporal_150",
        "steps": 150,
        "learning_rate": 0.03,
        "optimizer_name": "adam",
        "match_mode": "hybrid",
        "regularization": 1e-4,
        "trend_regularization": 1e-3,
        "trend_loss": "l1",
        "trend_detach": True,
        "periodicity_regularization": 1e-3,
        "periodicity_period": 4,
        "periodicity_loss": "l1",
        "low_resolution_regularization": 1e-3,
        "low_resolution_factor": 2,
        "low_resolution_loss": "l1",
    },
    {
        "name": "lbfgs_hybrid_75",
        "steps": 75,
        "learning_rate": 0.8,
        "optimizer_name": "lbfgs",
        "match_mode": "hybrid",
        "regularization": 1e-4,
    },
)


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
        label=f"uci-strong-reconstruction:{label}",
    ).to_dict()


def _run_attempt(
    *,
    model: torch.nn.Module,
    true_x: torch.Tensor,
    true_target: torch.Tensor,
    batch_index: int,
    restart_seed: int,
    variant: dict[str, object],
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
            steps=int(variant["steps"]),
            learning_rate=float(variant["learning_rate"]),
            regularization=float(variant.get("regularization", 0.0)),
            optimizer_name=str(variant["optimizer_name"]),
            match_mode=str(variant["match_mode"]),
            layer_weighting="parameter",
            gradient_clip_norm=10.0,
            record_every=max(1, int(variant["steps"]) // 10),
            trend_regularization=float(variant.get("trend_regularization", 0.0)),
            trend_loss=str(variant.get("trend_loss", "l1")),
            trend_detach=bool(variant.get("trend_detach", True)),
            periodicity_regularization=float(
                variant.get("periodicity_regularization", 0.0)
            ),
            periodicity_period=(
                None
                if variant.get("periodicity_period") is None
                else int(variant["periodicity_period"])
            ),
            periodicity_loss=str(variant.get("periodicity_loss", "l1")),
            low_resolution_regularization=float(
                variant.get("low_resolution_regularization", 0.0)
            ),
            low_resolution_factor=int(variant.get("low_resolution_factor", 2)),
            low_resolution_loss=str(variant.get("low_resolution_loss", "l1")),
        )
        result = attack.run()
        reconstructed_x = result.reconstruction.detach()
        reconstructed_target = result.reconstructed_target.detach()
        complete_reference = torch.cat((true_x, true_target), dim=1)
        complete_estimate = torch.cat((reconstructed_x, reconstructed_target), dim=1)
        alignment = permutation_invariant_channel_metrics(
            complete_reference,
            complete_estimate,
            tolerance=EXACT_TOLERANCE,
        )
        assignment = list(alignment.assignment)
        aligned_x = reconstructed_x[..., assignment]
        aligned_target = reconstructed_target[..., assignment]
        aligned_complete = torch.cat((aligned_x, aligned_target), dim=1)
        return {
            "status": "success",
            "variant": variant["name"],
            "batch_index": batch_index,
            "restart_seed": restart_seed,
            "seconds": time.perf_counter() - started,
            "best_objective": result.best_objective,
            "best_gradient_match": result.best_gradient_match,
            "best_step": result.best_step,
            "final_objective": result.final_objective,
            "final_gradient_match": result.final_gradient_match,
            "ordered_complete_metrics": reconstruction_metrics(
                complete_reference, complete_estimate, mode="timeseries"
            ),
            "aligned_complete_metrics": reconstruction_metrics(
                complete_reference, aligned_complete, mode="timeseries"
            ),
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
            "channel_alignment": alignment.to_dict(),
            "ordered_exact_within_tolerance": bool(
                (complete_reference - complete_estimate).abs().max()
                <= EXACT_TOLERANCE
            ),
            "aligned_exact_within_tolerance": bool(
                (complete_reference - aligned_complete).abs().max()
                <= EXACT_TOLERANCE
            ),
        }
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        return {
            "status": "failed",
            "variant": variant["name"],
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
    values: list[float] = []
    for attempt in selected:
        payload = attempt.get(section)
        if isinstance(payload, dict) and metric in payload:
            values.append(float(payload[metric]))
    return values


def _summarize_variant(
    attempts: list[dict[str, object]],
    selected: list[dict[str, object]],
    architecture: str,
    variant_name: str,
) -> dict[str, object]:
    failures = sum(item["status"] != "success" for item in attempts)
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
    exact_ordered = sum(
        bool(item["ordered_exact_within_tolerance"]) for item in selected
    )
    exact_aligned = sum(
        bool(item["aligned_exact_within_tolerance"]) for item in selected
    )
    rel_ordered = sum(value <= RELATIVE_L2_THRESHOLD for value in ordered_relative)
    rel_aligned = sum(value <= RELATIVE_L2_THRESHOLD for value in aligned_relative)
    alignment_improved = sum(
        aligned + 1e-12 < ordered for ordered, aligned in zip(ordered_mse, aligned_mse)
    )
    nonidentity = sum(
        not bool(item["channel_alignment"]["identity_assignment"])
        for item in selected
        if isinstance(item.get("channel_alignment"), dict)
    )
    label = f"{architecture}:{variant_name}"
    return {
        "total_restart_attempts": len(attempts),
        "failed_restart_attempts": failures,
        "selected_batches": len(selected),
        "restart_completion": summarize_proportion(
            len(attempts) - failures,
            len(attempts),
            confidence_level=0.95,
        ).to_dict(),
        "ordered_exact_within_0p1": summarize_proportion(
            exact_ordered, len(ATTACK_INDICES), confidence_level=0.95
        ).to_dict(),
        "channel_aligned_exact_within_0p1": summarize_proportion(
            exact_aligned, len(ATTACK_INDICES), confidence_level=0.95
        ).to_dict(),
        "ordered_relative_l2_le_0p5": summarize_proportion(
            rel_ordered, len(ATTACK_INDICES), confidence_level=0.95
        ).to_dict(),
        "channel_aligned_relative_l2_le_0p5": summarize_proportion(
            rel_aligned, len(ATTACK_INDICES), confidence_level=0.95
        ).to_dict(),
        "alignment_improved_mse": summarize_proportion(
            alignment_improved, max(1, len(selected)), confidence_level=0.95
        ).to_dict(),
        "selected_nonidentity_channel_assignment": summarize_proportion(
            nonidentity, max(1, len(selected)), confidence_level=0.95
        ).to_dict(),
        "ordered_mse": _scalar(ordered_mse, f"{label}:ordered-mse"),
        "channel_aligned_mse": _scalar(aligned_mse, f"{label}:aligned-mse"),
        "ordered_relative_l2": _scalar(
            ordered_relative, f"{label}:ordered-relative"
        ),
        "channel_aligned_relative_l2": _scalar(
            aligned_relative, f"{label}:aligned-relative"
        ),
        "ordered_correlation": _scalar(
            ordered_correlation, f"{label}:ordered-correlation"
        ),
        "channel_aligned_correlation": _scalar(
            aligned_correlation, f"{label}:aligned-correlation"
        ),
        "best_objective": _scalar(
            [float(item["best_objective"]) for item in selected],
            f"{label}:best-objective",
        ),
        "best_gradient_match": _scalar(
            [float(item["best_gradient_match"]) for item in selected],
            f"{label}:best-match",
        ),
        "selected_attack_seconds": _scalar(
            [float(item["seconds"]) for item in selected],
            f"{label}:seconds",
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=tuple(ARCHITECTURES), required=True)
    args = parser.parse_args()
    architecture = str(args.architecture)
    torch.set_num_threads(max(1, min(2, torch.get_num_threads())))

    observed_sha256 = _file_sha256(DATASET_PATH)
    if observed_sha256 != DATASET_SHA256:
        raise RuntimeError(
            f"UCI CSV SHA256 mismatch: expected {DATASET_SHA256}, "
            f"observed {observed_sha256}"
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
    victim_config = dict(ARCHITECTURES[architecture])
    victim_seed = ARCHITECTURE_SEEDS[architecture]
    training_config = {
        "epochs": 2,
        "batch_size": 8,
        "optimizer": "adamw",
        "learning_rate": 1e-3,
        "weight_decay": 1e-3,
    }
    _seed_everything(victim_seed)
    dataset, task, mode = _load_dataset(
        {"seed": victim_seed, "dataset": dataset_config}
    )
    if task != "forecasting" or mode != "timeseries":
        raise RuntimeError("confirmatory suite requires multivariate forecasting")
    inputs, targets = dataset.tensors
    model = _build_model(dataset, task, victim_config)
    training_started = time.perf_counter()
    _train(model, dataset, task, training_config)
    training_seconds = time.perf_counter() - training_started

    attempts: list[dict[str, object]] = []
    selected_indices: dict[str, list[int]] = {
        str(variant["name"]): [] for variant in VARIANTS
    }
    selected_attempts: dict[str, list[dict[str, object]]] = {
        str(variant["name"]): [] for variant in VARIANTS
    }
    variant_attempts: dict[str, list[dict[str, object]]] = {
        str(variant["name"]): [] for variant in VARIANTS
    }

    for batch_index in ATTACK_INDICES:
        true_x = inputs[batch_index : batch_index + 1].clone()
        true_target = targets[batch_index : batch_index + 1].clone()
        for variant in VARIANTS:
            variant_name = str(variant["name"])
            group_indices: list[int] = []
            for restart_seed in RESTART_SEEDS:
                attempt = _run_attempt(
                    model=model,
                    true_x=true_x,
                    true_target=true_target,
                    batch_index=batch_index,
                    restart_seed=restart_seed,
                    variant=variant,
                )
                group_indices.append(len(attempts))
                attempts.append(attempt)
                variant_attempts[variant_name].append(attempt)
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
                selected_indices[variant_name].append(chosen)
                selected_attempts[variant_name].append(attempts[chosen])

    summaries = {
        name: _summarize_variant(
            variant_attempts[name], selected_attempts[name], architecture, name
        )
        for name in sorted(variant_attempts)
    }
    no_failures = all(item["status"] == "success" for item in attempts)
    all_selected = all(
        len(selected_attempts[name]) == len(ATTACK_INDICES)
        for name in selected_attempts
    )
    quality_gate_passed = (
        observed_sha256 == DATASET_SHA256
        and len(VARIANTS) == 4
        and len(ATTACK_INDICES) == 10
        and len(RESTART_SEEDS) == 3
        and len(attempts) == 120
        and no_failures
        and all_selected
    )
    payload = {
        "schema_version": "qrecon.uci-strong-multivariate-reconstruction.v1",
        "dataset": dataset_config,
        "dataset_file_sha256": observed_sha256,
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
            "variants": list(VARIANTS),
            "known_target": False,
            "selection_rule": "minimum released-gradient objective within variant",
            "exact_tolerance": EXACT_TOLERANCE,
            "relative_l2_threshold": RELATIVE_L2_THRESHOLD,
            "evaluation_channel_assignment_uses_private_reference": True,
        },
        "attempts": attempts,
        "selected_attempt_indices": selected_indices,
        "variant_summaries": summaries,
        "quality_gate": {
            "immutable_non_ett_source": True,
            "modern_anonymous_channel_architecture": True,
            "four_predeclared_attack_variants": len(VARIANTS) == 4,
            "ten_private_records": len(ATTACK_INDICES) == 10,
            "three_restarts_per_variant_record": len(RESTART_SEEDS) == 3,
            "all_one_hundred_twenty_attempts_completed": no_failures,
            "every_variant_record_has_selected_attempt": all_selected,
            "passed": quality_gate_passed,
        },
        "environment": benchmark_environment_manifest(),
        "interpretation": (
            "The suite tests whether substantially stronger classical optimization "
            "recovers numerical content modulo a global channel permutation. Attempt "
            "selection uses only the released-gradient objective; private channel "
            "alignment is applied afterward for target-aware evaluation."
        ),
        "claim_boundary": (
            "Failure of every tested optimizer would not prove computational hardness, "
            "and success after channel alignment would not identify semantic labels. "
            "The exact orbit theorem remains information-theoretic and independent of "
            "these finite optimization runs."
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not quality_gate_passed:
        raise SystemExit("strong multivariate reconstruction quality gate failed")


if __name__ == "__main__":
    main()
