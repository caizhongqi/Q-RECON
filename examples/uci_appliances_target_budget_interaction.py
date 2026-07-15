from __future__ import annotations

import argparse
import hashlib
import json
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
from qrecon.metrics import permutation_invariant_channel_metrics, reconstruction_metrics


DATASET_PATH = "data/UCI-appliances/energydata_complete.csv"
DATASET_SHA256 = "2820bf712ad0275cb18b85a05250926100d8e65ebb9f4d2d016ca91ea152a25d"
CHANNELS = ("T1", "T2", "T3", "T4", "T5", "T6", "T7")
ATTACK_INDICES = tuple(range(40, 50))
RESTART_SEEDS = (101, 103, 107)
CONDITIONS = (
    ("private_400", "private", 400),
    ("public_60", "public_ordered", 60),
    ("public_400", "public_ordered", 400),
)
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
        label=f"uci-target-budget:{label}",
    ).to_dict()


def _run_attempt(
    *,
    model: torch.nn.Module,
    true_x: torch.Tensor,
    true_target: torch.Tensor,
    batch_index: int,
    restart_seed: int,
    condition: str,
    target_visibility: str,
    steps: int,
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
        known_target = true_target if target_visibility == "public_ordered" else None
        attack = GradientInversionAttack(
            model=model,
            observed_gradients=observed,
            prior=prior,
            task="forecasting",
            mode="timeseries",
            known_target=known_target,
            target_shape=tuple(true_target.shape),
            steps=steps,
            learning_rate=0.04,
            regularization=1e-4,
            optimizer_name="adam",
            match_mode="hybrid",
            layer_weighting="parameter",
            gradient_clip_norm=10.0,
            record_every=steps,
        )
        result = attack.run()
        reconstructed_x = result.reconstruction.detach()
        reconstructed_target = (
            true_target.detach()
            if target_visibility == "public_ordered"
            else result.reconstructed_target.detach()
        )
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
        return {
            "status": "success",
            "condition": condition,
            "target_visibility": target_visibility,
            "steps": steps,
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
            "channel_alignment": channel_alignment.to_dict(),
            "ordered_input_exact_within_tolerance": bool(
                (true_x - reconstructed_x).abs().max() <= EXACT_TOLERANCE
            ),
            "aligned_input_exact_within_tolerance": bool(
                (true_x - aligned_x).abs().max() <= EXACT_TOLERANCE
            ),
        }
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        return {
            "status": "failed",
            "condition": condition,
            "target_visibility": target_visibility,
            "steps": steps,
            "batch_index": batch_index,
            "restart_seed": restart_seed,
            "seconds": time.perf_counter() - started,
            "error_type": type(exc).__name__,
            "error_message_sha256": hashlib.sha256(message.encode("utf-8")).hexdigest(),
        }


def _select(attempts: list[dict[str, object]], indices: list[int]) -> int | None:
    successful = [
        index
        for index in indices
        if attempts[index]["status"] == "success"
        and attempts[index].get("best_objective") is not None
    ]
    if not successful:
        return None
    return min(
        successful,
        key=lambda index: (
            float(attempts[index]["best_objective"]),
            float(attempts[index]["best_gradient_match"]),
            int(attempts[index]["restart_seed"]),
        ),
    )


def _metric_series(
    selected: list[dict[str, object]], section: str, metric: str
) -> list[float]:
    values: list[float] = []
    for item in selected:
        payload = item.get(section)
        if isinstance(payload, dict) and metric in payload:
            values.append(float(payload[metric]))
    return values


def _condition_summary(
    architecture: str,
    condition: str,
    selected: list[dict[str, object]],
) -> dict[str, object]:
    ordered_mse = _metric_series(selected, "ordered_input_metrics", "mse")
    aligned_mse = _metric_series(selected, "aligned_input_metrics", "mse")
    ordered_relative = _metric_series(
        selected, "ordered_input_metrics", "relative_l2_error"
    )
    aligned_relative = _metric_series(
        selected, "aligned_input_metrics", "relative_l2_error"
    )
    ordered_correlation = _metric_series(
        selected, "ordered_input_metrics", "correlation"
    )
    aligned_correlation = _metric_series(
        selected, "aligned_input_metrics", "correlation"
    )
    exact_ordered = sum(
        bool(item["ordered_input_exact_within_tolerance"]) for item in selected
    )
    exact_aligned = sum(
        bool(item["aligned_input_exact_within_tolerance"]) for item in selected
    )
    rel_ordered = sum(value <= RELATIVE_L2_THRESHOLD for value in ordered_relative)
    rel_aligned = sum(value <= RELATIVE_L2_THRESHOLD for value in aligned_relative)
    alignment_improved = sum(
        aligned + 1e-12 < ordered for ordered, aligned in zip(ordered_mse, aligned_mse)
    )
    nonidentity = sum(
        not bool(item["channel_alignment"]["identity_assignment"])
        for item in selected
    )
    count = len(selected)
    return {
        "condition": condition,
        "target_visibility": selected[0]["target_visibility"] if selected else None,
        "steps": selected[0]["steps"] if selected else None,
        "selected_records": count,
        "ordered_exact_within_0p1": summarize_proportion(
            exact_ordered, max(1, count), confidence_level=0.95
        ).to_dict(),
        "channel_aligned_exact_within_0p1": summarize_proportion(
            exact_aligned, max(1, count), confidence_level=0.95
        ).to_dict(),
        "ordered_relative_l2_le_0p5": summarize_proportion(
            rel_ordered, max(1, count), confidence_level=0.95
        ).to_dict(),
        "channel_aligned_relative_l2_le_0p5": summarize_proportion(
            rel_aligned, max(1, count), confidence_level=0.95
        ).to_dict(),
        "alignment_improved_mse": summarize_proportion(
            alignment_improved, max(1, count), confidence_level=0.95
        ).to_dict(),
        "selected_nonidentity_channel_assignment": summarize_proportion(
            nonidentity, max(1, count), confidence_level=0.95
        ).to_dict(),
        "ordered_input_mse": _scalar(
            ordered_mse, f"{architecture}:{condition}:ordered-mse"
        ),
        "channel_aligned_input_mse": _scalar(
            aligned_mse, f"{architecture}:{condition}:aligned-mse"
        ),
        "ordered_input_relative_l2": _scalar(
            ordered_relative, f"{architecture}:{condition}:ordered-relative"
        ),
        "channel_aligned_input_relative_l2": _scalar(
            aligned_relative, f"{architecture}:{condition}:aligned-relative"
        ),
        "ordered_input_correlation": _scalar(
            ordered_correlation, f"{architecture}:{condition}:ordered-correlation"
        ),
        "channel_aligned_input_correlation": _scalar(
            aligned_correlation, f"{architecture}:{condition}:aligned-correlation"
        ),
        "best_objective": _scalar(
            [float(item["best_objective"]) for item in selected],
            f"{architecture}:{condition}:best-objective",
        ),
    }


def _paired_delta(
    architecture: str,
    left_condition: str,
    right_condition: str,
    selected_by_condition: dict[str, list[dict[str, object]]],
) -> dict[str, object]:
    left = {
        int(item["batch_index"]): item
        for item in selected_by_condition[left_condition]
    }
    right = {
        int(item["batch_index"]): item
        for item in selected_by_condition[right_condition]
    }
    records = sorted(set(left) & set(right))
    aligned_delta: list[float] = []
    ordered_delta: list[float] = []
    objective_delta: list[float] = []
    aligned_improves = 0
    ordered_improves = 0
    for record in records:
        left_item = left[record]
        right_item = right[record]
        left_aligned = float(left_item["aligned_input_metrics"]["mse"])
        right_aligned = float(right_item["aligned_input_metrics"]["mse"])
        left_ordered = float(left_item["ordered_input_metrics"]["mse"])
        right_ordered = float(right_item["ordered_input_metrics"]["mse"])
        aligned_delta.append(right_aligned - left_aligned)
        ordered_delta.append(right_ordered - left_ordered)
        objective_delta.append(
            float(right_item["best_objective"])
            - float(left_item["best_objective"])
        )
        aligned_improves += right_aligned < left_aligned
        ordered_improves += right_ordered < left_ordered
    label = f"{architecture}:{right_condition}-minus-{left_condition}"
    return {
        "left_condition": left_condition,
        "right_condition": right_condition,
        "paired_records": records,
        "aligned_input_mse_delta": _scalar(
            aligned_delta, f"{label}:aligned-mse"
        ),
        "ordered_input_mse_delta": _scalar(
            ordered_delta, f"{label}:ordered-mse"
        ),
        "best_objective_delta": _scalar(
            objective_delta, f"{label}:objective"
        ),
        "aligned_mse_improves": summarize_proportion(
            aligned_improves, max(1, len(records)), confidence_level=0.95
        ).to_dict(),
        "ordered_mse_improves": summarize_proportion(
            ordered_improves, max(1, len(records)), confidence_level=0.95
        ).to_dict(),
    }


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
        raise RuntimeError("target-budget interaction requires forecasting data")
    inputs, targets = dataset.tensors
    model = _build_model(dataset, task, victim_config)
    training_started = time.perf_counter()
    _train(model, dataset, task, training_config)
    training_seconds = time.perf_counter() - training_started

    attempts: list[dict[str, object]] = []
    selected_by_condition: dict[str, list[dict[str, object]]] = {
        condition: [] for condition, _, _ in CONDITIONS
    }
    selected_indices: dict[str, list[int]] = {
        condition: [] for condition, _, _ in CONDITIONS
    }
    selection_complete = True

    for condition, target_visibility, steps in CONDITIONS:
        for batch_index in ATTACK_INDICES:
            true_x = inputs[batch_index : batch_index + 1].clone()
            true_target = targets[batch_index : batch_index + 1].clone()
            group: list[int] = []
            for restart_seed in RESTART_SEEDS:
                group.append(len(attempts))
                attempts.append(
                    _run_attempt(
                        model=model,
                        true_x=true_x,
                        true_target=true_target,
                        batch_index=batch_index,
                        restart_seed=restart_seed,
                        condition=condition,
                        target_visibility=target_visibility,
                        steps=steps,
                    )
                )
            chosen = _select(attempts, group)
            if chosen is None:
                selection_complete = False
            else:
                selected_indices[condition].append(chosen)
                selected_by_condition[condition].append(attempts[chosen])

    failed_attempts = sum(item["status"] != "success" for item in attempts)
    summaries = {
        condition: _condition_summary(
            architecture, condition, selected_by_condition[condition]
        )
        for condition, _, _ in CONDITIONS
    }
    paired = {
        "public_400_minus_private_400": _paired_delta(
            architecture,
            "private_400",
            "public_400",
            selected_by_condition,
        ),
        "public_400_minus_public_60": _paired_delta(
            architecture,
            "public_60",
            "public_400",
            selected_by_condition,
        ),
    }

    expected_attempts = len(ATTACK_INDICES) * len(RESTART_SEEDS) * len(CONDITIONS)
    all_condition_groups_complete = all(
        len(selected_by_condition[condition]) == len(ATTACK_INDICES)
        for condition, _, _ in CONDITIONS
    )
    quality_gate_passed = (
        observed_file_sha256 == DATASET_SHA256
        and len(ATTACK_INDICES) == 10
        and len(RESTART_SEEDS) == 3
        and CONDITIONS
        == (
            ("private_400", "private", 400),
            ("public_60", "public_ordered", 60),
            ("public_400", "public_ordered", 400),
        )
        and len(attempts) == expected_attempts
        and failed_attempts == 0
        and selection_complete
        and all_condition_groups_complete
    )

    payload = {
        "schema_version": "qrecon.uci-target-budget-interaction.v1",
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
            "conditions": [
                {
                    "name": condition,
                    "target_visibility": target_visibility,
                    "steps": steps,
                }
                for condition, target_visibility, steps in CONDITIONS
            ],
            "selection_rule": "minimum released-gradient objective within condition",
            "match_mode": "hybrid",
            "layer_weighting": "parameter",
            "exact_tolerance": EXACT_TOLERANCE,
            "relative_l2_threshold": RELATIVE_L2_THRESHOLD,
            "evaluation_channel_assignment_uses_private_reference": True,
        },
        "attempts": attempts,
        "selected_attempt_indices": selected_indices,
        "condition_summaries": summaries,
        "paired_interactions": paired,
        "quality_gate": {
            "immutable_non_ett_source": True,
            "modern_anonymous_channel_architecture": True,
            "ten_paired_private_records": len(ATTACK_INDICES) == 10,
            "three_restarts_per_condition": len(RESTART_SEEDS) == 3,
            "predeclared_private400_public60_public400": CONDITIONS
            == (
                ("private_400", "private", 400),
                ("public_60", "public_ordered", 60),
                ("public_400", "public_ordered", 400),
            ),
            "all_ninety_attempts_completed": len(attempts) == 90
            and failed_attempts == 0,
            "every_record_selected_in_every_condition": selection_complete
            and all_condition_groups_complete,
            "passed": quality_gate_passed,
        },
        "environment": benchmark_environment_manifest(),
        "interpretation": (
            "The interaction matrix separates two interventions: supplying the future "
            "target tensor in semantic channel order and increasing optimization from 60 "
            "to 400 steps. Dataset, trained model, private records, restart seeds, attack "
            "objective and selection rule remain fixed."
        ),
        "claim_boundary": (
            "Public ordered targets change the threat model. The convergence results are "
            "optimizer evidence rather than an information-theoretic theorem. Private "
            "references are used only after restart selection for evaluation-only channel "
            "alignment. No coherent Transformer compiler or quantum advantage is claimed."
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not quality_gate_passed:
        raise SystemExit("UCI target-budget interaction quality gate failed")


if __name__ == "__main__":
    main()
