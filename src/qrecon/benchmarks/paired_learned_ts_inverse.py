from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass

import torch
from torch.utils.data import TensorDataset

from qrecon.attacks import (
    TSInverseStyleAttack,
    flatten_gradient_tuple,
    initialize_direct_prior_from_median,
    leak_gradients,
    predict_gradient_quantiles,
    train_gradient_quantile_network,
)
from qrecon.benchmarks.learned_ts_inverse import LearnedQuantileAuxiliaryConfig
from qrecon.benchmarks.modern_timeseries_reconstruction import (
    MODERN_FORECASTING_ARCHITECTURES,
    ModernTimeSeriesAttackManifest,
    _model_sha256,
    _tensor_sha256,
)
from qrecon.benchmarks.statistics import (
    benchmark_environment_manifest,
    summarize_proportion,
    summarize_scalar,
)
from qrecon.experiment import _build_model, _load_dataset, _prior, _seed_everything, _train
from qrecon.metrics import permutation_invariant_batch_metrics, reconstruction_metrics


@dataclass(frozen=True)
class PairedInitializerAttempt:
    batch_start: int
    restart_seed: int
    method: str
    status: str
    seconds: float
    best_objective: float | None
    best_gradient_l1: float | None
    best_step: int | None
    aligned_batch: dict[str, object] | None
    error_type: str | None = None
    error_message_sha256: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PairedLearnedQualityGate:
    architecture_is_modern: bool
    real_dataset: bool
    split_disjoint: bool
    attacked_samples_are_victim_training_data: bool
    enough_auxiliary_samples: bool
    enough_attack_batches: bool
    enough_attack_seeds: bool
    every_method_has_successful_attempt_per_batch: bool
    no_failed_attempts: bool
    publication_mode: bool
    passed: bool

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)


@dataclass(frozen=True)
class PairedLearnedTSInverseReport:
    manifest: ModernTimeSeriesAttackManifest
    auxiliary: LearnedQuantileAuxiliaryConfig
    report_sha256: str
    dataset_sha256: str
    model_sha256: str
    victim_class: str
    trainable_parameters: int
    victim_training_seconds: float
    auxiliary_gradient_seconds: float
    initializer_training: dict[str, object]
    initializer_summary: dict[str, object]
    attempts: tuple[PairedInitializerAttempt, ...]
    selected_attempt_indices: dict[str, tuple[int, ...]]
    method_summaries: dict[str, object]
    paired_summary: dict[str, object]
    quality_gate: PairedLearnedQualityGate
    environment: dict[str, object]
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest": self.manifest.to_dict(),
            "auxiliary": self.auxiliary.to_dict(),
            "report_sha256": self.report_sha256,
            "dataset_sha256": self.dataset_sha256,
            "model_sha256": self.model_sha256,
            "victim_class": self.victim_class,
            "trainable_parameters": self.trainable_parameters,
            "victim_training_seconds": self.victim_training_seconds,
            "auxiliary_gradient_seconds": self.auxiliary_gradient_seconds,
            "initializer_training": self.initializer_training,
            "initializer_summary": self.initializer_summary,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "selected_attempt_indices": {
                method: list(indices)
                for method, indices in sorted(self.selected_attempt_indices.items())
            },
            "method_summaries": self.method_summaries,
            "paired_summary": self.paired_summary,
            "quality_gate": self.quality_gate.to_dict(),
            "environment": self.environment,
            "claim_boundary": self.claim_boundary,
        }


def _hash_declaration(
    manifest: ModernTimeSeriesAttackManifest,
    auxiliary: LearnedQuantileAuxiliaryConfig,
) -> str:
    payload = {
        "schema_version": "qrecon.paired-learned-ts-inverse.v1",
        "manifest": manifest.to_dict(),
        "auxiliary": auxiliary.to_dict(),
        "methods": ("random_l1", "learned_quantile_l1"),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _scalar(
    values: list[float],
    manifest: ModernTimeSeriesAttackManifest,
    label: str,
) -> dict[str, object] | None:
    if not values:
        return None
    return summarize_scalar(
        values,
        confidence_level=manifest.confidence_level,
        bootstrap_samples=manifest.bootstrap_samples,
        bootstrap_seed=manifest.bootstrap_seed,
        label=f"paired-learned:{label}",
    ).to_dict()


def _coverage(reference: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> float:
    if reference.shape != lower.shape or reference.shape != upper.shape:
        raise ValueError("coverage tensors must have identical shapes")
    return float(((reference >= lower) & (reference <= upper)).float().mean())


def _attack_kwargs(
    attack_config: dict[str, object],
    *,
    quantile_bound_weight: float,
    input_lower: torch.Tensor | None = None,
    input_upper: torch.Tensor | None = None,
    target_lower: torch.Tensor | None = None,
    target_upper: torch.Tensor | None = None,
) -> dict[str, object]:
    return {
        "steps": int(attack_config.get("steps", 100)),
        "learning_rate": float(attack_config.get("learning_rate", 0.03)),
        "optimizer_name": str(attack_config.get("optimizer", "adam")),
        "gradient_l1_weight": float(attack_config.get("gradient_l1_weight", 1.0)),
        "input_total_variation_weight": float(
            attack_config.get("input_total_variation_weight", 0.0)
        ),
        "target_total_variation_weight": float(
            attack_config.get("target_total_variation_weight", 0.0)
        ),
        "trend_weight": float(attack_config.get("trend_weight", 0.0)),
        "trend_loss": str(attack_config.get("trend_loss", "l1")),
        "trend_detach": bool(attack_config.get("trend_detach", True)),
        "periodicity_weight": float(attack_config.get("periodicity_weight", 0.0)),
        "periodicity_period": (
            None
            if attack_config.get("periodicity_period") is None
            else int(attack_config["periodicity_period"])
        ),
        "periodicity_loss": str(attack_config.get("periodicity_loss", "l1")),
        "low_resolution_weight": float(
            attack_config.get("low_resolution_weight", 0.0)
        ),
        "low_resolution_factor": int(attack_config.get("low_resolution_factor", 2)),
        "low_resolution_loss": str(
            attack_config.get("low_resolution_loss", "l1")
        ),
        "quantile_bound_weight": float(quantile_bound_weight),
        "input_quantile_lower": input_lower,
        "input_quantile_upper": input_upper,
        "target_quantile_lower": target_lower,
        "target_quantile_upper": target_upper,
        "joint_channel": int(attack_config.get("joint_channel", 0)),
        "gradient_clip_norm": (
            None
            if attack_config.get("gradient_clip_norm") is None
            else float(attack_config["gradient_clip_norm"])
        ),
        "record_every": (
            None
            if attack_config.get("record_every") is None
            else int(attack_config["record_every"])
        ),
    }


def _run_one(
    *,
    method: str,
    model: torch.nn.Module,
    observed: tuple[torch.Tensor, ...],
    true_input: torch.Tensor,
    true_target: torch.Tensor,
    restart_seed: int,
    manifest: ModernTimeSeriesAttackManifest,
    prior: torch.nn.Module,
    quantile_bound_weight: float,
    input_lower: torch.Tensor | None = None,
    input_upper: torch.Tensor | None = None,
    target_lower: torch.Tensor | None = None,
    target_upper: torch.Tensor | None = None,
) -> PairedInitializerAttempt:
    started = time.perf_counter()
    try:
        attack = TSInverseStyleAttack(
            model,
            observed,
            prior,
            known_target=true_target,
            target_shape=tuple(true_target.shape),
            **_attack_kwargs(
                dict(manifest.attack),
                quantile_bound_weight=quantile_bound_weight,
                input_lower=input_lower,
                input_upper=input_upper,
                target_lower=target_lower,
                target_upper=target_upper,
            ),
        )
        result = attack.run()
        aligned = permutation_invariant_batch_metrics(
            true_input,
            result.reconstruction,
            mode="timeseries",
            tolerance=manifest.exact_tolerance,
        ).to_dict()
        return PairedInitializerAttempt(
            batch_start=0,
            restart_seed=restart_seed,
            method=method,
            status="success",
            seconds=time.perf_counter() - started,
            best_objective=result.best_objective,
            best_gradient_l1=result.best_gradient_match,
            best_step=result.best_step,
            aligned_batch=aligned,
        )
    except Exception as exc:  # failures remain visible and counted
        message = f"{type(exc).__name__}: {exc}"
        return PairedInitializerAttempt(
            batch_start=0,
            restart_seed=restart_seed,
            method=method,
            status="failed",
            seconds=time.perf_counter() - started,
            best_objective=None,
            best_gradient_l1=None,
            best_step=None,
            aligned_batch=None,
            error_type=type(exc).__name__,
            error_message_sha256=hashlib.sha256(message.encode("utf-8")).hexdigest(),
        )


def _method_summary(
    selected: list[PairedInitializerAttempt],
    manifest: ModernTimeSeriesAttackManifest,
) -> dict[str, object]:
    exact = 0
    relative = 0
    values: dict[str, list[float]] = {}
    for attempt in selected:
        assert attempt.aligned_batch is not None
        metrics = attempt.aligned_batch["aligned_metrics"]
        assert isinstance(metrics, dict)
        exact += int(bool(attempt.aligned_batch["exact_batch_within_tolerance"]))
        relative += int(
            float(metrics["relative_l2_error"]) <= manifest.relative_l2_threshold
        )
        for name, value in metrics.items():
            values.setdefault(str(name), []).append(float(value))
        values.setdefault("best_objective", []).append(float(attempt.best_objective))
        values.setdefault("best_gradient_l1", []).append(
            float(attempt.best_gradient_l1)
        )
        values.setdefault("attack_seconds", []).append(float(attempt.seconds))
    declared = len(manifest.attack_indices)
    return {
        "selected_successful_batches": len(selected),
        "exact_batch_success": summarize_proportion(
            exact, declared, confidence_level=manifest.confidence_level
        ).to_dict(),
        "relative_l2_success": summarize_proportion(
            relative, declared, confidence_level=manifest.confidence_level
        ).to_dict(),
        "scalar_summaries": {
            name: _scalar(sample, manifest, name)
            for name, sample in sorted(values.items())
        },
    }


def run_paired_learned_ts_inverse_benchmark(
    manifest: ModernTimeSeriesAttackManifest,
    auxiliary: LearnedQuantileAuxiliaryConfig,
) -> PairedLearnedTSInverseReport:
    if manifest.attack_batch_size != 1:
        raise ValueError("paired learned benchmark requires attack_batch_size=1")
    if not bool(manifest.attack.get("known_target", True)):
        raise ValueError("paired learned benchmark currently requires known_target=true")

    _seed_everything(manifest.victim_seed)
    dataset, task, mode = _load_dataset(
        {"seed": manifest.victim_seed, "dataset": dict(manifest.dataset)}
    )
    if task != "forecasting" or mode != "timeseries":
        raise ValueError("paired learned benchmark requires forecasting data")
    inputs, targets = dataset.tensors
    maximum = max(
        *manifest.attack_indices,
        *auxiliary.victim_training_indices,
        *auxiliary.auxiliary_indices,
    )
    if maximum >= len(inputs):
        raise ValueError(f"declared index {maximum} exceeds dataset size {len(inputs)}")
    attacked = set(manifest.attack_indices)
    victim_indices = set(auxiliary.victim_training_indices)
    if not attacked <= victim_indices:
        raise ValueError("every attacked sample must belong to victim training data")

    model = _build_model(dataset, task, dict(manifest.victim))
    victim_dataset = TensorDataset(
        inputs[list(auxiliary.victim_training_indices)].clone(),
        targets[list(auxiliary.victim_training_indices)].clone(),
    )
    started = time.perf_counter()
    _train(model, victim_dataset, task, dict(manifest.training))
    victim_training_seconds = time.perf_counter() - started

    gradient_features: list[torch.Tensor] = []
    auxiliary_inputs: list[torch.Tensor] = []
    auxiliary_targets: list[torch.Tensor] = []
    started = time.perf_counter()
    for index in auxiliary.auxiliary_indices:
        sample_input = inputs[index : index + 1].clone()
        sample_target = targets[index : index + 1].clone()
        gradients = leak_gradients(model, sample_input, sample_target, "forecasting")
        gradient_features.append(flatten_gradient_tuple(gradients).float())
        auxiliary_inputs.append(sample_input.squeeze(0).float())
        auxiliary_targets.append(sample_target.squeeze(0).float())
    auxiliary_gradient_seconds = time.perf_counter() - started

    quantile_model, normalizer, initializer_report = train_gradient_quantile_network(
        torch.stack(gradient_features),
        torch.stack(auxiliary_inputs),
        torch.stack(auxiliary_targets),
        hidden_sizes=auxiliary.hidden_sizes,
        dropout=auxiliary.dropout,
        quantiles=auxiliary.quantiles,
        validation_fraction=auxiliary.validation_fraction,
        epochs=auxiliary.epochs,
        batch_size=auxiliary.batch_size,
        learning_rate=auxiliary.learning_rate,
        weight_decay=auxiliary.weight_decay,
        crossing_weight=auxiliary.crossing_weight,
        seed=auxiliary.training_seed,
    )
    quantiles = tuple(initializer_report.quantiles)
    lower_index = 0
    upper_index = len(quantiles) - 1
    median_index = min(
        range(len(quantiles)), key=lambda index: (abs(quantiles[index] - 0.5), index)
    )

    attempts: list[PairedInitializerAttempt] = []
    initializer_mse: list[float] = []
    input_coverage: list[float] = []
    target_coverage: list[float] = []
    groups: dict[tuple[int, str], list[int]] = {}
    for batch_start in manifest.attack_indices:
        true_input = inputs[batch_start : batch_start + 1].clone()
        true_target = targets[batch_start : batch_start + 1].clone()
        observed = leak_gradients(model, true_input, true_target, "forecasting")
        input_quantiles, target_quantiles = predict_gradient_quantiles(
            quantile_model, normalizer, observed
        )
        if target_quantiles is None:
            raise RuntimeError("quantile model did not emit target quantiles")
        input_lower = input_quantiles[..., lower_index]
        input_median = input_quantiles[..., median_index]
        input_upper = input_quantiles[..., upper_index]
        target_lower = target_quantiles[..., lower_index]
        target_upper = target_quantiles[..., upper_index]
        initializer_mse.append(
            reconstruction_metrics(true_input, input_median, mode="timeseries")["mse"]
        )
        input_coverage.append(_coverage(true_input, input_lower, input_upper))
        target_coverage.append(_coverage(true_target, target_lower, target_upper))

        for restart_seed in manifest.attack_seeds:
            for method in ("random_l1", "learned_quantile_l1"):
                _seed_everything(restart_seed)
                if method == "random_l1":
                    prior = _prior(
                        tuple(true_input.shape), "timeseries", dict(manifest.attack)
                    )
                    quantile_weight = 0.0
                    bounds = (None, None, None, None)
                else:
                    prior = initialize_direct_prior_from_median(
                        input_median,
                        mode="timeseries",
                        bounded=bool(manifest.attack.get("bounded", True)),
                        jitter_standard_deviation=auxiliary.jitter_standard_deviation,
                        seed=restart_seed,
                    )
                    quantile_weight = float(
                        manifest.attack.get("quantile_bound_weight", 0.01)
                    )
                    bounds = (input_lower, input_upper, target_lower, target_upper)
                attempt = _run_one(
                    method=method,
                    model=model,
                    observed=observed,
                    true_input=true_input,
                    true_target=true_target,
                    restart_seed=restart_seed,
                    manifest=manifest,
                    prior=prior,
                    quantile_bound_weight=quantile_weight,
                    input_lower=bounds[0],
                    input_upper=bounds[1],
                    target_lower=bounds[2],
                    target_upper=bounds[3],
                )
                attempt = PairedInitializerAttempt(
                    batch_start=batch_start,
                    restart_seed=attempt.restart_seed,
                    method=attempt.method,
                    status=attempt.status,
                    seconds=attempt.seconds,
                    best_objective=attempt.best_objective,
                    best_gradient_l1=attempt.best_gradient_l1,
                    best_step=attempt.best_step,
                    aligned_batch=attempt.aligned_batch,
                    error_type=attempt.error_type,
                    error_message_sha256=attempt.error_message_sha256,
                )
                index = len(attempts)
                attempts.append(attempt)
                groups.setdefault((batch_start, method), []).append(index)

    selected_indices: dict[str, list[int]] = {
        "random_l1": [],
        "learned_quantile_l1": [],
    }
    selected: dict[str, list[PairedInitializerAttempt]] = {
        "random_l1": [],
        "learned_quantile_l1": [],
    }
    for batch_start in manifest.attack_indices:
        for method in selected:
            successful = [
                index
                for index in groups[(batch_start, method)]
                if attempts[index].status == "success"
                and attempts[index].best_objective is not None
            ]
            if successful:
                chosen = min(
                    successful,
                    key=lambda index: (
                        float(attempts[index].best_objective),
                        float(attempts[index].best_gradient_l1),
                        attempts[index].restart_seed,
                    ),
                )
                selected_indices[method].append(chosen)
                selected[method].append(attempts[chosen])

    method_summaries = {
        method: _method_summary(values, manifest)
        for method, values in selected.items()
    }
    paired_mse_differences: list[float] = []
    paired_relative_differences: list[float] = []
    learned_improved_mse = 0
    fully_paired = 0
    for batch_position, _ in enumerate(manifest.attack_indices):
        if batch_position >= len(selected["random_l1"]) or batch_position >= len(
            selected["learned_quantile_l1"]
        ):
            continue
        random_attempt = selected["random_l1"][batch_position]
        learned_attempt = selected["learned_quantile_l1"][batch_position]
        assert random_attempt.aligned_batch is not None
        assert learned_attempt.aligned_batch is not None
        random_metrics = random_attempt.aligned_batch["aligned_metrics"]
        learned_metrics = learned_attempt.aligned_batch["aligned_metrics"]
        assert isinstance(random_metrics, dict) and isinstance(learned_metrics, dict)
        mse_difference = float(learned_metrics["mse"]) - float(random_metrics["mse"])
        relative_difference = float(learned_metrics["relative_l2_error"]) - float(
            random_metrics["relative_l2_error"]
        )
        paired_mse_differences.append(mse_difference)
        paired_relative_differences.append(relative_difference)
        learned_improved_mse += int(mse_difference < 0.0)
        fully_paired += 1

    paired_summary: dict[str, object] = {
        "fully_paired_batches": fully_paired,
        "learned_initializer_improved_mse": (
            None
            if fully_paired == 0
            else summarize_proportion(
                learned_improved_mse,
                fully_paired,
                confidence_level=manifest.confidence_level,
            ).to_dict()
        ),
        "learned_minus_random_mse": _scalar(
            paired_mse_differences, manifest, "learned_minus_random_mse"
        ),
        "learned_minus_random_relative_l2": _scalar(
            paired_relative_differences, manifest, "learned_minus_random_relative_l2"
        ),
    }
    initializer_summary = {
        "input_mse": _scalar(initializer_mse, manifest, "initializer_mse"),
        "input_interval_coverage": _scalar(
            input_coverage, manifest, "input_interval_coverage"
        ),
        "target_interval_coverage": _scalar(
            target_coverage, manifest, "target_interval_coverage"
        ),
    }

    failures = sum(attempt.status != "success" for attempt in attempts)
    architecture = str(manifest.victim.get("architecture", "")).lower()
    dataset_name = str(manifest.dataset.get("name", "")).lower()
    architecture_is_modern = architecture in MODERN_FORECASTING_ARCHITECTURES
    real_dataset = not dataset_name.startswith("synthetic")
    split_disjoint = not (
        set(auxiliary.victim_training_indices) & set(auxiliary.auxiliary_indices)
    )
    attacks_are_training = attacked <= victim_indices
    enough_auxiliary = (
        len(auxiliary.auxiliary_indices)
        >= auxiliary.minimum_publication_auxiliary_samples
    )
    enough_batches = (
        len(manifest.attack_indices) >= manifest.minimum_publication_batches
    )
    enough_seeds = (
        len(manifest.attack_seeds) >= manifest.minimum_publication_attack_seeds
    )
    every_method = all(
        len(selected[method]) == len(manifest.attack_indices) for method in selected
    )
    no_failed = failures == 0
    passed = (
        manifest.publication_mode
        and architecture_is_modern
        and real_dataset
        and split_disjoint
        and attacks_are_training
        and enough_auxiliary
        and enough_batches
        and enough_seeds
        and every_method
        and no_failed
    )
    gate = PairedLearnedQualityGate(
        architecture_is_modern=architecture_is_modern,
        real_dataset=real_dataset,
        split_disjoint=split_disjoint,
        attacked_samples_are_victim_training_data=attacks_are_training,
        enough_auxiliary_samples=enough_auxiliary,
        enough_attack_batches=enough_batches,
        enough_attack_seeds=enough_seeds,
        every_method_has_successful_attempt_per_batch=every_method,
        no_failed_attempts=no_failed,
        publication_mode=manifest.publication_mode,
        passed=passed,
    )

    return PairedLearnedTSInverseReport(
        manifest=manifest,
        auxiliary=auxiliary,
        report_sha256=_hash_declaration(manifest, auxiliary),
        dataset_sha256=_tensor_sha256(inputs, targets),
        model_sha256=_model_sha256(model),
        victim_class=type(model).__name__,
        trainable_parameters=sum(parameter.numel() for parameter in model.parameters()),
        victim_training_seconds=victim_training_seconds,
        auxiliary_gradient_seconds=auxiliary_gradient_seconds,
        initializer_training=initializer_report.to_dict(),
        initializer_summary=initializer_summary,
        attempts=tuple(attempts),
        selected_attempt_indices={
            method: tuple(indices) for method, indices in selected_indices.items()
        },
        method_summaries=method_summaries,
        paired_summary=paired_summary,
        quality_gate=gate,
        environment=benchmark_environment_manifest(),
        claim_boundary=(
            "Random and learned initializers share the exact same victim, observations, "
            "attack batches, restart seeds and L1 optimization budget. The learned "
            "initializer is trained only on a disjoint auxiliary split. This known-"
            "target paired comparison does not reproduce unknown-target TS-Inverse and "
            "does not provide quantum-advantage evidence."
        ),
    )
