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
from qrecon.experiment import _build_model, _load_dataset, _seed_everything, _train
from qrecon.metrics import permutation_invariant_batch_metrics, reconstruction_metrics


@dataclass(frozen=True)
class LearnedQuantileAuxiliaryConfig:
    victim_training_indices: tuple[int, ...]
    auxiliary_indices: tuple[int, ...]
    hidden_sizes: tuple[int, ...] = (256, 128)
    dropout: float = 0.05
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    validation_fraction: float = 0.2
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    crossing_weight: float = 1e-2
    training_seed: int = 20260715
    jitter_standard_deviation: float = 0.02
    minimum_publication_auxiliary_samples: int = 64

    def __post_init__(self) -> None:
        if not self.victim_training_indices or not self.auxiliary_indices:
            raise ValueError("victim training and auxiliary indices must be non-empty")
        if any(index < 0 for index in self.victim_training_indices + self.auxiliary_indices):
            raise ValueError("dataset indices must be non-negative")
        if len(set(self.victim_training_indices)) != len(self.victim_training_indices):
            raise ValueError("victim training indices must be unique")
        if len(set(self.auxiliary_indices)) != len(self.auxiliary_indices):
            raise ValueError("auxiliary indices must be unique")
        if set(self.victim_training_indices) & set(self.auxiliary_indices):
            raise ValueError("auxiliary samples must be disjoint from victim training data")
        if not self.hidden_sizes or any(width <= 0 for width in self.hidden_sizes):
            raise ValueError("hidden_sizes must contain positive widths")
        if self.epochs <= 0 or self.batch_size <= 1:
            raise ValueError("epochs must be positive and batch_size must exceed one")
        if self.minimum_publication_auxiliary_samples <= 0:
            raise ValueError("minimum auxiliary sample count must be positive")
        if self.jitter_standard_deviation < 0.0 or not math.isfinite(
            float(self.jitter_standard_deviation)
        ):
            raise ValueError("jitter standard deviation must be finite and non-negative")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["victim_training_indices"] = list(self.victim_training_indices)
        payload["auxiliary_indices"] = list(self.auxiliary_indices)
        payload["hidden_sizes"] = list(self.hidden_sizes)
        payload["quantiles"] = list(self.quantiles)
        return payload


@dataclass(frozen=True)
class LearnedQuantileAttempt:
    batch_start: int
    restart_seed: int
    status: str
    seconds: float
    initializer_input_metrics: dict[str, float] | None
    initializer_target_metrics: dict[str, float] | None
    input_interval_coverage: float | None
    target_interval_coverage: float | None
    refined_batch: dict[str, object] | None
    best_objective: float | None
    best_gradient_l1: float | None
    best_step: int | None
    error_type: str | None = None
    error_message_sha256: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class LearnedQuantileQualityGate:
    architecture_is_modern: bool
    real_dataset: bool
    victim_and_auxiliary_splits_disjoint: bool
    attacked_samples_are_victim_training_data: bool
    enough_auxiliary_samples: bool
    enough_attack_batches: bool
    enough_attack_seeds: bool
    every_batch_has_successful_attempt: bool
    no_failed_attempts: bool
    publication_mode: bool
    learned_quantile_initializer_present: bool
    passed: bool

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)


@dataclass(frozen=True)
class LearnedQuantileBenchmarkReport:
    manifest: ModernTimeSeriesAttackManifest
    auxiliary: LearnedQuantileAuxiliaryConfig
    manifest_sha256: str
    dataset_sha256: str
    victim_model_sha256: str
    victim_class: str
    trainable_parameters: int
    victim_training_seconds: float
    auxiliary_gradient_seconds: float
    initializer_training: dict[str, object]
    attempts: tuple[LearnedQuantileAttempt, ...]
    selected_attempt_indices: tuple[int, ...]
    summary: dict[str, object]
    quality_gate: LearnedQuantileQualityGate
    environment: dict[str, object]
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest": self.manifest.to_dict(),
            "auxiliary": self.auxiliary.to_dict(),
            "manifest_sha256": self.manifest_sha256,
            "dataset_sha256": self.dataset_sha256,
            "victim_model_sha256": self.victim_model_sha256,
            "victim_class": self.victim_class,
            "trainable_parameters": self.trainable_parameters,
            "victim_training_seconds": self.victim_training_seconds,
            "auxiliary_gradient_seconds": self.auxiliary_gradient_seconds,
            "initializer_training": self.initializer_training,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "selected_attempt_indices": list(self.selected_attempt_indices),
            "summary": self.summary,
            "quality_gate": self.quality_gate.to_dict(),
            "environment": self.environment,
            "claim_boundary": self.claim_boundary,
        }


def _coverage(reference: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> float:
    if reference.shape != lower.shape or reference.shape != upper.shape:
        raise ValueError("coverage tensors must have identical shapes")
    return float(((reference >= lower) & (reference <= upper)).float().mean())


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
        label=f"learned-quantile:{label}",
    ).to_dict()


def _combined_sha256(
    manifest: ModernTimeSeriesAttackManifest,
    auxiliary: LearnedQuantileAuxiliaryConfig,
) -> str:
    payload = {
        "schema_version": "qrecon.learned-ts-inverse.v1",
        "manifest": manifest.to_dict(),
        "auxiliary": auxiliary.to_dict(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def run_learned_quantile_ts_inverse_benchmark(
    manifest: ModernTimeSeriesAttackManifest,
    auxiliary: LearnedQuantileAuxiliaryConfig,
) -> LearnedQuantileBenchmarkReport:
    """Train a disjoint auxiliary gradient-to-quantile initializer and refine it.

    This benchmark currently evaluates the known-target threat model used by the
    existing PatchTST reconstruction matrix. Input and target quantiles are both
    learned and audited, but the fixed true target is supplied during refinement.
    """

    if manifest.attack_batch_size != 1:
        raise ValueError("learned quantile benchmark currently requires attack_batch_size=1")
    if not bool(manifest.attack.get("known_target", True)):
        raise ValueError("learned quantile benchmark currently requires known_target=true")

    _seed_everything(manifest.victim_seed)
    dataset, task, mode = _load_dataset(
        {"seed": manifest.victim_seed, "dataset": dict(manifest.dataset)}
    )
    if task != "forecasting" or mode != "timeseries":
        raise ValueError("learned quantile benchmark requires forecasting data")
    inputs, targets = dataset.tensors
    maximum_index = max(
        *manifest.attack_indices,
        *auxiliary.victim_training_indices,
        *auxiliary.auxiliary_indices,
    )
    if maximum_index >= len(inputs):
        raise ValueError(
            f"declared index {maximum_index} exceeds dataset size {len(inputs)}"
        )
    attacked = set(manifest.attack_indices)
    victim_training = set(auxiliary.victim_training_indices)
    if not attacked <= victim_training:
        raise ValueError("every attacked sample must belong to the victim training split")

    model = _build_model(dataset, task, dict(manifest.victim))
    victim_dataset = TensorDataset(
        inputs[list(auxiliary.victim_training_indices)].clone(),
        targets[list(auxiliary.victim_training_indices)].clone(),
    )
    started = time.perf_counter()
    _train(model, victim_dataset, task, dict(manifest.training))
    victim_training_seconds = time.perf_counter() - started

    auxiliary_features: list[torch.Tensor] = []
    auxiliary_inputs: list[torch.Tensor] = []
    auxiliary_targets: list[torch.Tensor] = []
    started = time.perf_counter()
    for index in auxiliary.auxiliary_indices:
        sample_input = inputs[index : index + 1].clone()
        sample_target = targets[index : index + 1].clone()
        gradients = leak_gradients(model, sample_input, sample_target, "forecasting")
        auxiliary_features.append(flatten_gradient_tuple(gradients).float())
        auxiliary_inputs.append(sample_input.squeeze(0).float())
        auxiliary_targets.append(sample_target.squeeze(0).float())
    auxiliary_gradient_seconds = time.perf_counter() - started

    quantile_model, normalizer, initializer_report = train_gradient_quantile_network(
        torch.stack(auxiliary_features),
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

    quantile_values = tuple(initializer_report.quantiles)
    lower_index = 0
    upper_index = len(quantile_values) - 1
    median_index = min(
        range(len(quantile_values)),
        key=lambda index: (abs(quantile_values[index] - 0.5), index),
    )
    attack_config = dict(manifest.attack)
    attempts: list[LearnedQuantileAttempt] = []
    selected_indices: list[int] = []
    selected: list[LearnedQuantileAttempt] = []

    for batch_start in manifest.attack_indices:
        true_input = inputs[batch_start : batch_start + 1].clone()
        true_target = targets[batch_start : batch_start + 1].clone()
        observed = leak_gradients(model, true_input, true_target, "forecasting")
        input_quantiles, target_quantiles = predict_gradient_quantiles(
            quantile_model, normalizer, observed
        )
        if target_quantiles is None:
            raise RuntimeError("learned initializer did not emit target quantiles")
        input_lower = input_quantiles[..., lower_index]
        input_median = input_quantiles[..., median_index]
        input_upper = input_quantiles[..., upper_index]
        target_lower = target_quantiles[..., lower_index]
        target_median = target_quantiles[..., median_index]
        target_upper = target_quantiles[..., upper_index]
        initializer_input_metrics = reconstruction_metrics(
            true_input, input_median, mode="timeseries"
        )
        initializer_target_metrics = reconstruction_metrics(
            true_target, target_median, mode="timeseries"
        )
        input_coverage = _coverage(true_input, input_lower, input_upper)
        target_coverage = _coverage(true_target, target_lower, target_upper)
        group: list[int] = []

        for restart_seed in manifest.attack_seeds:
            started = time.perf_counter()
            try:
                _seed_everything(restart_seed)
                prior = initialize_direct_prior_from_median(
                    input_median,
                    mode="timeseries",
                    bounded=bool(attack_config.get("bounded", True)),
                    jitter_standard_deviation=auxiliary.jitter_standard_deviation,
                    seed=restart_seed,
                )
                attack = TSInverseStyleAttack(
                    model,
                    observed,
                    prior,
                    known_target=true_target,
                    target_shape=tuple(true_target.shape),
                    steps=int(attack_config.get("steps", 100)),
                    learning_rate=float(attack_config.get("learning_rate", 0.03)),
                    optimizer_name=str(attack_config.get("optimizer", "adam")),
                    gradient_l1_weight=float(
                        attack_config.get("gradient_l1_weight", 1.0)
                    ),
                    input_total_variation_weight=float(
                        attack_config.get("input_total_variation_weight", 0.0)
                    ),
                    target_total_variation_weight=float(
                        attack_config.get("target_total_variation_weight", 0.0)
                    ),
                    trend_weight=float(attack_config.get("trend_weight", 0.0)),
                    trend_loss=str(attack_config.get("trend_loss", "l1")),
                    trend_detach=bool(attack_config.get("trend_detach", True)),
                    periodicity_weight=float(
                        attack_config.get("periodicity_weight", 0.0)
                    ),
                    periodicity_period=(
                        None
                        if attack_config.get("periodicity_period") is None
                        else int(attack_config["periodicity_period"])
                    ),
                    periodicity_loss=str(
                        attack_config.get("periodicity_loss", "l1")
                    ),
                    low_resolution_weight=float(
                        attack_config.get("low_resolution_weight", 0.0)
                    ),
                    low_resolution_factor=int(
                        attack_config.get("low_resolution_factor", 2)
                    ),
                    low_resolution_loss=str(
                        attack_config.get("low_resolution_loss", "l1")
                    ),
                    quantile_bound_weight=float(
                        attack_config.get("quantile_bound_weight", 0.01)
                    ),
                    input_quantile_lower=input_lower,
                    input_quantile_upper=input_upper,
                    target_quantile_lower=target_lower,
                    target_quantile_upper=target_upper,
                    joint_channel=int(attack_config.get("joint_channel", 0)),
                    gradient_clip_norm=(
                        None
                        if attack_config.get("gradient_clip_norm") is None
                        else float(attack_config["gradient_clip_norm"])
                    ),
                    record_every=(
                        None
                        if attack_config.get("record_every") is None
                        else int(attack_config["record_every"])
                    ),
                )
                result = attack.run()
                refined = permutation_invariant_batch_metrics(
                    true_input,
                    result.reconstruction,
                    mode="timeseries",
                    tolerance=manifest.exact_tolerance,
                ).to_dict()
                attempt = LearnedQuantileAttempt(
                    batch_start=batch_start,
                    restart_seed=restart_seed,
                    status="success",
                    seconds=time.perf_counter() - started,
                    initializer_input_metrics=initializer_input_metrics,
                    initializer_target_metrics=initializer_target_metrics,
                    input_interval_coverage=input_coverage,
                    target_interval_coverage=target_coverage,
                    refined_batch=refined,
                    best_objective=result.best_objective,
                    best_gradient_l1=result.best_gradient_match,
                    best_step=result.best_step,
                )
            except Exception as exc:  # failures remain visible
                message = f"{type(exc).__name__}: {exc}"
                attempt = LearnedQuantileAttempt(
                    batch_start=batch_start,
                    restart_seed=restart_seed,
                    status="failed",
                    seconds=time.perf_counter() - started,
                    initializer_input_metrics=initializer_input_metrics,
                    initializer_target_metrics=initializer_target_metrics,
                    input_interval_coverage=input_coverage,
                    target_interval_coverage=target_coverage,
                    refined_batch=None,
                    best_objective=None,
                    best_gradient_l1=None,
                    best_step=None,
                    error_type=type(exc).__name__,
                    error_message_sha256=hashlib.sha256(
                        message.encode("utf-8")
                    ).hexdigest(),
                )
            group.append(len(attempts))
            attempts.append(attempt)

        successful = [
            index
            for index in group
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
            selected_indices.append(chosen)
            selected.append(attempts[chosen])

    failures = sum(attempt.status != "success" for attempt in attempts)
    exact = 0
    relative = 0
    initializer_values: dict[str, list[float]] = {}
    refined_values: dict[str, list[float]] = {}
    coverages_input: list[float] = []
    coverages_target: list[float] = []
    attack_seconds: list[float] = []
    for attempt in selected:
        assert attempt.initializer_input_metrics is not None
        assert attempt.refined_batch is not None
        metrics = attempt.refined_batch["aligned_metrics"]
        assert isinstance(metrics, dict)
        exact += int(bool(attempt.refined_batch["exact_batch_within_tolerance"]))
        relative += int(
            float(metrics["relative_l2_error"]) <= manifest.relative_l2_threshold
        )
        for name, value in attempt.initializer_input_metrics.items():
            initializer_values.setdefault(name, []).append(float(value))
        for name, value in metrics.items():
            refined_values.setdefault(str(name), []).append(float(value))
        coverages_input.append(float(attempt.input_interval_coverage))
        coverages_target.append(float(attempt.target_interval_coverage))
        attack_seconds.append(float(attempt.seconds))

    declared = len(manifest.attack_indices)
    summary: dict[str, object] = {
        "declared_attack_batches": declared,
        "selected_successful_batches": len(selected),
        "total_restart_attempts": len(attempts),
        "failed_restart_attempts": failures,
        "restart_completion": summarize_proportion(
            len(attempts) - failures,
            len(attempts),
            confidence_level=manifest.confidence_level,
        ).to_dict(),
        "exact_batch_success": summarize_proportion(
            exact,
            declared,
            confidence_level=manifest.confidence_level,
        ).to_dict(),
        "relative_l2_success": summarize_proportion(
            relative,
            declared,
            confidence_level=manifest.confidence_level,
        ).to_dict(),
        "initializer_metric_summaries": {
            name: _scalar(values, manifest, f"initializer:{name}")
            for name, values in sorted(initializer_values.items())
        },
        "refined_metric_summaries": {
            name: _scalar(values, manifest, f"refined:{name}")
            for name, values in sorted(refined_values.items())
        },
        "input_quantile_coverage": _scalar(
            coverages_input, manifest, "input_quantile_coverage"
        ),
        "target_quantile_coverage": _scalar(
            coverages_target, manifest, "target_quantile_coverage"
        ),
        "selected_attack_seconds": _scalar(
            attack_seconds, manifest, "selected_attack_seconds"
        ),
    }

    architecture = str(manifest.victim.get("architecture", "")).lower()
    dataset_name = str(manifest.dataset.get("name", "")).lower()
    architecture_is_modern = architecture in MODERN_FORECASTING_ARCHITECTURES
    real_dataset = not dataset_name.startswith("synthetic")
    split_disjoint = not (
        set(auxiliary.victim_training_indices) & set(auxiliary.auxiliary_indices)
    )
    attacks_are_training = attacked <= victim_training
    enough_auxiliary = (
        len(auxiliary.auxiliary_indices)
        >= auxiliary.minimum_publication_auxiliary_samples
    )
    enough_batches = declared >= manifest.minimum_publication_batches
    enough_seeds = (
        len(manifest.attack_seeds) >= manifest.minimum_publication_attack_seeds
    )
    every_batch = len(selected) == declared
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
        and every_batch
        and no_failed
    )
    gate = LearnedQuantileQualityGate(
        architecture_is_modern=architecture_is_modern,
        real_dataset=real_dataset,
        victim_and_auxiliary_splits_disjoint=split_disjoint,
        attacked_samples_are_victim_training_data=attacks_are_training,
        enough_auxiliary_samples=enough_auxiliary,
        enough_attack_batches=enough_batches,
        enough_attack_seeds=enough_seeds,
        every_batch_has_successful_attempt=every_batch,
        no_failed_attempts=no_failed,
        publication_mode=manifest.publication_mode,
        learned_quantile_initializer_present=True,
        passed=passed,
    )

    return LearnedQuantileBenchmarkReport(
        manifest=manifest,
        auxiliary=auxiliary,
        manifest_sha256=_combined_sha256(manifest, auxiliary),
        dataset_sha256=_tensor_sha256(inputs, targets),
        victim_model_sha256=_model_sha256(model),
        victim_class=type(model).__name__,
        trainable_parameters=sum(parameter.numel() for parameter in model.parameters()),
        victim_training_seconds=victim_training_seconds,
        auxiliary_gradient_seconds=auxiliary_gradient_seconds,
        initializer_training=initializer_report.to_dict(),
        attempts=tuple(attempts),
        selected_attempt_indices=tuple(selected_indices),
        summary=summary,
        quality_gate=gate,
        environment=benchmark_environment_manifest(),
        claim_boundary=(
            "This reproduces the learned gradient-to-input/target quantile initializer "
            "family with an explicit disjoint auxiliary split and then refines its input "
            "median under the known-target TS-Inverse L1 objective. It is not an "
            "unknown-target end-to-end reproduction and provides no quantum-advantage "
            "evidence."
        ),
    )
