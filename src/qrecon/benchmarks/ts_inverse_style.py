from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch

from qrecon.attacks import leak_gradients
from qrecon.attacks.ts_inverse_style import (
    TS_INVERSE_REFERENCE,
    TSInverseStyleAttack,
)
from qrecon.benchmarks.modern_timeseries_reconstruction import (
    MODERN_FORECASTING_ARCHITECTURES,
    ModernTimeSeriesAttackManifest,
)
from qrecon.benchmarks.statistics import (
    benchmark_environment_manifest,
    summarize_proportion,
    summarize_scalar,
)
from qrecon.experiment import _build_model, _load_dataset, _prior, _seed_everything, _train
from qrecon.metrics import permutation_invariant_batch_metrics


def _hash_tensors(*tensors: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _hash_model(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class TSInverseStyleAttempt:
    batch_start: int
    batch_indices: tuple[int, ...]
    restart_seed: int
    status: str
    seconds: float
    best_objective: float | None
    best_gradient_l1: float | None
    best_step: int | None
    final_objective: float | None
    final_gradient_l1: float | None
    aligned_batch: dict[str, object] | None
    error_type: str | None = None
    error_message_sha256: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["batch_indices"] = list(self.batch_indices)
        return payload


@dataclass(frozen=True)
class TSInverseStyleQualityGate:
    architecture_is_modern: bool
    real_dataset: bool
    enough_attack_batches: bool
    enough_attack_seeds: bool
    every_batch_has_successful_attempt: bool
    no_failed_attempts: bool
    official_objective_components_present: bool
    learned_quantile_model_present: bool
    publication_mode: bool
    passed_objective_baseline_gate: bool
    full_ts_inverse_reproduction: bool

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)


@dataclass(frozen=True)
class TSInverseStyleBenchmarkReport:
    manifest: ModernTimeSeriesAttackManifest
    manifest_sha256: str
    provenance: dict[str, object]
    dataset_sha256: str
    model_sha256: str
    victim_class: str
    trainable_parameters: int
    training_seconds: float
    attempts: tuple[TSInverseStyleAttempt, ...]
    selected_attempt_indices: tuple[int, ...]
    summary: dict[str, object]
    quality_gate: TSInverseStyleQualityGate
    environment: dict[str, object]
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest": self.manifest.to_dict(),
            "manifest_sha256": self.manifest_sha256,
            "provenance": self.provenance,
            "dataset_sha256": self.dataset_sha256,
            "model_sha256": self.model_sha256,
            "victim_class": self.victim_class,
            "trainable_parameters": self.trainable_parameters,
            "training_seconds": self.training_seconds,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "selected_attempt_indices": list(self.selected_attempt_indices),
            "summary": self.summary,
            "quality_gate": self.quality_gate.to_dict(),
            "environment": self.environment,
            "claim_boundary": self.claim_boundary,
        }


def _attack_kwargs(config: dict[str, object]) -> dict[str, object]:
    return {
        "steps": int(config.get("steps", 300)),
        "learning_rate": float(config.get("learning_rate", 0.05)),
        "optimizer_name": str(config.get("optimizer", "adam")),
        "gradient_l1_weight": float(config.get("gradient_l1_weight", 1.0)),
        "input_total_variation_weight": float(
            config.get("input_total_variation_weight", 0.0)
        ),
        "target_total_variation_weight": float(
            config.get("target_total_variation_weight", 0.0)
        ),
        "trend_weight": float(config.get("trend_weight", 0.0)),
        "trend_loss": str(config.get("trend_loss", "l1")),
        "trend_detach": bool(config.get("trend_detach", True)),
        "periodicity_weight": float(config.get("periodicity_weight", 0.0)),
        "periodicity_period": (
            None
            if config.get("periodicity_period") is None
            else int(config["periodicity_period"])
        ),
        "periodicity_loss": str(config.get("periodicity_loss", "l1")),
        "low_resolution_weight": float(config.get("low_resolution_weight", 0.0)),
        "low_resolution_factor": int(config.get("low_resolution_factor", 2)),
        "low_resolution_loss": str(config.get("low_resolution_loss", "l1")),
        "quantile_bound_weight": float(config.get("quantile_bound_weight", 0.0)),
        "joint_channel": int(config.get("joint_channel", 0)),
        "gradient_clip_norm": (
            None
            if config.get("gradient_clip_norm") is None
            else float(config["gradient_clip_norm"])
        ),
        "record_every": (
            None
            if config.get("record_every") is None
            else int(config["record_every"])
        ),
    }


def _scalar_summary(
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
        label=label,
    ).to_dict()


def run_ts_inverse_style_benchmark(
    manifest: ModernTimeSeriesAttackManifest,
) -> TSInverseStyleBenchmarkReport:
    """Run a matched TS-Inverse objective baseline on one declared victim.

    Every attack seed is applied to every declared batch. The selected restart is
    chosen by the released-gradient objective, never by the private-reference
    reconstruction metric. Failures stay in all denominators.
    """

    _seed_everything(manifest.victim_seed)
    dataset, task, mode = _load_dataset(
        {"seed": manifest.victim_seed, "dataset": dict(manifest.dataset)}
    )
    if task != "forecasting" or mode != "timeseries":
        raise ValueError("TS-Inverse-style benchmark requires forecasting data")
    x, targets = dataset.tensors
    model = _build_model(dataset, task, dict(manifest.victim))
    started = time.perf_counter()
    _train(model, dataset, task, dict(manifest.training))
    training_seconds = time.perf_counter() - started

    attempts: list[TSInverseStyleAttempt] = []
    selected_indices: list[int] = []
    selected: list[TSInverseStyleAttempt] = []
    attack_config = dict(manifest.attack)

    for batch_start in manifest.attack_indices:
        end = batch_start + manifest.attack_batch_size
        if end > len(x):
            raise ValueError(
                f"attack batch [{batch_start}, {end}) exceeds dataset size {len(x)}"
            )
        true_x = x[batch_start:end].clone()
        true_target = targets[batch_start:end].clone()
        observed = leak_gradients(model, true_x, true_target, "forecasting")
        batch_indices = tuple(range(batch_start, end))
        group: list[int] = []

        for restart_seed in manifest.attack_seeds:
            started = time.perf_counter()
            try:
                _seed_everything(restart_seed)
                prior = _prior(tuple(true_x.shape), "timeseries", attack_config)
                known_target = (
                    true_target if attack_config.get("known_target", True) else None
                )
                attack = TSInverseStyleAttack(
                    model,
                    observed,
                    prior,
                    known_target=known_target,
                    target_shape=tuple(true_target.shape),
                    **_attack_kwargs(attack_config),
                )
                result = attack.run()
                aligned = permutation_invariant_batch_metrics(
                    true_x,
                    result.reconstruction,
                    mode="timeseries",
                    tolerance=manifest.exact_tolerance,
                ).to_dict()
                attempt = TSInverseStyleAttempt(
                    batch_start=batch_start,
                    batch_indices=batch_indices,
                    restart_seed=restart_seed,
                    status="success",
                    seconds=time.perf_counter() - started,
                    best_objective=result.best_objective,
                    best_gradient_l1=result.best_gradient_match,
                    best_step=result.best_step,
                    final_objective=result.final_objective,
                    final_gradient_l1=result.final_gradient_match,
                    aligned_batch=aligned,
                )
            except Exception as exc:  # failures remain visible and counted
                message = f"{type(exc).__name__}: {exc}"
                attempt = TSInverseStyleAttempt(
                    batch_start=batch_start,
                    batch_indices=batch_indices,
                    restart_seed=restart_seed,
                    status="failed",
                    seconds=time.perf_counter() - started,
                    best_objective=None,
                    best_gradient_l1=None,
                    best_step=None,
                    final_objective=None,
                    final_gradient_l1=None,
                    aligned_batch=None,
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
    record_successes = 0
    record_trials = 0
    scalar_values: dict[str, list[float]] = {}
    for attempt in selected:
        assert attempt.aligned_batch is not None
        aligned = attempt.aligned_batch
        metrics = aligned["aligned_metrics"]
        assert isinstance(metrics, dict)
        exact += int(bool(aligned["exact_batch_within_tolerance"]))
        relative += int(
            float(metrics["relative_l2_error"]) <= manifest.relative_l2_threshold
        )
        record_successes += int(aligned["record_success_count"])
        record_trials += len(attempt.batch_indices)
        for name, value in metrics.items():
            scalar_values.setdefault(str(name), []).append(float(value))
        scalar_values.setdefault("best_objective", []).append(
            float(attempt.best_objective)
        )
        scalar_values.setdefault("best_gradient_l1", []).append(
            float(attempt.best_gradient_l1)
        )
        scalar_values.setdefault("attack_seconds", []).append(float(attempt.seconds))

    declared_batches = len(manifest.attack_indices)
    summary: dict[str, object] = {
        "declared_attack_batches": declared_batches,
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
            declared_batches,
            confidence_level=manifest.confidence_level,
        ).to_dict(),
        "relative_l2_success": summarize_proportion(
            relative,
            declared_batches,
            confidence_level=manifest.confidence_level,
        ).to_dict(),
        "record_tolerance_success": (
            None
            if record_trials == 0
            else summarize_proportion(
                record_successes,
                record_trials,
                confidence_level=manifest.confidence_level,
            ).to_dict()
        ),
        "scalar_summaries": {
            name: _scalar_summary(values, manifest, f"ts-inverse-style:{name}")
            for name, values in sorted(scalar_values.items())
        },
    }

    architecture = str(manifest.victim.get("architecture", "")).lower()
    dataset_name = str(manifest.dataset.get("name", "")).lower()
    architecture_is_modern = architecture in MODERN_FORECASTING_ARCHITECTURES
    real_dataset = not dataset_name.startswith("synthetic")
    enough_batches = declared_batches >= manifest.minimum_publication_batches
    enough_seeds = (
        len(manifest.attack_seeds) >= manifest.minimum_publication_attack_seeds
    )
    every_batch = len(selected) == declared_batches
    no_failed = failures == 0
    objective_components = True
    learned_quantile_model = False
    baseline_gate = (
        architecture_is_modern
        and real_dataset
        and every_batch
        and no_failed
        and objective_components
    )
    quality_gate = TSInverseStyleQualityGate(
        architecture_is_modern=architecture_is_modern,
        real_dataset=real_dataset,
        enough_attack_batches=enough_batches,
        enough_attack_seeds=enough_seeds,
        every_batch_has_successful_attempt=every_batch,
        no_failed_attempts=no_failed,
        official_objective_components_present=objective_components,
        learned_quantile_model_present=learned_quantile_model,
        publication_mode=manifest.publication_mode,
        passed_objective_baseline_gate=baseline_gate,
        full_ts_inverse_reproduction=False,
    )

    return TSInverseStyleBenchmarkReport(
        manifest=manifest,
        manifest_sha256=manifest.sha256,
        provenance=json.loads(json.dumps(TS_INVERSE_REFERENCE)),
        dataset_sha256=_hash_tensors(x, targets),
        model_sha256=_hash_model(model),
        victim_class=type(model).__name__,
        trainable_parameters=sum(parameter.numel() for parameter in model.parameters()),
        training_seconds=training_seconds,
        attempts=tuple(attempts),
        selected_attempt_indices=tuple(selected_indices),
        summary=summary,
        quality_gate=quality_gate,
        environment=benchmark_environment_manifest(),
        claim_boundary=(
            "This is a provenance-tracked reproduction of the public TS-Inverse "
            "optimization objective family, not the complete learned quantile/initializer "
            "pipeline. It is a classical white-box baseline and provides no quantum "
            "advantage evidence."
        ),
    )
