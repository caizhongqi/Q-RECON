from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import torch

from qrecon.attacks import GradientInversionAttack, leak_gradients
from qrecon.benchmarks.modern_timeseries_reconstruction import (
    MODERN_FORECASTING_ARCHITECTURES,
    ModernTimeSeriesAttackManifest,
)
from qrecon.benchmarks.statistics import (
    benchmark_environment_manifest,
    summarize_proportion,
    summarize_scalar,
)
from qrecon.experiment import (
    _build_model,
    _load_dataset,
    _prior,
    _seed_everything,
    _train,
)
from qrecon.metrics import permutation_invariant_batch_metrics


@dataclass(frozen=True)
class ModernAttackVariant:
    name: str
    overrides: Mapping[str, object]

    def __post_init__(self) -> None:
        normalized = self.name.strip()
        if not normalized:
            raise ValueError("attack variant name must be non-empty")
        object.__setattr__(self, "name", normalized)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "overrides": json.loads(json.dumps(dict(self.overrides), sort_keys=True)),
        }


@dataclass(frozen=True)
class ModernAttackSuiteAttempt:
    variant: str
    batch_start: int
    batch_indices: tuple[int, ...]
    restart_seed: int
    status: str
    seconds: float
    best_objective: float | None
    best_gradient_match: float | None
    best_step: int | None
    final_objective: float | None
    final_gradient_match: float | None
    aligned_batch: dict[str, object] | None
    error_type: str | None = None
    error_message_sha256: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["batch_indices"] = list(self.batch_indices)
        return payload


@dataclass(frozen=True)
class ModernAttackSuiteQualityGate:
    architecture_is_modern: bool
    real_dataset: bool
    enough_attack_batches: bool
    enough_attack_seeds: bool
    required_baselines_present: bool
    every_variant_batch_has_successful_attempt: bool
    no_failed_attempts: bool
    publication_mode: bool
    passed: bool

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)


@dataclass(frozen=True)
class ModernAttackSuiteReport:
    base_manifest: ModernTimeSeriesAttackManifest
    variants: tuple[ModernAttackVariant, ...]
    suite_sha256: str
    dataset_sha256: str
    model_sha256: str
    victim_class: str
    trainable_parameters: int
    training_seconds: float
    attempts: tuple[ModernAttackSuiteAttempt, ...]
    selected_attempt_indices: dict[str, tuple[int, ...]]
    variant_summaries: dict[str, dict[str, object]]
    quality_gate: ModernAttackSuiteQualityGate
    environment: dict[str, object]
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        return {
            "base_manifest": self.base_manifest.to_dict(),
            "variants": [variant.to_dict() for variant in self.variants],
            "suite_sha256": self.suite_sha256,
            "dataset_sha256": self.dataset_sha256,
            "model_sha256": self.model_sha256,
            "victim_class": self.victim_class,
            "trainable_parameters": self.trainable_parameters,
            "training_seconds": self.training_seconds,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "selected_attempt_indices": {
                name: list(indices)
                for name, indices in sorted(self.selected_attempt_indices.items())
            },
            "variant_summaries": self.variant_summaries,
            "quality_gate": self.quality_gate.to_dict(),
            "environment": self.environment,
            "claim_boundary": self.claim_boundary,
        }


def standard_modern_attack_variants(
    *,
    period: int,
) -> tuple[ModernAttackVariant, ...]:
    """Return a predeclared matched suite of established and temporal objectives."""

    if period <= 0:
        raise ValueError("period must be positive")
    return (
        ModernAttackVariant(
            "dlg_l2",
            {
                "match_mode": "l2",
                "layer_weighting": "parameter",
                "regularization": 0.0,
            },
        ),
        ModernAttackVariant(
            "invg_cosine",
            {
                "match_mode": "cosine",
                "layer_weighting": "parameter",
                "regularization": 0.0,
            },
        ),
        ModernAttackVariant(
            "qrecon_hybrid",
            {
                "match_mode": "hybrid",
                "layer_weighting": "parameter",
                "regularization": 1e-4,
            },
        ),
        ModernAttackVariant(
            "temporal_prior_hybrid",
            {
                "match_mode": "hybrid",
                "layer_weighting": "parameter",
                "regularization": 1e-4,
                "trend_regularization": 1e-3,
                "trend_loss": "l1",
                "trend_detach": True,
                "periodicity_regularization": 1e-3,
                "periodicity_period": int(period),
                "periodicity_loss": "l1",
                "low_resolution_regularization": 1e-3,
                "low_resolution_factor": 2,
                "low_resolution_loss": "l1",
            },
        ),
    )


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


def _suite_sha256(
    manifest: ModernTimeSeriesAttackManifest,
    variants: Sequence[ModernAttackVariant],
) -> str:
    payload = {
        "schema_version": "qrecon.modern-timeseries-attack-suite.v1",
        "base_manifest": manifest.to_dict(),
        "variants": [variant.to_dict() for variant in variants],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _attack_kwargs(config: Mapping[str, object]) -> dict[str, object]:
    return {
        "steps": int(config.get("steps", 300)),
        "learning_rate": float(config.get("learning_rate", 0.05)),
        "regularization": float(config.get("regularization", 1e-3)),
        "optimizer_name": str(config.get("optimizer", "adam")),
        "match_mode": str(config.get("match_mode", "hybrid")),
        "layer_weighting": str(config.get("layer_weighting", "parameter")),
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
        "trend_regularization": float(config.get("trend_regularization", 0.0)),
        "trend_loss": str(config.get("trend_loss", "l1")),
        "trend_detach": bool(config.get("trend_detach", True)),
        "periodicity_regularization": float(
            config.get("periodicity_regularization", 0.0)
        ),
        "periodicity_period": (
            None
            if config.get("periodicity_period") is None
            else int(config["periodicity_period"])
        ),
        "periodicity_loss": str(config.get("periodicity_loss", "l1")),
        "low_resolution_regularization": float(
            config.get("low_resolution_regularization", 0.0)
        ),
        "low_resolution_factor": int(config.get("low_resolution_factor", 2)),
        "low_resolution_loss": str(config.get("low_resolution_loss", "l1")),
    }


def _summarize_variant(
    attempts: Sequence[ModernAttackSuiteAttempt],
    selected: Sequence[ModernAttackSuiteAttempt],
    manifest: ModernTimeSeriesAttackManifest,
) -> dict[str, object]:
    failures = sum(attempt.status != "success" for attempt in attempts)
    exact = 0
    relative_l2 = 0
    records = 0
    record_trials = 0
    scalar_values: dict[str, list[float]] = {}
    for attempt in selected:
        assert attempt.aligned_batch is not None
        aligned = attempt.aligned_batch
        metrics = aligned["aligned_metrics"]
        assert isinstance(metrics, dict)
        exact += int(bool(aligned["exact_batch_within_tolerance"]))
        records += int(aligned["record_success_count"])
        record_trials += len(attempt.batch_indices)
        relative_l2 += int(
            float(metrics["relative_l2_error"]) <= manifest.relative_l2_threshold
        )
        for name, value in metrics.items():
            scalar_values.setdefault(str(name), []).append(float(value))
        scalar_values.setdefault("best_objective", []).append(
            float(attempt.best_objective)
        )
        scalar_values.setdefault("best_gradient_match", []).append(
            float(attempt.best_gradient_match)
        )
        scalar_values.setdefault("attack_seconds", []).append(float(attempt.seconds))

    def scalar(name: str, values: Sequence[float]) -> dict[str, object]:
        return summarize_scalar(
            values,
            confidence_level=manifest.confidence_level,
            bootstrap_samples=manifest.bootstrap_samples,
            bootstrap_seed=manifest.bootstrap_seed,
            label=f"modern-suite:{name}",
        ).to_dict()

    declared_batches = len(manifest.attack_indices)
    return {
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
            relative_l2,
            declared_batches,
            confidence_level=manifest.confidence_level,
        ).to_dict(),
        "record_tolerance_success": (
            None
            if record_trials == 0
            else summarize_proportion(
                records,
                record_trials,
                confidence_level=manifest.confidence_level,
            ).to_dict()
        ),
        "scalar_summaries": {
            name: scalar(name, values)
            for name, values in sorted(scalar_values.items())
            if values
        },
    }


def run_modern_timeseries_attack_suite(
    manifest: ModernTimeSeriesAttackManifest,
    variants: Sequence[ModernAttackVariant],
) -> ModernAttackSuiteReport:
    declared_variants = tuple(variants)
    if not declared_variants:
        raise ValueError("at least one attack variant is required")
    names = tuple(variant.name for variant in declared_variants)
    if len(set(names)) != len(names):
        raise ValueError("attack variant names must be unique")

    _seed_everything(manifest.victim_seed)
    dataset, task, mode = _load_dataset(
        {"seed": manifest.victim_seed, "dataset": dict(manifest.dataset)}
    )
    if task != "forecasting" or mode != "timeseries":
        raise ValueError("modern attack suites require a forecasting dataset")
    x, targets = dataset.tensors
    model = _build_model(dataset, task, dict(manifest.victim))
    started = time.perf_counter()
    _train(model, dataset, task, dict(manifest.training))
    training_seconds = time.perf_counter() - started

    attempts: list[ModernAttackSuiteAttempt] = []
    selected_indices: dict[str, list[int]] = {name: [] for name in names}
    selected_attempts: dict[str, list[ModernAttackSuiteAttempt]] = {
        name: [] for name in names
    }
    attempts_by_variant: dict[str, list[ModernAttackSuiteAttempt]] = {
        name: [] for name in names
    }

    for batch_start in manifest.attack_indices:
        end = batch_start + manifest.attack_batch_size
        if end > len(x):
            raise ValueError(
                f"attack batch [{batch_start}, {end}) exceeds dataset size {len(x)}"
            )
        true_x = x[batch_start:end].clone()
        true_target = targets[batch_start:end].clone()
        observed = leak_gradients(model, true_x, true_target, "forecasting")
        indices = tuple(range(batch_start, end))

        for variant in declared_variants:
            merged = dict(manifest.attack)
            merged.update(dict(variant.overrides))
            group_indices: list[int] = []
            for restart_seed in manifest.attack_seeds:
                started = time.perf_counter()
                try:
                    _seed_everything(restart_seed)
                    prior = _prior(tuple(true_x.shape), "timeseries", merged)
                    known_target = (
                        true_target if merged.get("known_target", True) else None
                    )
                    attack = GradientInversionAttack(
                        model=model,
                        observed_gradients=observed,
                        prior=prior,
                        task="forecasting",
                        mode="timeseries",
                        known_target=known_target,
                        target_shape=tuple(true_target.shape),
                        **_attack_kwargs(merged),  # type: ignore[arg-type]
                    )
                    result = attack.run()
                    aligned = permutation_invariant_batch_metrics(
                        true_x,
                        result.reconstruction,
                        mode="timeseries",
                        tolerance=manifest.exact_tolerance,
                    ).to_dict()
                    attempt = ModernAttackSuiteAttempt(
                        variant=variant.name,
                        batch_start=batch_start,
                        batch_indices=indices,
                        restart_seed=restart_seed,
                        status="success",
                        seconds=time.perf_counter() - started,
                        best_objective=result.best_objective,
                        best_gradient_match=result.best_gradient_match,
                        best_step=result.best_step,
                        final_objective=result.final_objective,
                        final_gradient_match=result.final_gradient_match,
                        aligned_batch=aligned,
                    )
                except Exception as exc:
                    message = f"{type(exc).__name__}: {exc}"
                    attempt = ModernAttackSuiteAttempt(
                        variant=variant.name,
                        batch_start=batch_start,
                        batch_indices=indices,
                        restart_seed=restart_seed,
                        status="failed",
                        seconds=time.perf_counter() - started,
                        best_objective=None,
                        best_gradient_match=None,
                        best_step=None,
                        final_objective=None,
                        final_gradient_match=None,
                        aligned_batch=None,
                        error_type=type(exc).__name__,
                        error_message_sha256=hashlib.sha256(
                            message.encode("utf-8")
                        ).hexdigest(),
                    )
                group_indices.append(len(attempts))
                attempts.append(attempt)
                attempts_by_variant[variant.name].append(attempt)

            successful = [
                index
                for index in group_indices
                if attempts[index].status == "success"
                and attempts[index].best_objective is not None
            ]
            if successful:
                chosen = min(
                    successful,
                    key=lambda index: (
                        float(attempts[index].best_objective),
                        float(attempts[index].best_gradient_match),
                        attempts[index].restart_seed,
                    ),
                )
                selected_indices[variant.name].append(chosen)
                selected_attempts[variant.name].append(attempts[chosen])

    summaries = {
        name: _summarize_variant(
            attempts_by_variant[name], selected_attempts[name], manifest
        )
        for name in names
    }

    required = {"dlg_l2", "invg_cosine", "qrecon_hybrid", "temporal_prior_hybrid"}
    architecture = str(manifest.victim.get("architecture", "")).lower()
    dataset_name = str(manifest.dataset.get("name", "")).lower()
    architecture_is_modern = architecture in MODERN_FORECASTING_ARCHITECTURES
    real_dataset = not dataset_name.startswith("synthetic")
    enough_batches = len(manifest.attack_indices) >= manifest.minimum_publication_batches
    enough_seeds = len(manifest.attack_seeds) >= manifest.minimum_publication_attack_seeds
    required_present = required.issubset(set(names))
    every_cell = all(
        len(selected_attempts[name]) == len(manifest.attack_indices) for name in names
    )
    no_failures = all(attempt.status == "success" for attempt in attempts)
    passed = (
        manifest.publication_mode
        and architecture_is_modern
        and real_dataset
        and enough_batches
        and enough_seeds
        and required_present
        and every_cell
        and no_failures
    )
    quality_gate = ModernAttackSuiteQualityGate(
        architecture_is_modern=architecture_is_modern,
        real_dataset=real_dataset,
        enough_attack_batches=enough_batches,
        enough_attack_seeds=enough_seeds,
        required_baselines_present=required_present,
        every_variant_batch_has_successful_attempt=every_cell,
        no_failed_attempts=no_failures,
        publication_mode=manifest.publication_mode,
        passed=passed,
    )

    return ModernAttackSuiteReport(
        base_manifest=manifest,
        variants=declared_variants,
        suite_sha256=_suite_sha256(manifest, declared_variants),
        dataset_sha256=_hash_tensors(x, targets),
        model_sha256=_hash_model(model),
        victim_class=type(model).__name__,
        trainable_parameters=sum(parameter.numel() for parameter in model.parameters()),
        training_seconds=training_seconds,
        attempts=tuple(attempts),
        selected_attempt_indices={
            name: tuple(indices) for name, indices in selected_indices.items()
        },
        variant_summaries=summaries,
        quality_gate=quality_gate,
        environment=benchmark_environment_manifest(),
        claim_boundary=(
            "The temporal-prior variant is an independently implemented, declared "
            "trend/periodicity/resolution baseline inspired by public time-series "
            "gradient-inversion methodology. It is not a reproduction claim and no "
            "variant implies coherent quantum access or quantum advantage."
        ),
    )
