from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
import torch

from qrecon.attacks import GradientInversionAttack, leak_gradients
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
from qrecon.metrics import (
    permutation_invariant_batch_metrics,
    reconstruction_metrics,
)


MODERN_FORECASTING_ARCHITECTURES = (
    "transformer",
    "patchtst",
    "itransformer",
)


def _json_copy(value: object) -> object:
    return json.loads(json.dumps(value, sort_keys=True))


def _finite_positive(name: str, value: float) -> float:
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return converted


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


@dataclass(frozen=True)
class ModernTimeSeriesAttackManifest:
    """Canonical experiment declaration for modern forecasting reconstruction.

    Attack indices denote batch start positions. Each declared batch is attacked
    with every independent attack seed. A selected reconstruction is chosen only
    by the released-gradient objective, never by access to the private reference.
    """

    dataset: Mapping[str, object]
    victim: Mapping[str, object]
    training: Mapping[str, object]
    attack: Mapping[str, object]
    victim_seed: int
    attack_indices: tuple[int, ...]
    attack_seeds: tuple[int, ...]
    attack_batch_size: int = 1
    exact_tolerance: float = 1e-2
    relative_l2_threshold: float = 0.1
    confidence_level: float = 0.95
    bootstrap_samples: int = 2000
    bootstrap_seed: int = 1729
    minimum_publication_batches: int = 20
    minimum_publication_attack_seeds: int = 3
    publication_mode: bool = False

    def __post_init__(self) -> None:
        if not self.dataset or not self.victim or not self.training or not self.attack:
            raise ValueError("dataset, victim, training and attack declarations are required")
        architecture = str(self.victim.get("architecture", "")).lower()
        if architecture not in (*MODERN_FORECASTING_ARCHITECTURES, "mlp"):
            raise ValueError(
                "victim architecture must be mlp, transformer, patchtst or itransformer"
            )
        if self.attack_batch_size <= 0:
            raise ValueError("attack_batch_size must be positive")
        if not self.attack_indices or any(index < 0 for index in self.attack_indices):
            raise ValueError("attack_indices must contain non-negative positions")
        if len(set(self.attack_indices)) != len(self.attack_indices):
            raise ValueError("attack_indices must be unique")
        if not self.attack_seeds:
            raise ValueError("attack_seeds must be non-empty")
        if len(set(self.attack_seeds)) != len(self.attack_seeds):
            raise ValueError("attack_seeds must be unique")
        _finite_positive("exact_tolerance", self.exact_tolerance)
        _finite_positive("relative_l2_threshold", self.relative_l2_threshold)
        if not 0.0 < float(self.confidence_level) < 1.0:
            raise ValueError("confidence_level must lie strictly between zero and one")
        if self.bootstrap_samples <= 0:
            raise ValueError("bootstrap_samples must be positive")
        if self.minimum_publication_batches <= 0:
            raise ValueError("minimum_publication_batches must be positive")
        if self.minimum_publication_attack_seeds <= 0:
            raise ValueError("minimum_publication_attack_seeds must be positive")

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ModernTimeSeriesAttackManifest":
        converted = dict(payload)
        converted["attack_indices"] = tuple(int(value) for value in payload["attack_indices"])
        converted["attack_seeds"] = tuple(int(value) for value in payload["attack_seeds"])
        return cls(**converted)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "qrecon.modern-timeseries-reconstruction.v1",
            "dataset": _json_copy(dict(self.dataset)),
            "victim": _json_copy(dict(self.victim)),
            "training": _json_copy(dict(self.training)),
            "attack": _json_copy(dict(self.attack)),
            "victim_seed": int(self.victim_seed),
            "attack_indices": list(self.attack_indices),
            "attack_seeds": list(self.attack_seeds),
            "attack_batch_size": int(self.attack_batch_size),
            "exact_tolerance": float(self.exact_tolerance),
            "relative_l2_threshold": float(self.relative_l2_threshold),
            "confidence_level": float(self.confidence_level),
            "bootstrap_samples": int(self.bootstrap_samples),
            "bootstrap_seed": int(self.bootstrap_seed),
            "minimum_publication_batches": int(self.minimum_publication_batches),
            "minimum_publication_attack_seeds": int(
                self.minimum_publication_attack_seeds
            ),
            "publication_mode": bool(self.publication_mode),
        }

    @property
    def sha256(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ModernAttackAttempt:
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
    ordered_metrics: dict[str, float] | None
    aligned_batch: dict[str, object] | None
    error_type: str | None = None
    error_message_sha256: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["batch_indices"] = list(self.batch_indices)
        return payload


@dataclass(frozen=True)
class ModernTimeSeriesQualityGate:
    architecture_is_modern: bool
    real_dataset: bool
    enough_attack_batches: bool
    enough_attack_seeds: bool
    every_batch_has_successful_attempt: bool
    no_failed_attempts: bool
    publication_mode: bool
    passed: bool

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)


@dataclass(frozen=True)
class ModernTimeSeriesBenchmarkReport:
    manifest: ModernTimeSeriesAttackManifest
    manifest_sha256: str
    dataset_sha256: str
    model_sha256: str
    victim_class: str
    trainable_parameters: int
    training_seconds: float
    attempts: tuple[ModernAttackAttempt, ...]
    selected_attempt_indices: tuple[int, ...]
    summary: dict[str, object]
    quality_gate: ModernTimeSeriesQualityGate
    environment: dict[str, object]
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest": self.manifest.to_dict(),
            "manifest_sha256": self.manifest_sha256,
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


def load_modern_timeseries_manifest(
    path: str,
) -> ModernTimeSeriesAttackManifest:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") not in (
        None,
        "qrecon.modern-timeseries-reconstruction.v1",
    ):
        raise ValueError("unsupported modern reconstruction manifest schema")
    payload.pop("schema_version", None)
    return ModernTimeSeriesAttackManifest.from_dict(payload)


def _attempt_attack(
    *,
    model: torch.nn.Module,
    true_x: torch.Tensor,
    true_target: torch.Tensor,
    batch_start: int,
    restart_seed: int,
    manifest: ModernTimeSeriesAttackManifest,
) -> ModernAttackAttempt:
    indices = tuple(range(batch_start, batch_start + int(true_x.shape[0])))
    started = time.perf_counter()
    try:
        _seed_everything(restart_seed)
        observed = leak_gradients(model, true_x, true_target, "forecasting")
        attack_config = dict(manifest.attack)
        known_target = true_target if attack_config.get("known_target", True) else None
        prior = _prior(tuple(true_x.shape), "timeseries", attack_config)
        attack = GradientInversionAttack(
            model=model,
            observed_gradients=observed,
            prior=prior,
            task="forecasting",
            mode="timeseries",
            known_target=known_target,
            target_shape=tuple(true_target.shape),
            steps=int(attack_config.get("steps", 300)),
            learning_rate=float(attack_config.get("learning_rate", 0.05)),
            regularization=float(attack_config.get("regularization", 1e-3)),
            optimizer_name=str(attack_config.get("optimizer", "adam")),
            match_mode=str(attack_config.get("match_mode", "hybrid")),  # type: ignore[arg-type]
            layer_weighting=str(
                attack_config.get("layer_weighting", "parameter")
            ),  # type: ignore[arg-type]
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
        ordered = reconstruction_metrics(
            true_x, result.reconstruction, mode="timeseries"
        )
        aligned = permutation_invariant_batch_metrics(
            true_x,
            result.reconstruction,
            mode="timeseries",
            tolerance=manifest.exact_tolerance,
        )
        return ModernAttackAttempt(
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
            ordered_metrics=ordered,
            aligned_batch=aligned.to_dict(),
        )
    except Exception as exc:  # failures must remain in the denominator
        message = f"{type(exc).__name__}: {exc}"
        return ModernAttackAttempt(
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
            ordered_metrics=None,
            aligned_batch=None,
            error_type=type(exc).__name__,
            error_message_sha256=hashlib.sha256(message.encode("utf-8")).hexdigest(),
        )


def _scalar_payload(
    values: list[float], manifest: ModernTimeSeriesAttackManifest, label: str
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


def run_modern_timeseries_reconstruction_benchmark(
    manifest: ModernTimeSeriesAttackManifest,
) -> ModernTimeSeriesBenchmarkReport:
    """Train one declared victim and run a multi-batch, multi-restart attack matrix."""

    _seed_everything(manifest.victim_seed)
    experiment_config: dict[str, Any] = {
        "seed": manifest.victim_seed,
        "dataset": dict(manifest.dataset),
    }
    dataset, task, mode = _load_dataset(experiment_config)
    if task != "forecasting" or mode != "timeseries":
        raise ValueError("modern time-series reconstruction requires a forecasting dataset")
    x, target = dataset.tensors
    dataset_hash = _tensor_sha256(x, target)

    model = _build_model(dataset, task, dict(manifest.victim))
    started = time.perf_counter()
    _train(model, dataset, task, dict(manifest.training))
    training_seconds = time.perf_counter() - started
    model_hash = _model_sha256(model)

    attempts: list[ModernAttackAttempt] = []
    selected_attempt_indices: list[int] = []
    selected: list[ModernAttackAttempt] = []
    batch_size = manifest.attack_batch_size

    for batch_start in manifest.attack_indices:
        if batch_start + batch_size > len(x):
            raise ValueError(
                f"attack batch [{batch_start}, {batch_start + batch_size}) exceeds "
                f"dataset size {len(x)}"
            )
        true_x = x[batch_start : batch_start + batch_size].clone()
        true_target = target[batch_start : batch_start + batch_size].clone()
        group_indices: list[int] = []
        for restart_seed in manifest.attack_seeds:
            attempt = _attempt_attack(
                model=model,
                true_x=true_x,
                true_target=true_target,
                batch_start=batch_start,
                restart_seed=restart_seed,
                manifest=manifest,
            )
            group_indices.append(len(attempts))
            attempts.append(attempt)
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
            selected_attempt_indices.append(chosen)
            selected.append(attempts[chosen])

    exact_successes = 0
    relative_l2_successes = 0
    record_successes = 0
    record_trials = 0
    aligned_metrics: dict[str, list[float]] = {}
    best_objectives: list[float] = []
    best_gradient_matches: list[float] = []
    selected_seconds: list[float] = []

    for attempt in selected:
        assert attempt.aligned_batch is not None
        aligned = attempt.aligned_batch
        metrics = aligned["aligned_metrics"]
        assert isinstance(metrics, dict)
        exact_successes += int(bool(aligned["exact_batch_within_tolerance"]))
        record_successes += int(aligned["record_success_count"])
        record_trials += len(attempt.batch_indices)
        relative_l2_successes += int(
            float(metrics["relative_l2_error"]) <= manifest.relative_l2_threshold
        )
        for name, value in metrics.items():
            aligned_metrics.setdefault(str(name), []).append(float(value))
        best_objectives.append(float(attempt.best_objective))
        best_gradient_matches.append(float(attempt.best_gradient_match))
        selected_seconds.append(float(attempt.seconds))

    failed_attempts = sum(attempt.status != "success" for attempt in attempts)
    selected_count = len(selected)
    summary: dict[str, object] = {
        "declared_attack_batches": len(manifest.attack_indices),
        "selected_successful_batches": selected_count,
        "total_restart_attempts": len(attempts),
        "failed_restart_attempts": failed_attempts,
        "restart_completion": summarize_proportion(
            len(attempts) - failed_attempts,
            len(attempts),
            confidence_level=manifest.confidence_level,
        ).to_dict(),
        "exact_batch_success": summarize_proportion(
            exact_successes,
            max(1, len(manifest.attack_indices)),
            confidence_level=manifest.confidence_level,
        ).to_dict(),
        "relative_l2_success": summarize_proportion(
            relative_l2_successes,
            max(1, len(manifest.attack_indices)),
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
        "metric_summaries": {
            name: _scalar_payload(values, manifest, f"modern:{name}")
            for name, values in sorted(aligned_metrics.items())
        },
        "best_objective": _scalar_payload(
            best_objectives, manifest, "modern:best_objective"
        ),
        "best_gradient_match": _scalar_payload(
            best_gradient_matches, manifest, "modern:best_gradient_match"
        ),
        "selected_attack_seconds": _scalar_payload(
            selected_seconds, manifest, "modern:selected_seconds"
        ),
    }

    architecture = str(manifest.victim.get("architecture", "")).lower()
    dataset_name = str(manifest.dataset.get("name", "")).lower()
    architecture_is_modern = architecture in MODERN_FORECASTING_ARCHITECTURES
    real_dataset = not dataset_name.startswith("synthetic")
    every_batch_has_success = selected_count == len(manifest.attack_indices)
    no_failed_attempts = failed_attempts == 0
    enough_batches = len(manifest.attack_indices) >= manifest.minimum_publication_batches
    enough_seeds = len(manifest.attack_seeds) >= manifest.minimum_publication_attack_seeds
    passed = (
        manifest.publication_mode
        and architecture_is_modern
        and real_dataset
        and enough_batches
        and enough_seeds
        and every_batch_has_success
        and no_failed_attempts
    )
    quality_gate = ModernTimeSeriesQualityGate(
        architecture_is_modern=architecture_is_modern,
        real_dataset=real_dataset,
        enough_attack_batches=enough_batches,
        enough_attack_seeds=enough_seeds,
        every_batch_has_successful_attempt=every_batch_has_success,
        no_failed_attempts=no_failed_attempts,
        publication_mode=manifest.publication_mode,
        passed=passed,
    )

    return ModernTimeSeriesBenchmarkReport(
        manifest=manifest,
        manifest_sha256=manifest.sha256,
        dataset_sha256=dataset_hash,
        model_sha256=model_hash,
        victim_class=type(model).__name__,
        trainable_parameters=sum(parameter.numel() for parameter in model.parameters()),
        training_seconds=training_seconds,
        attempts=tuple(attempts),
        selected_attempt_indices=tuple(selected_attempt_indices),
        summary=summary,
        quality_gate=quality_gate,
        environment=benchmark_environment_manifest(),
        claim_boundary=(
            "This report measures classical white-box gradient inversion against a "
            "modern forecasting victim. A passing experiment-completeness gate does "
            "not imply a coherent Transformer/PatchTST compiler or quantum advantage."
        ),
    )
