from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import torch

from qrecon.attacks import (
    GradientReleaseSpec,
    ReleasedGradientInversionAttack,
    last_biased_linear_parameter_indices,
    leak_gradients,
    release_gradients,
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
from qrecon.experiment import (
    _build_model,
    _load_dataset,
    _prior,
    _seed_everything,
    _train,
)
from qrecon.metrics import permutation_invariant_batch_metrics

VisibleScope = str


@dataclass(frozen=True)
class ModernGradientDefenseVariant:
    name: str
    release: Mapping[str, object]
    visible_scope: VisibleScope = "all"
    attack_overrides: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        normalized = self.name.strip()
        if not normalized:
            raise ValueError("defense variant name must be non-empty")
        if self.visible_scope not in ("all", "last_head"):
            raise ValueError("visible_scope must be 'all' or 'last_head'")
        object.__setattr__(self, "name", normalized)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "release": json.loads(json.dumps(dict(self.release), sort_keys=True)),
            "visible_scope": self.visible_scope,
            "attack_overrides": (
                None
                if self.attack_overrides is None
                else json.loads(
                    json.dumps(dict(self.attack_overrides), sort_keys=True)
                )
            ),
        }


@dataclass(frozen=True)
class ModernDefenseAttempt:
    variant: str
    batch_start: int
    batch_indices: tuple[int, ...]
    restart_seed: int
    status: str
    seconds: float
    release_metadata: dict[str, object]
    transform_metadata: dict[str, object] | None
    best_objective: float | None
    best_gradient_match: float | None
    best_step: int | None
    aligned_batch: dict[str, object] | None
    error_type: str | None = None
    error_message_sha256: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["batch_indices"] = list(self.batch_indices)
        return payload


@dataclass(frozen=True)
class ModernDefenseSuiteQualityGate:
    architecture_is_modern: bool
    real_dataset: bool
    enough_attack_batches: bool
    enough_attack_seeds: bool
    required_conditions_present: bool
    every_condition_batch_has_successful_attempt: bool
    no_failed_attempts: bool
    publication_mode: bool
    passed: bool

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)


@dataclass(frozen=True)
class ModernDefenseSuiteReport:
    base_manifest: ModernTimeSeriesAttackManifest
    variants: tuple[ModernGradientDefenseVariant, ...]
    suite_sha256: str
    dataset_sha256: str
    model_sha256: str
    victim_class: str
    trainable_parameters: int
    attempts: tuple[ModernDefenseAttempt, ...]
    selected_attempt_indices: dict[str, tuple[int, ...]]
    variant_summaries: dict[str, dict[str, object]]
    quality_gate: ModernDefenseSuiteQualityGate
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


def standard_modern_gradient_defenses() -> tuple[ModernGradientDefenseVariant, ...]:
    """A predeclared exact, clipped, quantized, noisy, and partial-view matrix."""

    return (
        ModernGradientDefenseVariant("full_exact", {}),
        ModernGradientDefenseVariant("global_clip_1", {"clip_norm": 1.0}),
        ModernGradientDefenseVariant(
            "symmetric_int8", {"quantization_bits": 8}
        ),
        ModernGradientDefenseVariant(
            "gaussian_noise_1e-3",
            {"noise_std": 1e-3, "noise_seed": 7001},
        ),
        ModernGradientDefenseVariant(
            "last_head_only",
            {},
            visible_scope="last_head",
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
    variants: Sequence[ModernGradientDefenseVariant],
) -> str:
    payload = {
        "schema_version": "qrecon.modern-timeseries-defense-suite.v1",
        "base_manifest": manifest.to_dict(),
        "variants": [variant.to_dict() for variant in variants],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _release_spec(
    model: torch.nn.Module,
    variant: ModernGradientDefenseVariant,
    batch_start: int,
) -> GradientReleaseSpec:
    payload = dict(variant.release)
    if "noise_seed" in payload:
        payload["noise_seed"] = int(payload["noise_seed"]) + int(batch_start)
    if variant.visible_scope == "last_head":
        payload["visible_parameter_indices"] = last_biased_linear_parameter_indices(
            model
        )
    return GradientReleaseSpec(**payload)  # type: ignore[arg-type]


def _attack_configuration(
    manifest: ModernTimeSeriesAttackManifest,
    variant: ModernGradientDefenseVariant,
) -> dict[str, object]:
    config = dict(manifest.attack)
    if variant.attack_overrides is not None:
        config.update(dict(variant.attack_overrides))
    return config


def _summarize_variant(
    attempts: Sequence[ModernDefenseAttempt],
    selected: Sequence[ModernDefenseAttempt],
    manifest: ModernTimeSeriesAttackManifest,
) -> dict[str, object]:
    failed = sum(attempt.status != "success" for attempt in attempts)
    exact = 0
    relative = 0
    scalar_values: dict[str, list[float]] = {}
    release_values: dict[str, list[float]] = {}
    for attempt in selected:
        assert attempt.aligned_batch is not None
        aligned = attempt.aligned_batch
        metrics = aligned["aligned_metrics"]
        assert isinstance(metrics, dict)
        exact += int(bool(aligned["exact_batch_within_tolerance"]))
        relative += int(
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
        for name in (
            "raw_l2_norm",
            "clipped_l2_norm",
            "clipping_factor",
            "noise_std",
            "quantized_saturation_rate",
        ):
            value = attempt.release_metadata.get(name)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                release_values.setdefault(name, []).append(float(value))

    def scalar(label: str, values: Sequence[float]) -> dict[str, object]:
        return summarize_scalar(
            values,
            confidence_level=manifest.confidence_level,
            bootstrap_samples=manifest.bootstrap_samples,
            bootstrap_seed=manifest.bootstrap_seed,
            label=f"defense:{label}",
        ).to_dict()

    trials = len(manifest.attack_indices)
    return {
        "declared_attack_batches": trials,
        "selected_successful_batches": len(selected),
        "total_restart_attempts": len(attempts),
        "failed_restart_attempts": failed,
        "restart_completion": summarize_proportion(
            len(attempts) - failed,
            len(attempts),
            confidence_level=manifest.confidence_level,
        ).to_dict(),
        "exact_batch_success": summarize_proportion(
            exact,
            trials,
            confidence_level=manifest.confidence_level,
        ).to_dict(),
        "relative_l2_success": summarize_proportion(
            relative,
            trials,
            confidence_level=manifest.confidence_level,
        ).to_dict(),
        "scalar_summaries": {
            name: scalar(name, values)
            for name, values in sorted(scalar_values.items())
            if values
        },
        "release_summaries": {
            name: scalar(f"release:{name}", values)
            for name, values in sorted(release_values.items())
            if values
        },
    }


def run_modern_timeseries_defense_suite(
    manifest: ModernTimeSeriesAttackManifest,
    variants: Sequence[ModernGradientDefenseVariant],
) -> ModernDefenseSuiteReport:
    declared = tuple(variants)
    if not declared:
        raise ValueError("at least one defense variant is required")
    names = tuple(variant.name for variant in declared)
    if len(set(names)) != len(names):
        raise ValueError("defense variant names must be unique")

    _seed_everything(manifest.victim_seed)
    dataset, task, mode = _load_dataset(
        {"seed": manifest.victim_seed, "dataset": dict(manifest.dataset)}
    )
    if task != "forecasting" or mode != "timeseries":
        raise ValueError("modern defense suites require forecasting data")
    x, targets = dataset.tensors
    model = _build_model(dataset, task, dict(manifest.victim))
    _train(model, dataset, task, dict(manifest.training))

    attempts: list[ModernDefenseAttempt] = []
    by_variant: dict[str, list[ModernDefenseAttempt]] = {name: [] for name in names}
    selected: dict[str, list[ModernDefenseAttempt]] = {name: [] for name in names}
    selected_indices: dict[str, list[int]] = {name: [] for name in names}

    for batch_start in manifest.attack_indices:
        end = batch_start + manifest.attack_batch_size
        if end > len(x):
            raise ValueError(
                f"attack batch [{batch_start}, {end}) exceeds dataset size {len(x)}"
            )
        true_x = x[batch_start:end].clone()
        true_target = targets[batch_start:end].clone()
        exact = leak_gradients(model, true_x, true_target, "forecasting")
        indices = tuple(range(batch_start, end))

        for variant in declared:
            spec = _release_spec(model, variant, batch_start)
            release = release_gradients(model, exact, spec)
            attack_config = _attack_configuration(manifest, variant)
            group: list[int] = []
            for restart_seed in manifest.attack_seeds:
                started = time.perf_counter()
                try:
                    _seed_everything(restart_seed)
                    prior = _prior(tuple(true_x.shape), "timeseries", attack_config)
                    known_target = (
                        true_target
                        if attack_config.get("known_target", True)
                        else None
                    )
                    attack = ReleasedGradientInversionAttack(
                        model,
                        release,
                        spec,
                        prior,
                        task="forecasting",
                        mode="timeseries",
                        known_target=known_target,
                        target_shape=tuple(true_target.shape),
                        steps=int(attack_config.get("steps", 300)),
                        learning_rate=float(
                            attack_config.get("learning_rate", 0.05)
                        ),
                        regularization=float(
                            attack_config.get("regularization", 1e-3)
                        ),
                        match_mode=str(
                            attack_config.get("match_mode", "hybrid")
                        ),  # type: ignore[arg-type]
                        layer_weighting=str(
                            attack_config.get("layer_weighting", "parameter")
                        ),  # type: ignore[arg-type]
                        gradient_clip_norm=(
                            None
                            if attack_config.get("gradient_clip_norm") is None
                            else float(attack_config["gradient_clip_norm"])
                        ),
                        quantization_straight_through=bool(
                            attack_config.get(
                                "quantization_straight_through", True
                            )
                        ),
                        record_every=(
                            None
                            if attack_config.get("record_every") is None
                            else int(attack_config["record_every"])
                        ),
                    )
                    result = attack.run()
                    aligned = permutation_invariant_batch_metrics(
                        true_x,
                        result.reconstruction,
                        mode="timeseries",
                        tolerance=manifest.exact_tolerance,
                    ).to_dict()
                    attempt = ModernDefenseAttempt(
                        variant=variant.name,
                        batch_start=batch_start,
                        batch_indices=indices,
                        restart_seed=restart_seed,
                        status="success",
                        seconds=time.perf_counter() - started,
                        release_metadata=release.to_dict(),
                        transform_metadata=attack.transform_report.to_dict(),
                        best_objective=result.best_objective,
                        best_gradient_match=result.best_gradient_match,
                        best_step=result.best_step,
                        aligned_batch=aligned,
                    )
                except Exception as exc:
                    message = f"{type(exc).__name__}: {exc}"
                    attempt = ModernDefenseAttempt(
                        variant=variant.name,
                        batch_start=batch_start,
                        batch_indices=indices,
                        restart_seed=restart_seed,
                        status="failed",
                        seconds=time.perf_counter() - started,
                        release_metadata=release.to_dict(),
                        transform_metadata=None,
                        best_objective=None,
                        best_gradient_match=None,
                        best_step=None,
                        aligned_batch=None,
                        error_type=type(exc).__name__,
                        error_message_sha256=hashlib.sha256(
                            message.encode("utf-8")
                        ).hexdigest(),
                    )
                group.append(len(attempts))
                attempts.append(attempt)
                by_variant[variant.name].append(attempt)

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
                        float(attempts[index].best_gradient_match),
                        attempts[index].restart_seed,
                    ),
                )
                selected_indices[variant.name].append(chosen)
                selected[variant.name].append(attempts[chosen])

    summaries = {
        name: _summarize_variant(by_variant[name], selected[name], manifest)
        for name in names
    }
    required = {
        "full_exact",
        "global_clip_1",
        "symmetric_int8",
        "gaussian_noise_1e-3",
        "last_head_only",
    }
    architecture = str(manifest.victim.get("architecture", "")).lower()
    dataset_name = str(manifest.dataset.get("name", "")).lower()
    architecture_is_modern = architecture in MODERN_FORECASTING_ARCHITECTURES
    real_dataset = not dataset_name.startswith("synthetic")
    enough_batches = len(manifest.attack_indices) >= manifest.minimum_publication_batches
    enough_seeds = len(manifest.attack_seeds) >= manifest.minimum_publication_attack_seeds
    required_present = required.issubset(set(names))
    every_cell = all(
        len(selected[name]) == len(manifest.attack_indices) for name in names
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
    quality_gate = ModernDefenseSuiteQualityGate(
        architecture_is_modern=architecture_is_modern,
        real_dataset=real_dataset,
        enough_attack_batches=enough_batches,
        enough_attack_seeds=enough_seeds,
        required_conditions_present=required_present,
        every_condition_batch_has_successful_attempt=every_cell,
        no_failed_attempts=no_failures,
        publication_mode=manifest.publication_mode,
        passed=passed,
    )

    return ModernDefenseSuiteReport(
        base_manifest=manifest,
        variants=declared,
        suite_sha256=_suite_sha256(manifest, declared),
        dataset_sha256=_hash_tensors(x, targets),
        model_sha256=_hash_model(model),
        victim_class=type(model).__name__,
        trainable_parameters=sum(parameter.numel() for parameter in model.parameters()),
        attempts=tuple(attempts),
        selected_attempt_indices={
            name: tuple(indices) for name, indices in selected_indices.items()
        },
        variant_summaries=summaries,
        quality_gate=quality_gate,
        environment=benchmark_environment_manifest(),
        claim_boundary=(
            "Noise seeds are retained only for artifact reproducibility; the attack "
            "does not receive the Gaussian realization. Quantized attacks use a "
            "declared straight-through optimization surrogate. This suite measures "
            "classical white-box leakage robustness and does not establish a coherent "
            "modern-model oracle or quantum advantage."
        ),
    )
