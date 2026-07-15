from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

import numpy as np
import torch

from qrecon.attacks import capture_final_linear_input, find_last_biased_linear
from qrecon.benchmarks.modern_timeseries_reconstruction import (
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
from qrecon.theory import (
    construct_known_target_rotation_collision,
    known_target_orbit_report,
    target_stabilizer_reflection,
)


@dataclass(frozen=True)
class FinalHeadOrbitPoint:
    batch_start: int
    effective_samples: int
    feature_dimension: int
    output_dimension: int
    target_constraint_rank: int
    orthogonal_complement_dimension: int
    projected_input_rank: int
    stabilizer_group_dimension: int
    continuous_orbit_dimension: int
    has_nontrivial_collision: bool
    has_continuous_family: bool
    collision_kind: str | None
    collision_input_displacement: float | None
    collision_statistic_error: float | None
    collision_actual_weight_gradient_error: float | None
    collision_actual_bias_gradient_error: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FinalHeadOrbitQualityGate:
    real_dataset: bool
    data_access_locked: bool
    enough_attack_batches: bool
    every_point_certified: bool
    publication_mode: bool
    passed: bool

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)


@dataclass(frozen=True)
class FinalHeadOrbitBenchmarkReport:
    manifest: ModernTimeSeriesAttackManifest
    report_sha256: str
    dataset_sha256: str
    model_sha256: str
    victim_class: str
    head_module_name: str
    points: tuple[FinalHeadOrbitPoint, ...]
    summary: dict[str, object]
    quality_gate: FinalHeadOrbitQualityGate
    environment: dict[str, object]
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest": self.manifest.to_dict(),
            "report_sha256": self.report_sha256,
            "dataset_sha256": self.dataset_sha256,
            "model_sha256": self.model_sha256,
            "victim_class": self.victim_class,
            "head_module_name": self.head_module_name,
            "points": [point.to_dict() for point in self.points],
            "summary": self.summary,
            "quality_gate": self.quality_gate.to_dict(),
            "environment": self.environment,
            "claim_boundary": self.claim_boundary,
        }


def _effective_target_rows(targets: torch.Tensor) -> torch.Tensor:
    """Match forecasting targets to the leading axes seen by the shared final head."""

    if targets.ndim == 2:
        return targets.reshape(targets.shape[0], targets.shape[1])
    if targets.ndim == 3:
        # Public forecasting convention: [batch, horizon, channels]. The shared
        # head is applied to [batch, channels, feature], so channels are folded
        # into the effective-sample axis in the same order.
        return targets.transpose(1, 2).reshape(-1, targets.shape[1])
    raise ValueError(
        "forecasting targets must be [batch,horizon] or [batch,horizon,channels]"
    )


def _mse_head_gradients(
    features: np.ndarray,
    targets: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    predictions = features @ weight.T + bias[None, :]
    residual = predictions - targets
    factor = 2.0 / float(features.shape[0] * targets.shape[1])
    return factor * residual.T @ features, factor * residual.sum(axis=0)


def _statistic_error(
    left: np.ndarray,
    right: np.ndarray,
    targets: np.ndarray,
) -> float:
    return max(
        float(np.max(np.abs(a - b)))
        for a, b in (
            (left.T @ left, right.T @ right),
            (left.sum(axis=0), right.sum(axis=0)),
            (targets.T @ left, targets.T @ right),
        )
    )


def _data_access_locked(dataset: dict[str, object]) -> bool:
    name = str(dataset.get("name", "")).lower()
    if name == "gift_eval":
        return bool(str(dataset.get("revision", "")).strip())
    if name == "multivariate_csv":
        return bool(str(dataset.get("expected_file_sha256", "")).strip())
    return False


def _report_sha256(manifest: ModernTimeSeriesAttackManifest) -> str:
    payload = {
        "schema_version": "qrecon.final-head-orbit.v1",
        "manifest": manifest.to_dict(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def analyze_final_head_orbit(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    batch_start: int = 0,
    collision_angle: float = 0.37,
) -> FinalHeadOrbitPoint:
    """Certify the known-target biased-linear gradient orbit at the final head.

    A complement of dimension at least two admits a continuous rotation. A
    one-dimensional complement admits a discrete reflection, which is still a
    valid non-identifiability witness but must not be called a continuous family.
    """

    if not math.isfinite(float(collision_angle)) or collision_angle == 0.0:
        raise ValueError("collision_angle must be finite and nonzero")
    captured = capture_final_linear_input(model, inputs).detach()
    features = captured.reshape(-1, captured.shape[-1]).cpu().double().numpy()
    target_rows = _effective_target_rows(targets).detach().cpu().double().numpy()
    if features.shape[0] != target_rows.shape[0]:
        raise ValueError(
            "final-head effective sample count does not match forecasting targets: "
            f"{features.shape[0]} versus {target_rows.shape[0]}"
        )

    orbit = known_target_orbit_report(features, target_rows)
    displacement: float | None = None
    statistic_error: float | None = None
    actual_weight_error: float | None = None
    actual_bias_error: float | None = None
    collision_kind: str | None = None

    _, head = find_last_biased_linear(model)
    if head.bias is None:
        raise ValueError("final head must have a bias")
    weight = head.weight.detach().cpu().double().numpy()
    bias = head.bias.detach().cpu().double().numpy()

    if orbit.has_nontrivial_collision:
        if orbit.orthogonal_complement_dimension >= 2:
            collision = construct_known_target_rotation_collision(
                features,
                target_rows,
                weight,
                bias,
                angle=float(collision_angle),
            )
            transformed = collision.transformed_inputs
            displacement = collision.input_displacement
            statistic_error = collision.statistic_error
            collision_kind = "continuous_rotation"
        else:
            transform = target_stabilizer_reflection(target_rows, axis=0)
            transformed = transform @ features
            displacement = float(np.linalg.norm(transformed - features))
            statistic_error = _statistic_error(features, transformed, target_rows)
            collision_kind = "discrete_reflection"

        clean_weight, clean_bias = _mse_head_gradients(
            features, target_rows, weight, bias
        )
        alternative_weight, alternative_bias = _mse_head_gradients(
            transformed, target_rows, weight, bias
        )
        actual_weight_error = float(
            np.linalg.norm(clean_weight - alternative_weight)
        )
        actual_bias_error = float(np.linalg.norm(clean_bias - alternative_bias))

    return FinalHeadOrbitPoint(
        batch_start=int(batch_start),
        effective_samples=int(features.shape[0]),
        feature_dimension=int(features.shape[1]),
        output_dimension=int(target_rows.shape[1]),
        target_constraint_rank=orbit.target_constraint_rank,
        orthogonal_complement_dimension=orbit.orthogonal_complement_dimension,
        projected_input_rank=orbit.projected_input_rank,
        stabilizer_group_dimension=orbit.stabilizer_group_dimension,
        continuous_orbit_dimension=orbit.continuous_orbit_dimension,
        has_nontrivial_collision=orbit.has_nontrivial_collision,
        has_continuous_family=orbit.has_continuous_family,
        collision_kind=collision_kind,
        collision_input_displacement=displacement,
        collision_statistic_error=statistic_error,
        collision_actual_weight_gradient_error=actual_weight_error,
        collision_actual_bias_gradient_error=actual_bias_error,
    )


def run_final_head_orbit_benchmark(
    manifest: ModernTimeSeriesAttackManifest,
) -> FinalHeadOrbitBenchmarkReport:
    _seed_everything(manifest.victim_seed)
    dataset, task, mode = _load_dataset(
        {"seed": manifest.victim_seed, "dataset": dict(manifest.dataset)}
    )
    if task != "forecasting" or mode != "timeseries":
        raise ValueError("final-head orbit benchmark requires forecasting data")
    inputs, targets = dataset.tensors
    model = _build_model(dataset, task, dict(manifest.victim))
    _train(model, dataset, task, dict(manifest.training))
    module_name, _ = find_last_biased_linear(model)

    points: list[FinalHeadOrbitPoint] = []
    for batch_start in manifest.attack_indices:
        end = batch_start + manifest.attack_batch_size
        if end > len(inputs):
            raise ValueError("attack batch exceeds dataset size")
        points.append(
            analyze_final_head_orbit(
                model,
                inputs[batch_start:end].clone(),
                targets[batch_start:end].clone(),
                batch_start=batch_start,
            )
        )

    collisions = sum(point.has_nontrivial_collision for point in points)
    continuous = sum(point.has_continuous_family for point in points)
    gradient_errors = [
        max(
            float(point.collision_actual_weight_gradient_error or 0.0),
            float(point.collision_actual_bias_gradient_error or 0.0),
        )
        for point in points
        if point.has_nontrivial_collision
    ]

    def scalar(values: list[float], label: str) -> dict[str, object] | None:
        if not values:
            return None
        return summarize_scalar(
            values,
            confidence_level=manifest.confidence_level,
            bootstrap_samples=manifest.bootstrap_samples,
            bootstrap_seed=manifest.bootstrap_seed,
            label=f"final-head-orbit:{label}",
        ).to_dict()

    summary: dict[str, object] = {
        "points": len(points),
        "nontrivial_collision": summarize_proportion(
            collisions,
            len(points),
            confidence_level=manifest.confidence_level,
        ).to_dict(),
        "continuous_family": summarize_proportion(
            continuous,
            len(points),
            confidence_level=manifest.confidence_level,
        ).to_dict(),
        "effective_samples": scalar(
            [float(point.effective_samples) for point in points],
            "effective_samples",
        ),
        "orthogonal_complement_dimension": scalar(
            [float(point.orthogonal_complement_dimension) for point in points],
            "orthogonal_complement_dimension",
        ),
        "continuous_orbit_dimension": scalar(
            [float(point.continuous_orbit_dimension) for point in points],
            "continuous_orbit_dimension",
        ),
        "collision_gradient_error": scalar(
            gradient_errors,
            "collision_gradient_error",
        ),
    }

    dataset_name = str(manifest.dataset.get("name", "")).lower()
    real_dataset = not dataset_name.startswith("synthetic")
    data_locked = _data_access_locked(dict(manifest.dataset))
    enough_batches = len(points) >= manifest.minimum_publication_batches
    every_certified = all(
        (not point.has_nontrivial_collision)
        or (
            point.collision_kind is not None
            and point.collision_statistic_error is not None
            and point.collision_actual_weight_gradient_error is not None
            and point.collision_actual_bias_gradient_error is not None
        )
        for point in points
    )
    passed = (
        manifest.publication_mode
        and real_dataset
        and data_locked
        and enough_batches
        and every_certified
    )
    gate = FinalHeadOrbitQualityGate(
        real_dataset=real_dataset,
        data_access_locked=data_locked,
        enough_attack_batches=enough_batches,
        every_point_certified=every_certified,
        publication_mode=manifest.publication_mode,
        passed=passed,
    )

    return FinalHeadOrbitBenchmarkReport(
        manifest=manifest,
        report_sha256=_report_sha256(manifest),
        dataset_sha256=_tensor_sha256(inputs, targets),
        model_sha256=_model_sha256(model),
        victim_class=type(model).__name__,
        head_module_name=module_name,
        points=tuple(points),
        summary=summary,
        quality_gate=gate,
        environment=benchmark_environment_manifest(),
        claim_boundary=(
            "This is a final-head representation-level result. A nontrivial target-"
            "stabilizer orbit proves that the released biased-linear MSE head gradients "
            "do not identify the original effective representation matrix. It does not "
            "by itself prove that every transformed representation is reachable from a "
            "valid raw time series through the fixed encoder."
        ),
    )
