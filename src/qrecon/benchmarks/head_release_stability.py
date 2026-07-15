from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Sequence

import torch

from qrecon.attacks import (
    GradientRelease,
    GradientReleaseSpec,
    last_biased_linear_parameter_indices,
    leak_gradients,
    release_gradients,
)
from qrecon.benchmarks.modern_timeseries_defense_suite import (
    ModernGradientDefenseVariant,
    _release_spec,
)
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
    HeadPerturbationNormBounds,
    certify_head_representation_perturbation,
    combine_head_perturbation_bounds,
    common_scale_invariance_error,
    gaussian_head_bounds,
    recover_head_representation,
    uniform_quantization_head_bounds,
)


@dataclass(frozen=True)
class HeadReleaseStabilityPoint:
    variant: str
    batch_start: int
    effective_samples: int
    head_visible: bool
    certifiable: bool
    common_scale_invariance_l2_error: float | None
    actual_representation_l2_error: float | None
    posterior_l2_error_bound: float | None
    certificate_sound: bool | None
    bias_error_ratio: float | None
    weight_error_normalized: float | None
    quantization_saturated: bool
    release_metadata: dict[str, object]
    skip_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class HeadReleaseStabilityReport:
    base_manifest: ModernTimeSeriesAttackManifest
    variants: tuple[ModernGradientDefenseVariant, ...]
    report_sha256: str
    dataset_sha256: str
    model_sha256: str
    victim_class: str
    points: tuple[HeadReleaseStabilityPoint, ...]
    summary: dict[str, object]
    environment: dict[str, object]
    theorem_scope: str

    def to_dict(self) -> dict[str, object]:
        return {
            "base_manifest": self.base_manifest.to_dict(),
            "variants": [variant.to_dict() for variant in self.variants],
            "report_sha256": self.report_sha256,
            "dataset_sha256": self.dataset_sha256,
            "model_sha256": self.model_sha256,
            "victim_class": self.victim_class,
            "points": [point.to_dict() for point in self.points],
            "summary": self.summary,
            "environment": self.environment,
            "theorem_scope": self.theorem_scope,
        }


def _report_sha256(
    manifest: ModernTimeSeriesAttackManifest,
    variants: Sequence[ModernGradientDefenseVariant],
    failure_probability: float,
) -> str:
    payload = {
        "schema_version": "qrecon.head-release-stability.v1",
        "base_manifest": manifest.to_dict(),
        "variants": [variant.to_dict() for variant in variants],
        "failure_probability": float(failure_probability),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _visible_head_pair(
    release: GradientRelease,
    weight_index: int,
    bias_index: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    positions = {
        parameter_index: position
        for position, parameter_index in enumerate(release.visible_parameter_indices)
    }
    if weight_index not in positions or bias_index not in positions:
        return None
    return (
        release.gradients[positions[weight_index]],
        release.gradients[positions[bias_index]],
    )


def _perturbation_bounds(
    release: GradientRelease,
    output_dimension: int,
    feature_dimension: int,
    failure_probability: float,
) -> HeadPerturbationNormBounds | None:
    components: list[HeadPerturbationNormBounds] = []
    if release.noise_std > 0.0:
        components.append(
            gaussian_head_bounds(
                output_dimension,
                feature_dimension,
                release.noise_std,
                failure_probability,
            )
        )
    if release.quantization_bits is not None:
        if release.quantized_saturation_count > 0 or release.quantization_scale is None:
            return None
        components.append(
            uniform_quantization_head_bounds(
                output_dimension,
                feature_dimension,
                release.quantization_scale,
            )
        )
    if not components:
        return HeadPerturbationNormBounds(
            bias_l2=0.0,
            weight_frobenius=0.0,
            provenance="exact gradients after a common global clipping scale",
        )
    return combine_head_perturbation_bounds(
        *components,
        provenance="release noise and no-saturation quantization",
    )


def run_head_release_stability_audit(
    manifest: ModernTimeSeriesAttackManifest,
    variants: Sequence[ModernGradientDefenseVariant],
    *,
    failure_probability: float = 0.01,
) -> HeadReleaseStabilityReport:
    declared = tuple(variants)
    if not declared:
        raise ValueError("at least one defense variant is required")
    if not math.isfinite(float(failure_probability)) or not (
        0.0 < float(failure_probability) < 1.0
    ):
        raise ValueError("failure_probability must lie strictly between zero and one")

    _seed_everything(manifest.victim_seed)
    dataset, task, mode = _load_dataset(
        {"seed": manifest.victim_seed, "dataset": dict(manifest.dataset)}
    )
    if task != "forecasting" or mode != "timeseries":
        raise ValueError("head-release stability requires forecasting data")
    inputs, targets = dataset.tensors
    model = _build_model(dataset, task, dict(manifest.victim))
    _train(model, dataset, task, dict(manifest.training))
    weight_index, bias_index = last_biased_linear_parameter_indices(model)

    points: list[HeadReleaseStabilityPoint] = []
    for batch_start in manifest.attack_indices:
        end = batch_start + manifest.attack_batch_size
        if end > len(inputs):
            raise ValueError("attack batch exceeds dataset size")
        true_x = inputs[batch_start:end].clone()
        true_target = targets[batch_start:end].clone()
        channels = int(true_x.shape[2]) if true_x.ndim == 3 else 1
        effective_samples = int(true_x.shape[0]) * channels
        exact = leak_gradients(model, true_x, true_target, "forecasting")
        clean_weight = exact[weight_index]
        clean_bias = exact[bias_index]

        for variant in declared:
            spec: GradientReleaseSpec = _release_spec(model, variant, batch_start)
            release = release_gradients(model, exact, spec)
            pair = _visible_head_pair(release, weight_index, bias_index)
            metadata = release.to_dict()
            if pair is None:
                points.append(
                    HeadReleaseStabilityPoint(
                        variant=variant.name,
                        batch_start=batch_start,
                        effective_samples=effective_samples,
                        head_visible=False,
                        certifiable=False,
                        common_scale_invariance_l2_error=None,
                        actual_representation_l2_error=None,
                        posterior_l2_error_bound=None,
                        certificate_sound=None,
                        bias_error_ratio=None,
                        weight_error_normalized=None,
                        quantization_saturated=(release.quantized_saturation_count > 0),
                        release_metadata=metadata,
                        skip_reason="final head weight and bias gradients are not both visible",
                    )
                )
                continue
            if effective_samples != 1:
                points.append(
                    HeadReleaseStabilityPoint(
                        variant=variant.name,
                        batch_start=batch_start,
                        effective_samples=effective_samples,
                        head_visible=True,
                        certifiable=False,
                        common_scale_invariance_l2_error=None,
                        actual_representation_l2_error=None,
                        posterior_l2_error_bound=None,
                        certificate_sound=None,
                        bias_error_ratio=None,
                        weight_error_normalized=None,
                        quantization_saturated=(release.quantized_saturation_count > 0),
                        release_metadata=metadata,
                        skip_reason="rank-one ratio theorem requires one effective sample",
                    )
                )
                continue

            observed_weight, observed_bias = pair
            clean_feature = recover_head_representation(clean_weight, clean_bias)
            observed_feature = recover_head_representation(observed_weight, observed_bias)
            actual_error = float((observed_feature - clean_feature).norm())
            scale_error = common_scale_invariance_error(
                clean_weight,
                clean_bias,
                release.clipping_factor,
            )
            bounds = _perturbation_bounds(
                release,
                int(clean_bias.numel()),
                int(clean_feature.numel()),
                float(failure_probability),
            )
            if bounds is None:
                points.append(
                    HeadReleaseStabilityPoint(
                        variant=variant.name,
                        batch_start=batch_start,
                        effective_samples=1,
                        head_visible=True,
                        certifiable=False,
                        common_scale_invariance_l2_error=scale_error,
                        actual_representation_l2_error=actual_error,
                        posterior_l2_error_bound=None,
                        certificate_sound=None,
                        bias_error_ratio=None,
                        weight_error_normalized=None,
                        quantization_saturated=True,
                        release_metadata=metadata,
                        skip_reason="quantization saturation invalidates the uniform bound",
                    )
                )
                continue

            certificate = certify_head_representation_perturbation(
                observed_weight,
                observed_bias,
                bounds,
            )
            posterior = certificate.posterior_l2_error_bound
            points.append(
                HeadReleaseStabilityPoint(
                    variant=variant.name,
                    batch_start=batch_start,
                    effective_samples=1,
                    head_visible=True,
                    certifiable=certificate.certifiable,
                    common_scale_invariance_l2_error=scale_error,
                    actual_representation_l2_error=actual_error,
                    posterior_l2_error_bound=posterior,
                    certificate_sound=(
                        None
                        if posterior is None
                        else actual_error <= posterior + 1e-9
                    ),
                    bias_error_ratio=certificate.bias_error_ratio,
                    weight_error_normalized=certificate.weight_error_normalized,
                    quantization_saturated=False,
                    release_metadata=metadata,
                )
            )

    certified = [point for point in points if point.certifiable]
    sound = sum(point.certificate_sound is True for point in certified)
    actual_errors = [
        float(point.actual_representation_l2_error)
        for point in points
        if point.actual_representation_l2_error is not None
    ]
    posterior_bounds = [
        float(point.posterior_l2_error_bound)
        for point in certified
        if point.posterior_l2_error_bound is not None
    ]
    scale_errors = [
        float(point.common_scale_invariance_l2_error)
        for point in points
        if point.common_scale_invariance_l2_error is not None
    ]

    def scalar(values: list[float], label: str) -> dict[str, object] | None:
        if not values:
            return None
        return summarize_scalar(
            values,
            confidence_level=manifest.confidence_level,
            bootstrap_samples=manifest.bootstrap_samples,
            bootstrap_seed=manifest.bootstrap_seed,
            label=f"head-release:{label}",
        ).to_dict()

    summary: dict[str, object] = {
        "points": len(points),
        "head_visible": summarize_proportion(
            sum(point.head_visible for point in points),
            len(points),
            confidence_level=manifest.confidence_level,
        ).to_dict(),
        "certifiable": summarize_proportion(
            len(certified),
            len(points),
            confidence_level=manifest.confidence_level,
        ).to_dict(),
        "certificate_sound": (
            None
            if not certified
            else summarize_proportion(
                sound,
                len(certified),
                confidence_level=manifest.confidence_level,
            ).to_dict()
        ),
        "actual_representation_l2_error": scalar(actual_errors, "actual_error"),
        "posterior_l2_error_bound": scalar(posterior_bounds, "posterior_bound"),
        "common_scale_invariance_l2_error": scalar(scale_errors, "scale_error"),
    }

    return HeadReleaseStabilityReport(
        base_manifest=manifest,
        variants=declared,
        report_sha256=_report_sha256(manifest, declared, failure_probability),
        dataset_sha256=_tensor_sha256(inputs, targets),
        model_sha256=_model_sha256(model),
        victim_class=type(model).__name__,
        points=tuple(points),
        summary=summary,
        environment=benchmark_environment_manifest(),
        theorem_scope=(
            "The certificate applies only when the final biased Linear receives one "
            "effective sample and both final weight and bias gradients are visible. "
            "It certifies hidden-representation stability, not exact input recovery."
        ),
    )
