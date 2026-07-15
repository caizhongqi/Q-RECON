from __future__ import annotations

import hashlib
import json
import math
from typing import Literal, Mapping

import torch

from qrecon.attacks import GradientReleaseSpec
from qrecon.benchmarks.channel_permutation_release import (
    ChannelPermutationReleaseQualityGate,
    ChannelPermutationReleaseReport,
    analyze_channel_permutation_releases,
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

EvaluationDType = Literal["float32", "float64"]


def _resolve_evaluation_dtype(name: str) -> tuple[EvaluationDType, torch.dtype]:
    normalized = str(name).strip().lower()
    if normalized == "float32":
        return "float32", torch.float32
    if normalized == "float64":
        return "float64", torch.float64
    raise ValueError("evaluation_dtype must be 'float32' or 'float64'")


def _data_access_locked(dataset: Mapping[str, object]) -> bool:
    name = str(dataset.get("name", "")).lower()
    if name == "gift_eval":
        return bool(str(dataset.get("revision", "")).strip())
    if name == "multivariate_csv":
        return bool(str(dataset.get("expected_file_sha256", "")).strip())
    return False


def _report_sha256(
    manifest: ModernTimeSeriesAttackManifest,
    variants: Mapping[str, GradientReleaseSpec],
    evaluation_dtype: EvaluationDType,
) -> str:
    payload = {
        "schema_version": "qrecon.channel-permutation-release-precision.v1",
        "manifest": manifest.to_dict(),
        "evaluation_dtype": evaluation_dtype,
        "variants": {
            name: spec.to_dict() for name, spec in sorted(variants.items())
        },
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def run_channel_permutation_release_benchmark_with_dtype(
    manifest: ModernTimeSeriesAttackManifest,
    variants: Mapping[str, GradientReleaseSpec],
    *,
    evaluation_dtype: str = "float64",
    tolerance: float = 1e-10,
) -> ChannelPermutationReleaseReport:
    """Audit release closure under an explicit floating-point evaluation contract.

    The model is trained according to the manifest and then the fixed trained state,
    private tensors, and all subsequent forward/backward release computations are cast
    to ``evaluation_dtype``. This does not alter the exact symmetry theorem. It makes
    the executable certificate state its numerical semantics and prevents float32
    reduction noise from being silently amplified by a discontinuous quantizer.
    """

    dtype_name, dtype = _resolve_evaluation_dtype(evaluation_dtype)
    if not math.isfinite(float(tolerance)) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    if bool(manifest.victim.get("revin", True)) and bool(
        manifest.victim.get("revin_affine", True)
    ):
        raise ValueError(
            "the channel-permutation theorem requires revin=false or "
            "revin_affine=false"
        )
    if bool(manifest.victim.get("individual_head", False)):
        raise ValueError("the channel-permutation theorem requires a shared head")
    if not variants:
        raise ValueError("at least one release variant is required")

    _seed_everything(manifest.victim_seed)
    dataset, task, mode = _load_dataset(
        {"seed": manifest.victim_seed, "dataset": dict(manifest.dataset)}
    )
    if task != "forecasting" or mode != "timeseries":
        raise ValueError("release benchmark requires forecasting data")
    source_inputs, source_targets = dataset.tensors
    if (
        source_inputs.ndim != 3
        or source_targets.ndim != 3
        or source_inputs.shape[-1] < 2
    ):
        raise ValueError("release benchmark requires multivariate data")
    dataset_sha256 = _tensor_sha256(source_inputs, source_targets)

    model = _build_model(dataset, task, dict(manifest.victim))
    _train(model, dataset, task, dict(manifest.training))
    model = model.to(dtype=dtype).eval()
    inputs = source_inputs.to(dtype=dtype)
    targets = source_targets.to(dtype=dtype)
    channels = int(inputs.shape[-1])
    permutation = tuple(range(1, channels)) + (0,)

    points = []
    for batch_start in manifest.attack_indices:
        end = batch_start + manifest.attack_batch_size
        if end > len(inputs):
            raise ValueError("attack batch exceeds dataset size")
        points.append(
            analyze_channel_permutation_releases(
                model,
                inputs[batch_start:end].clone(),
                targets[batch_start:end].clone(),
                permutation,
                variants,
                batch_start=batch_start,
                tolerance=tolerance,
            )
        )

    def scalar(values: list[float], label: str) -> dict[str, object]:
        return summarize_scalar(
            values,
            confidence_level=manifest.confidence_level,
            bootstrap_samples=manifest.bootstrap_samples,
            bootstrap_seed=manifest.bootstrap_seed,
            label=f"channel-release-{dtype_name}:{label}",
        ).to_dict()

    variant_summary: dict[str, object] = {}
    for name in sorted(variants):
        selected = [
            next(variant for variant in point.variants if variant.name == name)
            for point in points
        ]
        variant_summary[name] = {
            "certified_identical_release_distribution": summarize_proportion(
                sum(item.identical_release_distribution_certified for item in selected),
                len(selected),
                confidence_level=manifest.confidence_level,
            ).to_dict(),
            "maximum_absolute_difference": scalar(
                [
                    item.released_difference.maximum_absolute_difference
                    for item in selected
                ],
                f"{name}:maximum-absolute-difference",
            ),
            "relative_l2_difference": scalar(
                [item.released_difference.relative_l2_difference for item in selected],
                f"{name}:relative-l2-difference",
            ),
        }

    all_pass = all(point.all_release_checks_pass for point in points)
    every_nontrivial = all(point.orbit_size > 1 for point in points)
    declarations = {name: spec.to_dict() for name, spec in sorted(variants.items())}
    has_full = any(
        spec.clip_norm is None
        and spec.noise_std == 0.0
        and spec.quantization_bits is None
        and spec.visible_parameter_indices is None
        for spec in variants.values()
    )
    has_clip = any(spec.clip_norm is not None for spec in variants.values())
    has_quantization = any(
        spec.quantization_bits is not None for spec in variants.values()
    )
    has_noise = any(spec.noise_std > 0.0 for spec in variants.values())
    has_partial = any(
        spec.visible_parameter_indices is not None for spec in variants.values()
    )
    dataset_name = str(manifest.dataset.get("name", "")).lower()
    real_dataset = not dataset_name.startswith("synthetic")
    locked = _data_access_locked(manifest.dataset)
    enough = len(points) >= manifest.minimum_publication_batches
    gate_passed = (
        manifest.publication_mode
        and real_dataset
        and locked
        and enough
        and every_nontrivial
        and has_full
        and has_clip
        and has_quantization
        and has_noise
        and has_partial
        and all_pass
    )
    gate = ChannelPermutationReleaseQualityGate(
        real_dataset=real_dataset,
        data_access_locked=locked,
        enough_attack_batches=enough,
        multivariate=True,
        every_orbit_nontrivial=every_nontrivial,
        full_gradient_variant_present=has_full,
        clipping_variant_present=has_clip,
        quantization_variant_present=has_quantization,
        gaussian_noise_variant_present=has_noise,
        partial_visibility_variant_present=has_partial,
        every_release_check_passed=all_pass,
        publication_mode=manifest.publication_mode,
        passed=gate_passed,
    )
    summary = {
        "evaluation_dtype": dtype_name,
        "numerical_tolerance": float(tolerance),
        "points": len(points),
        "all_release_checks": summarize_proportion(
            sum(point.all_release_checks_pass for point in points),
            len(points),
            confidence_level=manifest.confidence_level,
        ).to_dict(),
        "orbit_size": scalar(
            [float(point.orbit_size) for point in points], "orbit-size"
        ),
        "uniform_exact_ordered_recovery_ceiling": scalar(
            [point.uniform_exact_ordered_recovery_ceiling for point in points],
            "ordered-recovery-ceiling",
        ),
        "raw_gradient_maximum_absolute_difference": scalar(
            [point.raw_gradient_difference.maximum_absolute_difference for point in points],
            "raw-gradient-maximum-absolute-difference",
        ),
        "variants": variant_summary,
    }
    environment = benchmark_environment_manifest()
    environment["release_evaluation_dtype"] = dtype_name

    return ChannelPermutationReleaseReport(
        manifest=manifest,
        report_sha256=_report_sha256(manifest, variants, dtype_name),
        dataset_sha256=dataset_sha256,
        model_sha256=_model_sha256(model),
        victim_class=type(model).__name__,
        trainable_parameters=sum(parameter.numel() for parameter in model.parameters()),
        variant_declarations=declarations,
        points=tuple(points),
        summary=summary,
        quality_gate=gate,
        environment=environment,
        theorem=(
            "If two private objects induce the same exact gradient, deterministic "
            "postprocessing and data-independent randomized release kernels induce "
            "identical observations. The executable certificate evaluates the fixed "
            f"trained model in {dtype_name}."
        ),
        claim_boundary=(
            "Floating-point equality is an executable numerical witness, not the exact "
            "algebraic proof. The evaluation dtype and tolerance are part of the report; "
            "they must not be changed after observing outcomes. Semantic channel labels "
            "or channel-indexed parameters define a different observation channel."
        ),
    )
