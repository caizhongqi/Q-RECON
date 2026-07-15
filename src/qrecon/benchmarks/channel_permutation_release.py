from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import torch

from qrecon.attacks import (
    GradientReleaseSpec,
    leak_gradients,
    release_gradients,
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
from qrecon.theory.channel_permutation import (
    apply_channel_permutation,
    tensor_channel_permutation_fibre_bound,
    validate_channel_permutation,
)


@dataclass(frozen=True)
class GradientTupleDifference:
    maximum_absolute_difference: float
    relative_l2_difference: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class ChannelPermutationReleaseVariantPoint:
    name: str
    release_spec: dict[str, object]
    released_difference: GradientTupleDifference
    visible_parameter_indices_match: bool
    visible_parameter_names_match: bool
    quantization_scale_match: bool
    shared_randomness_pathwise_match: bool
    identical_release_distribution_certified: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "release_spec": self.release_spec,
            "released_difference": self.released_difference.to_dict(),
            "visible_parameter_indices_match": self.visible_parameter_indices_match,
            "visible_parameter_names_match": self.visible_parameter_names_match,
            "quantization_scale_match": self.quantization_scale_match,
            "shared_randomness_pathwise_match": self.shared_randomness_pathwise_match,
            "identical_release_distribution_certified": (
                self.identical_release_distribution_certified
            ),
        }


@dataclass(frozen=True)
class ChannelPermutationReleasePoint:
    batch_start: int
    permutation: tuple[int, ...]
    orbit_size: int
    uniform_exact_ordered_recovery_ceiling: float
    raw_gradient_difference: GradientTupleDifference
    variants: tuple[ChannelPermutationReleaseVariantPoint, ...]
    all_release_checks_pass: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "batch_start": self.batch_start,
            "permutation": list(self.permutation),
            "orbit_size": self.orbit_size,
            "uniform_exact_ordered_recovery_ceiling": (
                self.uniform_exact_ordered_recovery_ceiling
            ),
            "raw_gradient_difference": self.raw_gradient_difference.to_dict(),
            "variants": [variant.to_dict() for variant in self.variants],
            "all_release_checks_pass": self.all_release_checks_pass,
        }


@dataclass(frozen=True)
class ChannelPermutationReleaseQualityGate:
    real_dataset: bool
    data_access_locked: bool
    enough_attack_batches: bool
    multivariate: bool
    every_orbit_nontrivial: bool
    full_gradient_variant_present: bool
    clipping_variant_present: bool
    quantization_variant_present: bool
    gaussian_noise_variant_present: bool
    partial_visibility_variant_present: bool
    every_release_check_passed: bool
    publication_mode: bool
    passed: bool

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)


@dataclass(frozen=True)
class ChannelPermutationReleaseReport:
    manifest: ModernTimeSeriesAttackManifest
    report_sha256: str
    dataset_sha256: str
    model_sha256: str
    victim_class: str
    trainable_parameters: int
    variant_declarations: dict[str, dict[str, object]]
    points: tuple[ChannelPermutationReleasePoint, ...]
    summary: dict[str, object]
    quality_gate: ChannelPermutationReleaseQualityGate
    environment: dict[str, object]
    theorem: str
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest": self.manifest.to_dict(),
            "report_sha256": self.report_sha256,
            "dataset_sha256": self.dataset_sha256,
            "model_sha256": self.model_sha256,
            "victim_class": self.victim_class,
            "trainable_parameters": self.trainable_parameters,
            "variant_declarations": self.variant_declarations,
            "points": [point.to_dict() for point in self.points],
            "summary": self.summary,
            "quality_gate": self.quality_gate.to_dict(),
            "environment": self.environment,
            "theorem": self.theorem,
            "claim_boundary": self.claim_boundary,
        }


def _tuple_difference(
    left: Sequence[torch.Tensor], right: Sequence[torch.Tensor]
) -> GradientTupleDifference:
    if len(left) != len(right):
        raise ValueError("gradient tuples must have equal lengths")
    maximum = 0.0
    numerator = 0.0
    denominator = 0.0
    for first, second in zip(left, right):
        if first.shape != second.shape:
            raise ValueError("corresponding gradient tensors must have equal shapes")
        first_value = first.detach().double()
        second_value = second.detach().double()
        difference = first_value - second_value
        maximum = max(maximum, float(difference.abs().max()))
        numerator += float(difference.square().sum())
        denominator += float(first_value.square().sum())
    relative = math.sqrt(numerator) / max(math.sqrt(denominator), 1e-30)
    return GradientTupleDifference(maximum, relative)


def _close(left: float | None, right: float | None, tolerance: float) -> bool:
    if left is None or right is None:
        return left is right
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)


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
) -> str:
    payload = {
        "schema_version": "qrecon.channel-permutation-release.v1",
        "manifest": manifest.to_dict(),
        "variants": {
            name: spec.to_dict() for name, spec in sorted(variants.items())
        },
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def analyze_channel_permutation_releases(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    permutation: Sequence[int],
    variants: Mapping[str, GradientReleaseSpec],
    *,
    batch_start: int = 0,
    tolerance: float = 1e-5,
) -> ChannelPermutationReleasePoint:
    """Certify release-channel closure of one channel-permutation gradient fibre.

    Exact full-gradient equality implies equality after every deterministic
    postprocessing map.  A randomized mechanism whose coins are independent of the
    private orbit member also has identical conditional output distributions.  The
    implementation couples randomized mechanisms with the same public audit seed;
    pathwise equality under that coupling is an executable witness of the equal-law
    theorem, not an assumption that the attacker observes the noise realization.
    """

    if inputs.ndim != 3 or targets.ndim != 3:
        raise ValueError("release analysis requires multivariate three-dimensional tensors")
    if inputs.shape[0] != targets.shape[0] or inputs.shape[-1] != targets.shape[-1]:
        raise ValueError("input and target batch/channel dimensions must match")
    if not variants:
        raise ValueError("at least one release variant is required")
    if not math.isfinite(float(tolerance)) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")

    values = validate_channel_permutation(permutation, int(inputs.shape[-1]))
    permuted_inputs = apply_channel_permutation(inputs, values)
    permuted_targets = apply_channel_permutation(targets, values)
    exact = leak_gradients(model, inputs, targets, "forecasting")
    permuted_exact = leak_gradients(
        model, permuted_inputs, permuted_targets, "forecasting"
    )
    exact_difference = _tuple_difference(exact, permuted_exact)
    exact_match = (
        exact_difference.maximum_absolute_difference <= tolerance
        and exact_difference.relative_l2_difference <= tolerance
    )

    variant_points: list[ChannelPermutationReleaseVariantPoint] = []
    for name, spec in sorted(variants.items()):
        left = release_gradients(model, exact, spec)
        right = release_gradients(model, permuted_exact, spec)
        difference = _tuple_difference(left.gradients, right.gradients)
        index_match = left.visible_parameter_indices == right.visible_parameter_indices
        name_match = left.visible_parameter_names == right.visible_parameter_names
        scale_match = _close(
            left.quantization_scale, right.quantization_scale, tolerance
        )
        pathwise = (
            exact_match
            and difference.maximum_absolute_difference <= tolerance
            and difference.relative_l2_difference <= tolerance
            and index_match
            and name_match
            and scale_match
        )
        variant_points.append(
            ChannelPermutationReleaseVariantPoint(
                name=str(name),
                release_spec=spec.to_dict(),
                released_difference=difference,
                visible_parameter_indices_match=index_match,
                visible_parameter_names_match=name_match,
                quantization_scale_match=scale_match,
                shared_randomness_pathwise_match=pathwise,
                identical_release_distribution_certified=pathwise,
            )
        )

    fibre = tensor_channel_permutation_fibre_bound(inputs, targets)
    return ChannelPermutationReleasePoint(
        batch_start=int(batch_start),
        permutation=values,
        orbit_size=fibre.orbit_size,
        uniform_exact_ordered_recovery_ceiling=(
            fibre.uniform_exact_ordered_recovery_ceiling
        ),
        raw_gradient_difference=exact_difference,
        variants=tuple(variant_points),
        all_release_checks_pass=exact_match
        and all(
            variant.identical_release_distribution_certified
            for variant in variant_points
        ),
    )


def run_channel_permutation_release_benchmark(
    manifest: ModernTimeSeriesAttackManifest,
    variants: Mapping[str, GradientReleaseSpec],
    *,
    tolerance: float = 1e-5,
) -> ChannelPermutationReleaseReport:
    """Run a real-data release-closure study for an anonymous-channel forecaster."""

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
    inputs, targets = dataset.tensors
    if inputs.ndim != 3 or targets.ndim != 3 or inputs.shape[-1] < 2:
        raise ValueError("release benchmark requires multivariate data")

    model = _build_model(dataset, task, dict(manifest.victim))
    _train(model, dataset, task, dict(manifest.training))
    model.eval()
    channels = int(inputs.shape[-1])
    permutation = tuple(range(1, channels)) + (0,)

    points: list[ChannelPermutationReleasePoint] = []
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
            label=f"channel-release:{label}",
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
                [item.released_difference.maximum_absolute_difference for item in selected],
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

    return ChannelPermutationReleaseReport(
        manifest=manifest,
        report_sha256=_report_sha256(manifest, variants),
        dataset_sha256=_tensor_sha256(inputs, targets),
        model_sha256=_model_sha256(model),
        victim_class=type(model).__name__,
        trainable_parameters=sum(parameter.numel() for parameter in model.parameters()),
        variant_declarations=declarations,
        points=tuple(points),
        summary=summary,
        quality_gate=gate,
        environment=benchmark_environment_manifest(),
        theorem=(
            "If two private objects induce the same exact gradient, every deterministic "
            "postprocessing map and every data-independent randomized release kernel "
            "induces identical observations. Clipping, fixed or adaptive quantization, "
            "fixed parameter visibility, and independent additive Gaussian noise cannot "
            "split a channel-permutation fibre. Repeating such releases does not improve "
            "the Bayes exact-order ceiling."
        ),
        claim_boundary=(
            "The result assumes the release mechanism depends on the private batch only "
            "through the invariant gradient and uses randomness independent of the orbit "
            "member. Data-dependent side information, semantic channel labels, or "
            "channel-indexed parameters define a different observation channel."
        ),
    )
