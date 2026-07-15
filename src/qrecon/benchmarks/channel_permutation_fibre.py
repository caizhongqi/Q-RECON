from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Sequence

import torch

from qrecon.experiment import _build_model, _load_dataset, _seed_everything, _train
from qrecon.theory.channel_permutation import (
    ChannelPermutationFibreBound,
    channel_permutation_fibre_bound,
)

from .modern_timeseries_reconstruction import ModernTimeSeriesAttackManifest
from .statistics import (
    ProportionSummary,
    ScalarSummary,
    benchmark_environment_manifest,
    summarize_proportion,
    summarize_scalar,
)


@dataclass(frozen=True)
class ChannelPermutationGeneratorCheck:
    left_channel: int
    right_channel: int
    output_equivariance_max_abs_error: float
    gradient_max_abs_difference: float
    gradient_relative_l2_difference: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class ChannelPermutationFibrePoint:
    batch_start: int
    channels: int
    orbit_size: int
    uniform_exact_ordered_recovery_ceiling: float
    multiplicities: tuple[int, ...]
    generator_count: int
    maximum_output_equivariance_error: float
    maximum_gradient_invariance_error: float
    maximum_gradient_relative_error: float
    all_generator_checks_pass: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["multiplicities"] = list(self.multiplicities)
        return payload


@dataclass(frozen=True)
class ChannelPermutationFibreSummary:
    points: int
    nontrivial_orbit: ProportionSummary
    generator_checks: ProportionSummary
    orbit_size: ScalarSummary
    uniform_exact_ordered_recovery_ceiling: ScalarSummary
    maximum_output_equivariance_error: ScalarSummary
    maximum_gradient_invariance_error: ScalarSummary

    def to_dict(self) -> dict[str, object]:
        return {
            "points": self.points,
            "nontrivial_orbit": self.nontrivial_orbit.to_dict(),
            "generator_checks": self.generator_checks.to_dict(),
            "orbit_size": self.orbit_size.to_dict(),
            "uniform_exact_ordered_recovery_ceiling": (
                self.uniform_exact_ordered_recovery_ceiling.to_dict()
            ),
            "maximum_output_equivariance_error": (
                self.maximum_output_equivariance_error.to_dict()
            ),
            "maximum_gradient_invariance_error": (
                self.maximum_gradient_invariance_error.to_dict()
            ),
        }


@dataclass(frozen=True)
class ChannelPermutationFibreQualityGate:
    real_dataset: bool
    data_access_locked: bool
    multivariate: bool
    enough_attack_batches: bool
    every_orbit_nontrivial: bool
    all_generator_checks_pass: bool
    publication_mode: bool

    @property
    def passed(self) -> bool:
        return all(asdict(self).values())

    def to_dict(self) -> dict[str, bool]:
        payload = asdict(self)
        payload["passed"] = self.passed
        return payload


@dataclass(frozen=True)
class ChannelPermutationFibreReport:
    manifest: dict[str, object]
    dataset_sha256: str
    model_sha256: str
    report_sha256: str
    victim_class: str
    points: tuple[ChannelPermutationFibrePoint, ...]
    summary: ChannelPermutationFibreSummary
    quality_gate: ChannelPermutationFibreQualityGate
    environment: dict[str, object]
    theorem: str
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest": self.manifest,
            "dataset_sha256": self.dataset_sha256,
            "model_sha256": self.model_sha256,
            "report_sha256": self.report_sha256,
            "victim_class": self.victim_class,
            "points": [point.to_dict() for point in self.points],
            "summary": self.summary.to_dict(),
            "quality_gate": self.quality_gate.to_dict(),
            "environment": self.environment,
            "theorem": self.theorem,
            "claim_boundary": self.claim_boundary,
        }


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    header = f"{value.dtype}:{tuple(value.shape)}:".encode("ascii")
    return hashlib.sha256(header + value.numpy().tobytes(order="C")).hexdigest()


def _dataset_sha256(inputs: torch.Tensor, targets: torch.Tensor) -> str:
    return hashlib.sha256(
        (_tensor_sha256(inputs) + ":" + _tensor_sha256(targets)).encode("ascii")
    ).hexdigest()


def _model_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(f":{tensor.dtype}:{tuple(tensor.shape)}:".encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _channel_signatures(
    inputs: torch.Tensor, targets: torch.Tensor
) -> tuple[str, ...]:
    signatures: list[str] = []
    for channel in range(inputs.shape[-1]):
        digest = hashlib.sha256()
        for tensor in (inputs[..., channel], targets[..., channel]):
            value = tensor.detach().cpu().contiguous()
            digest.update(f"{value.dtype}:{tuple(value.shape)}:".encode("ascii"))
            digest.update(value.numpy().tobytes(order="C"))
        signatures.append(digest.hexdigest())
    return tuple(signatures)


def _gradient_tuple(
    model: torch.nn.Module, inputs: torch.Tensor, targets: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    loss = (model(inputs) - targets).square().mean()
    parameters = tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
    return tuple(
        torch.zeros_like(parameter) if gradient is None else gradient.detach()
        for parameter, gradient in zip(parameters, gradients)
    )


def _tuple_difference(
    left: Sequence[torch.Tensor], right: Sequence[torch.Tensor]
) -> tuple[float, float]:
    maximum = 0.0
    numerator = 0.0
    denominator = 0.0
    for first, second in zip(left, right):
        difference = (first - second).double()
        maximum = max(maximum, float(difference.abs().max()))
        numerator += float(difference.square().sum())
        denominator += float(first.double().square().sum())
    return maximum, math.sqrt(numerator) / max(
        math.sqrt(denominator), torch.finfo(torch.float64).tiny
    )


def _adjacent_transpositions(channels: int) -> tuple[tuple[int, ...], ...]:
    permutations: list[tuple[int, ...]] = []
    for left in range(channels - 1):
        values = list(range(channels))
        values[left], values[left + 1] = values[left + 1], values[left]
        permutations.append(tuple(values))
    return tuple(permutations)


def analyze_channel_permutation_fibre(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    batch_start: int,
    tolerance: float,
) -> ChannelPermutationFibrePoint:
    channels = int(inputs.shape[-1])
    reference_outputs = model(inputs)
    reference_gradients = _gradient_tuple(model, inputs, targets)
    checks: list[ChannelPermutationGeneratorCheck] = []
    generators = _adjacent_transpositions(channels)
    for left_channel, permutation in enumerate(generators):
        indices = torch.tensor(permutation, dtype=torch.long, device=inputs.device)
        permuted_inputs = inputs.index_select(-1, indices)
        permuted_targets = targets.index_select(-1, indices)
        expected_outputs = reference_outputs.index_select(-1, indices)
        actual_outputs = model(permuted_inputs)
        output_error = float((actual_outputs - expected_outputs).detach().abs().max())
        permuted_gradients = _gradient_tuple(model, permuted_inputs, permuted_targets)
        gradient_maximum, gradient_relative = _tuple_difference(
            reference_gradients, permuted_gradients
        )
        checks.append(
            ChannelPermutationGeneratorCheck(
                left_channel=left_channel,
                right_channel=left_channel + 1,
                output_equivariance_max_abs_error=output_error,
                gradient_max_abs_difference=gradient_maximum,
                gradient_relative_l2_difference=gradient_relative,
            )
        )

    fibre: ChannelPermutationFibreBound = channel_permutation_fibre_bound(
        _channel_signatures(inputs, targets)
    )
    maximum_output_error = max(
        check.output_equivariance_max_abs_error for check in checks
    )
    maximum_gradient_error = max(
        check.gradient_max_abs_difference for check in checks
    )
    maximum_gradient_relative_error = max(
        check.gradient_relative_l2_difference for check in checks
    )
    passed = (
        maximum_output_error <= tolerance
        and maximum_gradient_error <= tolerance
        and maximum_gradient_relative_error <= tolerance
    )
    return ChannelPermutationFibrePoint(
        batch_start=int(batch_start),
        channels=channels,
        orbit_size=fibre.orbit_size,
        uniform_exact_ordered_recovery_ceiling=(
            fibre.uniform_exact_ordered_recovery_ceiling
        ),
        multiplicities=fibre.multiplicities,
        generator_count=len(generators),
        maximum_output_equivariance_error=maximum_output_error,
        maximum_gradient_invariance_error=maximum_gradient_error,
        maximum_gradient_relative_error=maximum_gradient_relative_error,
        all_generator_checks_pass=passed,
    )


def _data_access_locked(dataset: dict[str, object]) -> bool:
    name = str(dataset.get("name", "")).lower()
    if name == "gift_eval":
        return bool(str(dataset.get("revision", "")).strip())
    if name == "multivariate_csv":
        return bool(str(dataset.get("expected_file_sha256", "")).strip())
    return False


def run_channel_permutation_fibre_benchmark(
    manifest: ModernTimeSeriesAttackManifest,
    *,
    tolerance: float = 1e-5,
) -> ChannelPermutationFibreReport:
    """Run a real-data simultaneous channel-permutation fibre study."""

    if bool(manifest.victim.get("revin", True)) and bool(
        manifest.victim.get("revin_affine", True)
    ):
        raise ValueError(
            "the channel-permutation theorem requires revin=false or "
            "revin_affine=false because learned per-channel affine parameters "
            "attach identities to channel positions"
        )
    if bool(manifest.victim.get("individual_head", False)):
        raise ValueError(
            "the current permutation theorem requires a shared rather than per-channel head"
        )

    _seed_everything(manifest.victim_seed)
    dataset, task, mode = _load_dataset(
        {"seed": manifest.victim_seed, "dataset": dict(manifest.dataset)}
    )
    if task != "forecasting" or mode != "timeseries":
        raise ValueError("channel-permutation benchmark requires forecasting data")
    inputs, targets = dataset.tensors
    if inputs.ndim != 3 or targets.ndim != 3:
        raise ValueError("channel-permutation benchmark requires multivariate data")
    if inputs.shape[2] < 2:
        raise ValueError("at least two channels are required")

    model = _build_model(dataset, task, dict(manifest.victim))
    _train(model, dataset, task, dict(manifest.training))
    model.eval()

    points: list[ChannelPermutationFibrePoint] = []
    for batch_start in manifest.attack_indices:
        end = batch_start + manifest.attack_batch_size
        if end > len(inputs):
            raise ValueError("attack batch exceeds dataset size")
        points.append(
            analyze_channel_permutation_fibre(
                model,
                inputs[batch_start:end].clone(),
                targets[batch_start:end].clone(),
                batch_start=batch_start,
                tolerance=tolerance,
            )
        )

    confidence_level = manifest.confidence_level
    bootstrap_samples = manifest.bootstrap_samples
    bootstrap_seed = manifest.bootstrap_seed
    summary = ChannelPermutationFibreSummary(
        points=len(points),
        nontrivial_orbit=summarize_proportion(
            sum(point.orbit_size > 1 for point in points),
            len(points),
            confidence_level=confidence_level,
        ),
        generator_checks=summarize_proportion(
            sum(point.all_generator_checks_pass for point in points),
            len(points),
            confidence_level=confidence_level,
        ),
        orbit_size=summarize_scalar(
            [float(point.orbit_size) for point in points],
            confidence_level=confidence_level,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
            label="channel-permutation:orbit-size",
        ),
        uniform_exact_ordered_recovery_ceiling=summarize_scalar(
            [point.uniform_exact_ordered_recovery_ceiling for point in points],
            confidence_level=confidence_level,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
            label="channel-permutation:recovery-ceiling",
        ),
        maximum_output_equivariance_error=summarize_scalar(
            [point.maximum_output_equivariance_error for point in points],
            confidence_level=confidence_level,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
            label="channel-permutation:output-error",
        ),
        maximum_gradient_invariance_error=summarize_scalar(
            [point.maximum_gradient_invariance_error for point in points],
            confidence_level=confidence_level,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
            label="channel-permutation:gradient-error",
        ),
    )

    dataset_name = str(manifest.dataset.get("name", "")).lower()
    real_dataset = dataset_name in {"gift_eval", "multivariate_csv"}
    quality_gate = ChannelPermutationFibreQualityGate(
        real_dataset=real_dataset,
        data_access_locked=_data_access_locked(dict(manifest.dataset)),
        multivariate=inputs.ndim == 3 and inputs.shape[-1] > 1,
        enough_attack_batches=len(points) >= manifest.minimum_publication_batches,
        every_orbit_nontrivial=all(point.orbit_size > 1 for point in points),
        all_generator_checks_pass=all(
            point.all_generator_checks_pass for point in points
        ),
        publication_mode=manifest.publication_mode,
    )
    manifest_payload = manifest.to_dict()
    dataset_hash = _dataset_sha256(inputs, targets)
    model_hash = _model_sha256(model)
    report_basis = {
        "manifest": manifest_payload,
        "dataset_sha256": dataset_hash,
        "model_sha256": model_hash,
        "victim_class": type(model).__name__,
        "points": [point.to_dict() for point in points],
        "summary": summary.to_dict(),
        "quality_gate": quality_gate.to_dict(),
    }
    report_sha256 = hashlib.sha256(
        json.dumps(report_basis, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return ChannelPermutationFibreReport(
        manifest=manifest_payload,
        dataset_sha256=dataset_hash,
        model_sha256=model_hash,
        report_sha256=report_sha256,
        victim_class=type(model).__name__,
        points=tuple(points),
        summary=summary,
        quality_gate=quality_gate,
        environment=benchmark_environment_manifest(),
        theorem=(
            "Every adjacent channel transposition leaves the full gradient invariant; "
            "because adjacent transpositions generate S_C, the full simultaneous "
            "channel-permutation orbit is one observation fibre."
        ),
        claim_boundary=(
            "The private target contains both ordered input histories and ordered "
            "forecast targets. Public channel labels, channel-indexed parameters, "
            "channel-specific heads, or a recovery target defined modulo permutation "
            "change the threat model."
        ),
    )
