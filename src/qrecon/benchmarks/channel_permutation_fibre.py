from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

import torch

from qrecon.attacks import leak_gradients
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
from qrecon.theory.channel_permutation import channel_permutation_fibre_bound


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
class ChannelPermutationFibreQualityGate:
    real_dataset: bool
    data_access_locked: bool
    enough_attack_batches: bool
    multivariate: bool
    all_generator_checks_pass: bool
    every_orbit_nontrivial: bool
    publication_mode: bool
    passed: bool

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)


@dataclass(frozen=True)
class ChannelPermutationFibreReport:
    manifest: ModernTimeSeriesAttackManifest
    report_sha256: str
    dataset_sha256: str
    model_sha256: str
    victim_class: str
    points: tuple[ChannelPermutationFibrePoint, ...]
    summary: dict[str, object]
    quality_gate: ChannelPermutationFibreQualityGate
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
            "points": [point.to_dict() for point in self.points],
            "summary": self.summary,
            "quality_gate": self.quality_gate.to_dict(),
            "environment": self.environment,
            "theorem": self.theorem,
            "claim_boundary": self.claim_boundary,
        }


def _report_sha256(manifest: ModernTimeSeriesAttackManifest) -> str:
    payload = {
        "schema_version": "qrecon.channel-permutation-fibre.v1",
        "manifest": manifest.to_dict(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _channel_signatures(inputs: torch.Tensor, targets: torch.Tensor) -> tuple[str, ...]:
    if inputs.ndim != 3 or targets.ndim != 3:
        raise ValueError("channel signatures require multivariate [batch,time,channel] tensors")
    if inputs.shape[0] != targets.shape[0] or inputs.shape[2] != targets.shape[2]:
        raise ValueError("input and target batch/channel dimensions must match")
    signatures: list[str] = []
    for channel in range(inputs.shape[2]):
        digest = hashlib.sha256()
        for tensor in (inputs[:, :, channel], targets[:, :, channel]):
            value = tensor.detach().cpu().contiguous()
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(json.dumps(list(value.shape)).encode("ascii"))
            digest.update(value.numpy().tobytes(order="C"))
        signatures.append(digest.hexdigest())
    return tuple(signatures)


def _adjacent_transpositions(channels: int) -> tuple[torch.Tensor, ...]:
    if channels < 2:
        return ()
    result: list[torch.Tensor] = []
    for left in range(channels - 1):
        permutation = torch.arange(channels)
        permutation[left], permutation[left + 1] = (
            permutation[left + 1].clone(),
            permutation[left].clone(),
        )
        result.append(permutation)
    return tuple(result)


def _gradient_errors(
    reference: tuple[torch.Tensor, ...],
    candidate: tuple[torch.Tensor, ...],
) -> tuple[float, float]:
    if len(reference) != len(candidate):
        raise ValueError("gradient tuples have different lengths")
    maximum = 0.0
    numerator = 0.0
    denominator = 0.0
    for left, right in zip(reference, candidate):
        difference = left - right
        maximum = max(maximum, float(difference.abs().max()))
        numerator += float(difference.square().sum())
        denominator += float(left.square().sum())
    return maximum, math.sqrt(numerator / max(denominator, 1e-30))


def analyze_channel_permutation_fibre(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    batch_start: int = 0,
    tolerance: float = 1e-5,
) -> ChannelPermutationFibrePoint:
    """Verify output equivariance and full-gradient invariance on generators of S_C."""

    if inputs.ndim != 3 or targets.ndim != 3:
        raise ValueError("channel-permutation analysis requires multivariate tensors")
    if inputs.shape[2] != targets.shape[2]:
        raise ValueError("input and target channel counts differ")
    if not math.isfinite(float(tolerance)) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")

    channels = int(inputs.shape[2])
    signatures = _channel_signatures(inputs, targets)
    fibre = channel_permutation_fibre_bound(signatures)
    generators = _adjacent_transpositions(channels)
    with torch.no_grad():
        reference_output = model(inputs).detach()
    reference_gradients = leak_gradients(model, inputs, targets, "forecasting")

    maximum_output_error = 0.0
    maximum_gradient_error = 0.0
    maximum_gradient_relative_error = 0.0
    for permutation in generators:
        permuted_inputs = inputs[:, :, permutation]
        permuted_targets = targets[:, :, permutation]
        with torch.no_grad():
            permuted_output = model(permuted_inputs).detach()
        expected_output = reference_output[:, :, permutation]
        maximum_output_error = max(
            maximum_output_error,
            float((permuted_output - expected_output).abs().max()),
        )
        permuted_gradients = leak_gradients(
            model,
            permuted_inputs,
            permuted_targets,
            "forecasting",
        )
        absolute_error, relative_error = _gradient_errors(
            reference_gradients,
            permuted_gradients,
        )
        maximum_gradient_error = max(maximum_gradient_error, absolute_error)
        maximum_gradient_relative_error = max(
            maximum_gradient_relative_error,
            relative_error,
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

    if bool(manifest.victim.get("revin", True)):
        raise ValueError(
            "the current permutation theorem requires revin=false because learned "
            "per-channel RevIN affine parameters attach identities to channel positions"
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

    def scalar(values: list[float], label: str) -> dict[str, object]:
        return summarize_scalar(
            values,
            confidence_level=manifest.confidence_level,
            bootstrap_samples=manifest.bootstrap_samples,
            bootstrap_seed=manifest.bootstrap_seed,
            label=f"channel-permutation:{label}",
        ).to_dict()

    all_checks = all(point.all_generator_checks_pass for point in points)
    every_nontrivial = all(point.orbit_size > 1 for point in points)
    summary: dict[str, object] = {
        "points": len(points),
        "generator_checks": summarize_proportion(
            sum(point.all_generator_checks_pass for point in points),
            len(points),
            confidence_level=manifest.confidence_level,
        ).to_dict(),
        "nontrivial_orbit": summarize_proportion(
            sum(point.orbit_size > 1 for point in points),
            len(points),
            confidence_level=manifest.confidence_level,
        ).to_dict(),
        "orbit_size": scalar(
            [float(point.orbit_size) for point in points],
            "orbit_size",
        ),
        "uniform_exact_ordered_recovery_ceiling": scalar(
            [
                point.uniform_exact_ordered_recovery_ceiling
                for point in points
            ],
            "uniform_exact_ordered_recovery_ceiling",
        ),
        "maximum_output_equivariance_error": scalar(
            [point.maximum_output_equivariance_error for point in points],
            "output_equivariance_error",
        ),
        "maximum_gradient_invariance_error": scalar(
            [point.maximum_gradient_invariance_error for point in points],
            "gradient_invariance_error",
        ),
    }

    dataset_name = str(manifest.dataset.get("name", "")).lower()
    real_dataset = not dataset_name.startswith("synthetic")
    locked = _data_access_locked(dict(manifest.dataset))
    enough = len(points) >= manifest.minimum_publication_batches
    passed = (
        manifest.publication_mode
        and real_dataset
        and locked
        and enough
        and all_checks
        and every_nontrivial
    )
    gate = ChannelPermutationFibreQualityGate(
        real_dataset=real_dataset,
        data_access_locked=locked,
        enough_attack_batches=enough,
        multivariate=True,
        all_generator_checks_pass=all_checks,
        every_orbit_nontrivial=every_nontrivial,
        publication_mode=manifest.publication_mode,
        passed=passed,
    )

    return ChannelPermutationFibreReport(
        manifest=manifest,
        report_sha256=_report_sha256(manifest),
        dataset_sha256=_tensor_sha256(inputs, targets),
        model_sha256=_model_sha256(model),
        victim_class=type(model).__name__,
        points=tuple(points),
        summary=summary,
        quality_gate=gate,
        environment=benchmark_environment_manifest(),
        theorem=(
            "If F_theta(Px)=P F_theta(x) for every channel permutation P and the "
            "loss is invariant under simultaneous permutation of prediction and "
            "target, then grad_theta L(F_theta(Px), Py)=grad_theta L(F_theta(x), y). "
            "The observation fibre contains the full simultaneous channel orbit."
        ),
        claim_boundary=(
            "This bound treats channel identities and their ordering as part of the "
            "private training object and assumes both histories and targets are private. "
            "If ordered targets are public, or channel permutations are declared an "
            "acceptable target equivalence, the exact labeled-order ceiling does not apply."
        ),
    )
