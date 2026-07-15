from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
from dataclasses import asdict, dataclass
from typing import Sequence

import torch

from qrecon.benchmarks.modern_timeseries_reconstruction import (
    ModernTimeSeriesAttackManifest,
    _tensor_sha256,
)
from qrecon.benchmarks.statistics import (
    benchmark_environment_manifest,
    summarize_proportion,
    summarize_scalar,
)
from qrecon.experiment import _load_dataset
from qrecon.theory.channel_permutation import (
    tensor_channel_permutation_fibre_bound,
    validate_channel_permutation,
)


@dataclass(frozen=True)
class ChannelSideInformationPoint:
    batch_start: int
    permutation_seed: int
    hidden_permutation: tuple[int, ...]
    predicted_labels_by_recovered_slot: tuple[int, ...]
    exact_labeled_order_recovered: bool
    correctly_labeled_channels: int
    channel_accuracy: float
    assignment_cost: float
    second_best_cost: float
    assignment_margin: float
    orbit_size: int
    no_side_information_exact_ceiling: float

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["hidden_permutation"] = list(self.hidden_permutation)
        payload["predicted_labels_by_recovered_slot"] = list(
            self.predicted_labels_by_recovered_slot
        )
        return payload


@dataclass(frozen=True)
class ChannelSideInformationQualityGate:
    real_dataset: bool
    data_access_locked: bool
    multivariate: bool
    disjoint_calibration_and_evaluation: bool
    enough_calibration_windows: bool
    enough_evaluation_windows: bool
    enough_permutation_trials: bool
    no_failed_trials: bool
    publication_mode: bool
    passed: bool

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)


@dataclass(frozen=True)
class ChannelSideInformationReport:
    manifest: dict[str, object]
    report_sha256: str
    dataset_sha256: str
    calibration_indices: tuple[int, ...]
    evaluation_indices: tuple[int, ...]
    permutation_seeds: tuple[int, ...]
    feature_names: tuple[str, ...]
    points: tuple[ChannelSideInformationPoint, ...]
    summary: dict[str, object]
    quality_gate: ChannelSideInformationQualityGate
    environment: dict[str, object]
    interpretation: str
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest": self.manifest,
            "report_sha256": self.report_sha256,
            "dataset_sha256": self.dataset_sha256,
            "calibration_indices": list(self.calibration_indices),
            "evaluation_indices": list(self.evaluation_indices),
            "permutation_seeds": list(self.permutation_seeds),
            "feature_names": list(self.feature_names),
            "points": [point.to_dict() for point in self.points],
            "summary": self.summary,
            "quality_gate": self.quality_gate.to_dict(),
            "environment": self.environment,
            "interpretation": self.interpretation,
            "claim_boundary": self.claim_boundary,
        }


def _feature_names(context: int, spectral_bins: int) -> tuple[str, ...]:
    lags = tuple(lag for lag in (1, 2, 4, 8) if lag < context)
    return (
        "first",
        "last",
        "minimum",
        "maximum",
        "q25",
        "median",
        "q75",
        "linear_slope",
        "difference_absolute_mean",
        "difference_standard_deviation",
        *(f"autocorrelation_lag_{lag}" for lag in lags),
        *(f"relative_fft_bin_{index}" for index in range(1, spectral_bins + 1)),
    )


def _channel_features(
    window: torch.Tensor,
    *,
    spectral_bins: int,
) -> torch.Tensor:
    """Return one deterministic feature vector per channel for `[time, channels]`."""

    if window.ndim != 2 or window.shape[0] < 3 or window.shape[1] < 2:
        raise ValueError("window must have shape [time, channels] with at least 3x2 entries")
    values = window.detach().double()
    time, channels = values.shape
    positions = torch.linspace(-1.0, 1.0, time, dtype=torch.float64)
    slope_denominator = float(positions.square().sum())
    quantiles = torch.quantile(
        values,
        torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64),
        dim=0,
    )
    differences = values[1:] - values[:-1]
    pieces: list[torch.Tensor] = [
        values[0],
        values[-1],
        values.amin(dim=0),
        values.amax(dim=0),
        quantiles[0],
        quantiles[1],
        quantiles[2],
        (positions[:, None] * values).sum(dim=0) / slope_denominator,
        differences.abs().mean(dim=0),
        differences.std(dim=0, unbiased=False),
    ]
    centered = values - values.mean(dim=0, keepdim=True)
    variance = centered.square().mean(dim=0).clamp_min(1e-12)
    for lag in (1, 2, 4, 8):
        if lag < time:
            correlation = (
                centered[:-lag] * centered[lag:]
            ).mean(dim=0) / variance
            pieces.append(correlation)
    spectrum = torch.fft.rfft(centered, dim=0).abs()
    spectral_norm = spectrum[1:].square().sum(dim=0).sqrt().clamp_min(1e-12)
    available = max(0, min(int(spectral_bins), spectrum.shape[0] - 1))
    for index in range(1, available + 1):
        pieces.append(spectrum[index] / spectral_norm)
    for _ in range(available, int(spectral_bins)):
        pieces.append(torch.zeros(channels, dtype=torch.float64))
    return torch.stack(pieces, dim=1)


def _prototype_statistics(
    inputs: torch.Tensor,
    calibration_indices: Sequence[int],
    *,
    spectral_bins: int,
    variance_floor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected = tuple(int(index) for index in calibration_indices)
    if not selected:
        raise ValueError("calibration_indices must be non-empty")
    features = torch.stack(
        [
            _channel_features(inputs[index], spectral_bins=spectral_bins)
            for index in selected
        ],
        dim=0,
    )
    means = features.mean(dim=0)
    variances = features.var(dim=0, unbiased=False)
    # One shared floor per feature prevents a nearly constant channel statistic from
    # dominating solely through floating-point noise while retaining scale adaptation.
    feature_scale = variances.mean(dim=0, keepdim=True).clamp_min(float(variance_floor))
    return means, variances + feature_scale


def _assignment_costs(
    recovered_features: torch.Tensor,
    prototype_means: torch.Tensor,
    prototype_variances: torch.Tensor,
) -> torch.Tensor:
    if recovered_features.shape != prototype_means.shape:
        raise ValueError("recovered and prototype feature matrices must have equal shape")
    if prototype_variances.shape != prototype_means.shape:
        raise ValueError("prototype variance matrix shape mismatch")
    differences = recovered_features[:, None, :] - prototype_means[None, :, :]
    # Diagonal Gaussian negative log-likelihood up to an additive constant.
    return (
        differences.square() / prototype_variances[None, :, :]
        + prototype_variances[None, :, :].log()
    ).sum(dim=-1)


def _best_assignment(costs: torch.Tensor) -> tuple[tuple[int, ...], float, float]:
    channels = int(costs.shape[0])
    if costs.shape != (channels, channels):
        raise ValueError("assignment cost matrix must be square")
    if channels > 9:
        raise ValueError("exact assignment enumeration is limited to at most 9 channels")
    ranked: list[tuple[float, tuple[int, ...]]] = []
    rows = torch.arange(channels)
    for assignment in itertools.permutations(range(channels)):
        columns = torch.tensor(assignment, dtype=torch.long)
        value = float(costs[rows, columns].sum())
        ranked.append((value, tuple(assignment)))
    ranked.sort(key=lambda item: (item[0], item[1]))
    best_cost, best = ranked[0]
    second_cost = ranked[1][0] if len(ranked) > 1 else math.inf
    return best, best_cost, second_cost


def analyze_public_calibration_side_information(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    calibration_indices: Sequence[int],
    evaluation_indices: Sequence[int],
    permutation_seeds: Sequence[int],
    spectral_bins: int = 4,
    variance_floor: float = 1e-4,
) -> tuple[tuple[ChannelSideInformationPoint, ...], tuple[str, ...]]:
    """Measure how public labeled calibration data shrinks a permutation fibre.

    The attacker receives an exact numerical orbit representative—strictly stronger
    than any imperfect reconstruction—whose channel order is hidden. It also receives
    disjoint, correctly labeled public calibration windows from the same data source.
    An exact assignment solver matches recovered channels to label prototypes using
    temporal shape, autocorrelation, trend, and spectral features.
    """

    if inputs.ndim != 3 or targets.ndim != 3:
        raise ValueError("inputs and targets must be [samples,time,channels]")
    if inputs.shape[0] != targets.shape[0] or inputs.shape[-1] != targets.shape[-1]:
        raise ValueError("input and target sample/channel dimensions must match")
    calibration = tuple(int(index) for index in calibration_indices)
    evaluation = tuple(int(index) for index in evaluation_indices)
    seeds = tuple(int(seed) for seed in permutation_seeds)
    if not calibration or not evaluation or not seeds:
        raise ValueError("calibration, evaluation and permutation seeds must be non-empty")
    if set(calibration) & set(evaluation):
        raise ValueError("calibration and evaluation indices must be disjoint")
    if min(calibration + evaluation) < 0 or max(calibration + evaluation) >= len(inputs):
        raise ValueError("calibration/evaluation index is outside the dataset")
    if spectral_bins <= 0 or variance_floor <= 0.0:
        raise ValueError("spectral_bins and variance_floor must be positive")

    means, variances = _prototype_statistics(
        inputs,
        calibration,
        spectral_bins=spectral_bins,
        variance_floor=variance_floor,
    )
    channels = int(inputs.shape[-1])
    points: list[ChannelSideInformationPoint] = []
    for batch_start in evaluation:
        private_inputs = inputs[batch_start : batch_start + 1]
        private_targets = targets[batch_start : batch_start + 1]
        fibre = tensor_channel_permutation_fibre_bound(
            private_inputs, private_targets
        )
        for seed in seeds:
            rng = random.Random((seed << 32) ^ batch_start)
            permutation = list(range(channels))
            rng.shuffle(permutation)
            hidden = validate_channel_permutation(permutation, channels)
            recovered = private_inputs[0, :, list(hidden)]
            features = _channel_features(recovered, spectral_bins=spectral_bins)
            costs = _assignment_costs(features, means, variances)
            predicted, best_cost, second_cost = _best_assignment(costs)
            correct = sum(
                predicted[slot] == hidden[slot] for slot in range(channels)
            )
            points.append(
                ChannelSideInformationPoint(
                    batch_start=batch_start,
                    permutation_seed=seed,
                    hidden_permutation=hidden,
                    predicted_labels_by_recovered_slot=predicted,
                    exact_labeled_order_recovered=predicted == hidden,
                    correctly_labeled_channels=correct,
                    channel_accuracy=correct / channels,
                    assignment_cost=best_cost,
                    second_best_cost=second_cost,
                    assignment_margin=second_cost - best_cost,
                    orbit_size=fibre.orbit_size,
                    no_side_information_exact_ceiling=(
                        fibre.uniform_exact_ordered_recovery_ceiling
                    ),
                )
            )
    return tuple(points), _feature_names(int(inputs.shape[1]), spectral_bins)


def _data_access_locked(dataset: dict[str, object]) -> bool:
    name = str(dataset.get("name", "")).lower()
    if name == "gift_eval":
        return bool(str(dataset.get("revision", "")).strip())
    if name == "multivariate_csv":
        return bool(str(dataset.get("expected_file_sha256", "")).strip())
    return False


def run_channel_side_information_benchmark(
    manifest: ModernTimeSeriesAttackManifest,
    *,
    calibration_indices: Sequence[int],
    evaluation_indices: Sequence[int],
    permutation_seeds: Sequence[int],
    spectral_bins: int = 4,
    variance_floor: float = 1e-4,
    minimum_calibration_windows: int = 20,
    minimum_permutation_trials: int = 100,
) -> ChannelSideInformationReport:
    dataset, task, mode = _load_dataset(
        {"seed": manifest.victim_seed, "dataset": dict(manifest.dataset)}
    )
    if task != "forecasting" or mode != "timeseries":
        raise ValueError("side-information benchmark requires forecasting data")
    inputs, targets = dataset.tensors
    if inputs.ndim != 3 or inputs.shape[-1] < 2:
        raise ValueError("side-information benchmark requires multivariate inputs")
    points, feature_names = analyze_public_calibration_side_information(
        inputs,
        targets,
        calibration_indices=calibration_indices,
        evaluation_indices=evaluation_indices,
        permutation_seeds=permutation_seeds,
        spectral_bins=spectral_bins,
        variance_floor=variance_floor,
    )

    def scalar(values: list[float], label: str) -> dict[str, object]:
        return summarize_scalar(
            values,
            confidence_level=manifest.confidence_level,
            bootstrap_samples=manifest.bootstrap_samples,
            bootstrap_seed=manifest.bootstrap_seed,
            label=f"channel-side-information:{label}",
        ).to_dict()

    exact_success = summarize_proportion(
        sum(point.exact_labeled_order_recovered for point in points),
        len(points),
        confidence_level=manifest.confidence_level,
    ).to_dict()
    summary = {
        "trials": len(points),
        "exact_labeled_order_recovery": exact_success,
        "channel_accuracy": scalar(
            [point.channel_accuracy for point in points], "channel-accuracy"
        ),
        "assignment_margin": scalar(
            [point.assignment_margin for point in points], "assignment-margin"
        ),
        "no_side_information_exact_ceiling": scalar(
            [point.no_side_information_exact_ceiling for point in points],
            "no-side-information-ceiling",
        ),
        "multiplicative_exact_success_over_uniform_ceiling": scalar(
            [
                float(point.exact_labeled_order_recovered)
                / point.no_side_information_exact_ceiling
                for point in points
            ],
            "multiplicative-success-over-ceiling",
        ),
    }
    dataset_name = str(manifest.dataset.get("name", "")).lower()
    real_dataset = not dataset_name.startswith("synthetic")
    calibration = tuple(int(index) for index in calibration_indices)
    evaluation = tuple(int(index) for index in evaluation_indices)
    seeds = tuple(int(seed) for seed in permutation_seeds)
    gate_passed = (
        manifest.publication_mode
        and real_dataset
        and _data_access_locked(dict(manifest.dataset))
        and inputs.shape[-1] > 1
        and not (set(calibration) & set(evaluation))
        and len(calibration) >= int(minimum_calibration_windows)
        and len(evaluation) >= manifest.minimum_publication_batches
        and len(points) >= int(minimum_permutation_trials)
    )
    gate = ChannelSideInformationQualityGate(
        real_dataset=real_dataset,
        data_access_locked=_data_access_locked(dict(manifest.dataset)),
        multivariate=inputs.shape[-1] > 1,
        disjoint_calibration_and_evaluation=not bool(set(calibration) & set(evaluation)),
        enough_calibration_windows=len(calibration) >= int(minimum_calibration_windows),
        enough_evaluation_windows=(
            len(evaluation) >= manifest.minimum_publication_batches
        ),
        enough_permutation_trials=len(points) >= int(minimum_permutation_trials),
        no_failed_trials=len(points) == len(evaluation) * len(seeds),
        publication_mode=manifest.publication_mode,
        passed=gate_passed,
    )
    report_basis = {
        "schema_version": "qrecon.channel-side-information.v1",
        "manifest": manifest.to_dict(),
        "calibration_indices": list(calibration),
        "evaluation_indices": list(evaluation),
        "permutation_seeds": list(seeds),
        "spectral_bins": spectral_bins,
        "variance_floor": variance_floor,
        "dataset_sha256": _tensor_sha256(inputs, targets),
        "points": [point.to_dict() for point in points],
        "summary": summary,
        "quality_gate": gate.to_dict(),
    }
    report_sha256 = hashlib.sha256(
        json.dumps(report_basis, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return ChannelSideInformationReport(
        manifest=manifest.to_dict(),
        report_sha256=report_sha256,
        dataset_sha256=report_basis["dataset_sha256"],
        calibration_indices=calibration,
        evaluation_indices=evaluation,
        permutation_seeds=seeds,
        feature_names=feature_names,
        points=points,
        summary=summary,
        quality_gate=gate,
        environment=benchmark_environment_manifest(),
        interpretation=(
            "The no-side-information orbit ceiling is conditional. Public labeled "
            "calibration data can attach statistical identities to anonymous recovered "
            "channels and may substantially increase labeled-order recovery."
        ),
        claim_boundary=(
            "This benchmark gives the attacker an exact orbit representative and "
            "disjoint labeled calibration windows. It measures one explicit temporal-"
            "feature matcher, not the Bayes-optimal use of all possible side information."
        ),
    )
