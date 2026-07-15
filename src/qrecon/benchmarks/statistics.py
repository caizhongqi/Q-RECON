from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import statistics as _statistics
import sys
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from importlib import metadata
from statistics import NormalDist
from typing import Any


@dataclass(frozen=True)
class ConfidenceInterval:
    confidence_level: float
    lower: float
    upper: float
    method: str

    def to_dict(self) -> dict[str, float | str]:
        return asdict(self)


@dataclass(frozen=True)
class ScalarSummary:
    count: int
    mean: float
    sample_standard_deviation: float
    median: float
    minimum: float
    maximum: float
    mean_interval: ConfidenceInterval
    median_interval: ConfidenceInterval

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["mean_interval"] = self.mean_interval.to_dict()
        payload["median_interval"] = self.median_interval.to_dict()
        return payload


@dataclass(frozen=True)
class ProportionSummary:
    successes: int
    trials: int
    rate: float
    interval: ConfidenceInterval

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["interval"] = self.interval.to_dict()
        return payload


@dataclass(frozen=True)
class LogLogScalingFit:
    point_count: int
    distinct_x_values: int
    slope: float
    intercept: float
    r_squared: float
    slope_interval: ConfidenceInterval | None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["slope_interval"] = (
            None if self.slope_interval is None else self.slope_interval.to_dict()
        )
        return payload


@dataclass(frozen=True)
class FixedPointMLPConfigSummary:
    config: dict[str, object]
    seeds: tuple[int, ...]
    candidate_count: int
    full_word_population: int
    solution_count: ScalarSummary
    uniquely_identifiable: ProportionSummary
    classical_oracle_agreement: ProportionSummary
    branch_and_bound_leaf_fraction: ScalarSummary
    branch_and_bound_seconds: ScalarSummary
    oracle_build_seconds: ScalarSummary
    basis_permutation_checked: int
    basis_permutation_skipped: int
    basis_permutation_success: ProportionSummary | None
    z3_attempted: int
    z3_complete: ProportionSummary | None
    z3_solution_set_agreement: ProportionSummary | None
    z3_seconds: ScalarSummary | None
    oracle_resource_summaries: dict[str, ScalarSummary]
    bbht_expected_phase_oracle_calls: ScalarSummary | None

    def to_dict(self) -> dict[str, object]:
        return {
            "config": self.config,
            "seeds": list(self.seeds),
            "candidate_count": self.candidate_count,
            "full_word_population": self.full_word_population,
            "solution_count": self.solution_count.to_dict(),
            "uniquely_identifiable": self.uniquely_identifiable.to_dict(),
            "classical_oracle_agreement": self.classical_oracle_agreement.to_dict(),
            "branch_and_bound_leaf_fraction": self.branch_and_bound_leaf_fraction.to_dict(),
            "branch_and_bound_seconds": self.branch_and_bound_seconds.to_dict(),
            "oracle_build_seconds": self.oracle_build_seconds.to_dict(),
            "basis_permutation_checked": self.basis_permutation_checked,
            "basis_permutation_skipped": self.basis_permutation_skipped,
            "basis_permutation_success": (
                None
                if self.basis_permutation_success is None
                else self.basis_permutation_success.to_dict()
            ),
            "z3_attempted": self.z3_attempted,
            "z3_complete": None if self.z3_complete is None else self.z3_complete.to_dict(),
            "z3_solution_set_agreement": (
                None
                if self.z3_solution_set_agreement is None
                else self.z3_solution_set_agreement.to_dict()
            ),
            "z3_seconds": None if self.z3_seconds is None else self.z3_seconds.to_dict(),
            "oracle_resource_summaries": {
                name: summary.to_dict()
                for name, summary in sorted(self.oracle_resource_summaries.items())
            },
            "bbht_expected_phase_oracle_calls": (
                None
                if self.bbht_expected_phase_oracle_calls is None
                else self.bbht_expected_phase_oracle_calls.to_dict()
            ),
        }


@dataclass(frozen=True)
class BenchmarkQualityGate:
    minimum_required_seeds_per_config: int
    observed_minimum_seeds_per_config: int
    minimum_required_candidate_scales: int
    observed_candidate_scales: int
    balanced_seed_matrix: bool
    enough_seeds: bool
    enough_candidate_scales: bool
    all_classical_oracle_solution_sets_match: bool
    all_attempted_z3_runs_complete: bool | None
    all_completed_z3_solution_sets_match: bool | None
    passed: bool

    def to_dict(self) -> dict[str, int | bool | None]:
        return asdict(self)


@dataclass(frozen=True)
class FixedPointMLPMatrixSummary:
    total_instances: int
    configuration_count: int
    confidence_level: float
    bootstrap_samples: int
    bootstrap_seed: int
    configurations: tuple[FixedPointMLPConfigSummary, ...]
    overall_uniquely_identifiable: ProportionSummary
    overall_classical_oracle_agreement: ProportionSummary
    overall_z3_complete: ProportionSummary | None
    overall_z3_solution_set_agreement: ProportionSummary | None
    branch_and_bound_scaling: LogLogScalingFit | None
    oracle_build_scaling: LogLogScalingFit | None
    quality_gate: BenchmarkQualityGate
    environment: dict[str, object]
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        return {
            "total_instances": self.total_instances,
            "configuration_count": self.configuration_count,
            "confidence_level": self.confidence_level,
            "bootstrap_samples": self.bootstrap_samples,
            "bootstrap_seed": self.bootstrap_seed,
            "configurations": [summary.to_dict() for summary in self.configurations],
            "overall_uniquely_identifiable": self.overall_uniquely_identifiable.to_dict(),
            "overall_classical_oracle_agreement": (
                self.overall_classical_oracle_agreement.to_dict()
            ),
            "overall_z3_complete": (
                None if self.overall_z3_complete is None else self.overall_z3_complete.to_dict()
            ),
            "overall_z3_solution_set_agreement": (
                None
                if self.overall_z3_solution_set_agreement is None
                else self.overall_z3_solution_set_agreement.to_dict()
            ),
            "branch_and_bound_scaling": (
                None
                if self.branch_and_bound_scaling is None
                else self.branch_and_bound_scaling.to_dict()
            ),
            "oracle_build_scaling": (
                None
                if self.oracle_build_scaling is None
                else self.oracle_build_scaling.to_dict()
            ),
            "quality_gate": self.quality_gate.to_dict(),
            "environment": self.environment,
            "claim_boundary": self.claim_boundary,
        }


def _validate_confidence(confidence_level: float) -> float:
    value = float(confidence_level)
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")
    return value


def _finite_values(values: Sequence[float]) -> tuple[float, ...]:
    converted = tuple(float(value) for value in values)
    if not converted:
        raise ValueError("values must be non-empty")
    if not all(math.isfinite(value) for value in converted):
        raise ValueError("values must be finite")
    return converted


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("sorted_values must be non-empty")
    if probability <= 0.0:
        return float(sorted_values[0])
    if probability >= 1.0:
        return float(sorted_values[-1])
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def _derived_seed(base_seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{int(base_seed)}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _bootstrap_interval(
    values: tuple[float, ...],
    statistic: Callable[[Sequence[float]], float],
    confidence_level: float,
    bootstrap_samples: int,
    seed: int,
    label: str,
) -> ConfidenceInterval:
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    if len(values) == 1:
        point = float(statistic(values))
        return ConfidenceInterval(confidence_level, point, point, "degenerate-bootstrap")
    rng = random.Random(_derived_seed(seed, label))
    n = len(values)
    draws = sorted(
        float(statistic(tuple(values[rng.randrange(n)] for _ in range(n))))
        for _ in range(bootstrap_samples)
    )
    alpha = (1.0 - confidence_level) / 2.0
    return ConfidenceInterval(
        confidence_level,
        _percentile(draws, alpha),
        _percentile(draws, 1.0 - alpha),
        "percentile-bootstrap",
    )


def summarize_scalar(
    values: Sequence[float],
    *,
    confidence_level: float = 0.95,
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 1729,
    label: str = "scalar",
) -> ScalarSummary:
    """Summarize a scalar sample with deterministic bootstrap intervals."""

    confidence = _validate_confidence(confidence_level)
    sample = _finite_values(values)
    return ScalarSummary(
        count=len(sample),
        mean=float(_statistics.fmean(sample)),
        sample_standard_deviation=(
            float(_statistics.stdev(sample)) if len(sample) > 1 else 0.0
        ),
        median=float(_statistics.median(sample)),
        minimum=min(sample),
        maximum=max(sample),
        mean_interval=_bootstrap_interval(
            sample,
            _statistics.fmean,
            confidence,
            bootstrap_samples,
            bootstrap_seed,
            f"{label}:mean",
        ),
        median_interval=_bootstrap_interval(
            sample,
            _statistics.median,
            confidence,
            bootstrap_samples,
            bootstrap_seed,
            f"{label}:median",
        ),
    )


def summarize_proportion(
    successes: int,
    trials: int,
    *,
    confidence_level: float = 0.95,
) -> ProportionSummary:
    """Wilson score interval for a Bernoulli rate."""

    confidence = _validate_confidence(confidence_level)
    successful = int(successes)
    total = int(trials)
    if total <= 0:
        raise ValueError("trials must be positive")
    if successful < 0 or successful > total:
        raise ValueError("successes must lie in [0, trials]")
    rate = successful / total
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return ProportionSummary(
        successes=successful,
        trials=total,
        rate=rate,
        interval=ConfidenceInterval(
            confidence,
            max(0.0, center - radius),
            min(1.0, center + radius),
            "wilson-score",
        ),
    )


def fit_loglog_scaling(
    points: Sequence[tuple[float, float]],
    *,
    confidence_level: float = 0.95,
) -> LogLogScalingFit | None:
    """Fit ``log(y)=intercept+slope*log(x)`` for positive measured points."""

    confidence = _validate_confidence(confidence_level)
    converted = tuple((float(x), float(y)) for x, y in points)
    if any(
        not math.isfinite(x) or not math.isfinite(y) or x <= 0.0 or y <= 0.0
        for x, y in converted
    ):
        raise ValueError("scaling points must be finite and strictly positive")
    if len(converted) < 2 or len({x for x, _ in converted}) < 2:
        return None
    xs = [math.log(x) for x, _ in converted]
    ys = [math.log(y) for _, y in converted]
    x_mean = _statistics.fmean(xs)
    y_mean = _statistics.fmean(ys)
    sxx = sum((x - x_mean) ** 2 for x in xs)
    if sxx == 0.0:
        return None
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / sxx
    intercept = y_mean - slope * x_mean
    predicted = [intercept + slope * x for x in xs]
    residual_sum = sum((y - estimate) ** 2 for y, estimate in zip(ys, predicted))
    total_sum = sum((y - y_mean) ** 2 for y in ys)
    r_squared = 1.0 if total_sum == 0.0 and residual_sum == 0.0 else (
        0.0 if total_sum == 0.0 else 1.0 - residual_sum / total_sum
    )
    slope_interval: ConfidenceInterval | None = None
    if len(converted) >= 3:
        residual_variance = residual_sum / (len(converted) - 2)
        standard_error = math.sqrt(max(0.0, residual_variance / sxx))
        z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
        slope_interval = ConfidenceInterval(
            confidence,
            slope - z * standard_error,
            slope + z * standard_error,
            "normal-approximation-ols",
        )
    return LogLogScalingFit(
        point_count=len(converted),
        distinct_x_values=len({x for x, _ in converted}),
        slope=slope,
        intercept=intercept,
        r_squared=r_squared,
        slope_interval=slope_interval,
    )


def _config_payload(config: object) -> dict[str, object]:
    if is_dataclass(config):
        payload = asdict(config)
    elif isinstance(config, Mapping):
        payload = dict(config)
    elif hasattr(config, "__dict__"):
        payload = dict(vars(config))
    else:
        raise TypeError("benchmark config must be a dataclass, mapping, or object with __dict__")
    return json.loads(json.dumps(payload, sort_keys=True))


def _config_signature(config: object) -> str:
    return json.dumps(_config_payload(config), sort_keys=True, separators=(",", ":"))


def _numeric_resources(results: Sequence[Any]) -> dict[str, tuple[float, ...]]:
    names = set.intersection(
        *(
            {
                str(name)
                for name, value in result.oracle_resources.items()
                if isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
            }
            for result in results
        )
    )
    return {
        name: tuple(float(result.oracle_resources[name]) for result in results)
        for name in sorted(names)
    }


def benchmark_environment_manifest() -> dict[str, object]:
    versions: dict[str, str | None] = {}
    for distribution in ("q-recon", "numpy", "torch", "z3-solver"):
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[distribution] = None
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "package_versions": versions,
        "github_sha": os.environ.get("GITHUB_SHA"),
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "executable": sys.executable,
        "timing_scope": (
            "Observed wall-clock timings are descriptive unless the runner, warmup, "
            "repetition policy, affinity, and common cost conversion are pinned."
        ),
    }


def summarize_fixed_point_mlp_benchmark_matrix(
    results: Sequence[Any],
    *,
    confidence_level: float = 0.95,
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 1729,
    minimum_seeds_per_config: int = 10,
    minimum_candidate_scales: int = 3,
    include_environment: bool = True,
) -> FixedPointMLPMatrixSummary:
    """Produce a publication-oriented statistical report over benchmark results.

    Duplicate ``(config, seed)`` observations are rejected to prevent accidental
    pseudo-replication. Timings are summarized with deterministic bootstrap
    intervals, Boolean outcomes with Wilson intervals, and scale trends with a
    descriptive log-log fit. The quality gate is an internal evidence gate, not
    a conference-acceptance predictor.
    """

    confidence = _validate_confidence(confidence_level)
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    if minimum_seeds_per_config <= 0 or minimum_candidate_scales <= 0:
        raise ValueError("quality-gate thresholds must be positive")
    sample = tuple(results)
    if not sample:
        raise ValueError("results must be non-empty")

    grouped: dict[str, list[Any]] = defaultdict(list)
    payloads: dict[str, dict[str, object]] = {}
    for result in sample:
        signature = _config_signature(result.config)
        grouped[signature].append(result)
        payloads[signature] = _config_payload(result.config)

    config_summaries: list[FixedPointMLPConfigSummary] = []
    seed_sets: list[tuple[int, ...]] = []
    for signature in sorted(grouped):
        group = sorted(grouped[signature], key=lambda result: int(result.seed))
        seeds = tuple(int(result.seed) for result in group)
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"duplicate seed detected for config {signature}")
        seed_sets.append(seeds)
        candidate_counts = {int(result.candidate_count) for result in group}
        word_populations = {int(result.full_word_population) for result in group}
        if len(candidate_counts) != 1 or len(word_populations) != 1:
            raise ValueError("candidate populations must be constant within a config")

        def scalar(values: Sequence[float], metric: str) -> ScalarSummary:
            return summarize_scalar(
                values,
                confidence_level=confidence,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
                label=f"{signature}:{metric}",
            )

        basis_checked = [
            bool(result.oracle_basis_permutation_verified)
            for result in group
            if result.oracle_basis_permutation_verified is not None
        ]
        z3_runs = [result for result in group if result.z3_report is not None]
        z3_complete_values = [bool(result.z3_report.get("complete")) for result in z3_runs]
        z3_match_values = [
            result.branch_and_bound_z3_solution_sets_match is True for result in z3_runs
        ]
        resource_summaries = {
            name: scalar(values, f"resource:{name}")
            for name, values in _numeric_resources(group).items()
        }
        bbht_values = [
            float(result.bbht_certificate["maximum_expected_phase_oracle_calls"])
            for result in group
            if result.bbht_certificate is not None
            and result.bbht_certificate.get("maximum_expected_phase_oracle_calls") is not None
        ]
        config_summaries.append(
            FixedPointMLPConfigSummary(
                config=payloads[signature],
                seeds=seeds,
                candidate_count=next(iter(candidate_counts)),
                full_word_population=next(iter(word_populations)),
                solution_count=scalar(
                    [float(result.solution_count) for result in group], "solution_count"
                ),
                uniquely_identifiable=summarize_proportion(
                    sum(bool(result.uniquely_identifiable_on_domain) for result in group),
                    len(group),
                    confidence_level=confidence,
                ),
                classical_oracle_agreement=summarize_proportion(
                    sum(bool(result.classical_oracle_solution_sets_match) for result in group),
                    len(group),
                    confidence_level=confidence,
                ),
                branch_and_bound_leaf_fraction=scalar(
                    [float(result.branch_and_bound.leaf_fraction) for result in group],
                    "branch_and_bound_leaf_fraction",
                ),
                branch_and_bound_seconds=scalar(
                    [float(result.branch_and_bound_seconds) for result in group],
                    "branch_and_bound_seconds",
                ),
                oracle_build_seconds=scalar(
                    [float(result.oracle_build_seconds) for result in group],
                    "oracle_build_seconds",
                ),
                basis_permutation_checked=len(basis_checked),
                basis_permutation_skipped=len(group) - len(basis_checked),
                basis_permutation_success=(
                    None
                    if not basis_checked
                    else summarize_proportion(
                        sum(basis_checked), len(basis_checked), confidence_level=confidence
                    )
                ),
                z3_attempted=len(z3_runs),
                z3_complete=(
                    None
                    if not z3_runs
                    else summarize_proportion(
                        sum(z3_complete_values), len(z3_runs), confidence_level=confidence
                    )
                ),
                z3_solution_set_agreement=(
                    None
                    if not z3_runs
                    else summarize_proportion(
                        sum(z3_match_values), len(z3_runs), confidence_level=confidence
                    )
                ),
                z3_seconds=(
                    None
                    if not z3_runs
                    else scalar([float(result.z3_seconds) for result in z3_runs], "z3_seconds")
                ),
                oracle_resource_summaries=resource_summaries,
                bbht_expected_phase_oracle_calls=(
                    None
                    if not bbht_values
                    else scalar(bbht_values, "bbht_expected_phase_oracle_calls")
                ),
            )
        )

    overall_z3 = [result for result in sample if result.z3_report is not None]
    all_z3_complete = (
        None
        if not overall_z3
        else all(bool(result.z3_report.get("complete")) for result in overall_z3)
    )
    all_z3_match = (
        None
        if not overall_z3
        else all(result.branch_and_bound_z3_solution_sets_match is True for result in overall_z3)
    )
    candidate_scales = len({summary.candidate_count for summary in config_summaries})
    minimum_observed_seeds = min(len(summary.seeds) for summary in config_summaries)
    balanced = all(seed_set == seed_sets[0] for seed_set in seed_sets[1:])
    all_classical_match = all(
        bool(result.classical_oracle_solution_sets_match) for result in sample
    )
    enough_seeds = minimum_observed_seeds >= minimum_seeds_per_config
    enough_scales = candidate_scales >= minimum_candidate_scales
    z3_gate = (all_z3_complete is not False) and (all_z3_match is not False)
    quality_gate = BenchmarkQualityGate(
        minimum_required_seeds_per_config=minimum_seeds_per_config,
        observed_minimum_seeds_per_config=minimum_observed_seeds,
        minimum_required_candidate_scales=minimum_candidate_scales,
        observed_candidate_scales=candidate_scales,
        balanced_seed_matrix=balanced,
        enough_seeds=enough_seeds,
        enough_candidate_scales=enough_scales,
        all_classical_oracle_solution_sets_match=all_classical_match,
        all_attempted_z3_runs_complete=all_z3_complete,
        all_completed_z3_solution_sets_match=all_z3_match,
        passed=(
            balanced
            and enough_seeds
            and enough_scales
            and all_classical_match
            and z3_gate
        ),
    )

    branch_fit = fit_loglog_scaling(
        [
            (summary.candidate_count, summary.branch_and_bound_seconds.median)
            for summary in config_summaries
            if summary.branch_and_bound_seconds.median > 0.0
        ],
        confidence_level=confidence,
    )
    oracle_fit = fit_loglog_scaling(
        [
            (summary.candidate_count, summary.oracle_build_seconds.median)
            for summary in config_summaries
            if summary.oracle_build_seconds.median > 0.0
        ],
        confidence_level=confidence,
    )

    return FixedPointMLPMatrixSummary(
        total_instances=len(sample),
        configuration_count=len(config_summaries),
        confidence_level=confidence,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=int(bootstrap_seed),
        configurations=tuple(config_summaries),
        overall_uniquely_identifiable=summarize_proportion(
            sum(bool(result.uniquely_identifiable_on_domain) for result in sample),
            len(sample),
            confidence_level=confidence,
        ),
        overall_classical_oracle_agreement=summarize_proportion(
            sum(bool(result.classical_oracle_solution_sets_match) for result in sample),
            len(sample),
            confidence_level=confidence,
        ),
        overall_z3_complete=(
            None
            if not overall_z3
            else summarize_proportion(
                sum(bool(result.z3_report.get("complete")) for result in overall_z3),
                len(overall_z3),
                confidence_level=confidence,
            )
        ),
        overall_z3_solution_set_agreement=(
            None
            if not overall_z3
            else summarize_proportion(
                sum(
                    result.branch_and_bound_z3_solution_sets_match is True
                    for result in overall_z3
                ),
                len(overall_z3),
                confidence_level=confidence,
            )
        ),
        branch_and_bound_scaling=branch_fit,
        oracle_build_scaling=oracle_fit,
        quality_gate=quality_gate,
        environment=benchmark_environment_manifest() if include_environment else {},
        claim_boundary=(
            "Bootstrap and Wilson intervals quantify finite-seed uncertainty, but CI wall-clock "
            "timings remain descriptive. Hardware-comparable claims require pinned runners, "
            "warmup, repeated timing within each seed, resource-to-cost conversion, and a "
            "predeclared robustness analysis. Passing this gate is necessary, not sufficient, "
            "for a top-tier end-to-end advantage claim."
        ),
    )
