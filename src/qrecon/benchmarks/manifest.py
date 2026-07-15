from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from statistics import median
from time import perf_counter
from typing import Any, Literal

from .fixed_point_mlp import (
    FixedPointMLPBenchmarkConfig,
    FixedPointMLPBenchmarkResult,
    run_fixed_point_mlp_benchmark,
)
from .statistics import (
    FixedPointMLPMatrixSummary,
    ProportionSummary,
    ScalarSummary,
    summarize_fixed_point_mlp_benchmark_matrix,
    summarize_proportion,
    summarize_scalar,
)

RunPhase = Literal["warmup", "measurement"]
RunStatus = Literal["success", "error"]


@dataclass(frozen=True)
class FixedPointMLPBenchmarkManifest:
    configurations: tuple[FixedPointMLPBenchmarkConfig, ...]
    seeds: tuple[int, ...]
    repeats_per_seed: int = 5
    warmup_runs: int = 1
    use_z3: bool = False
    z3_timeout_ms: int | None = None
    target_success: float = 0.75
    bbht_growth_factor: float = 8.0 / 7.0
    bbht_attempts_per_stage: int = 1
    bbht_max_stages: int | None = None
    schema_version: str = "qrecon.fixed-point-mlp-manifest.v1"

    def __post_init__(self) -> None:
        if not self.configurations:
            raise ValueError("configurations must be non-empty")
        if not self.seeds:
            raise ValueError("seeds must be non-empty")
        normalized_seeds = tuple(int(seed) for seed in self.seeds)
        if len(set(normalized_seeds)) != len(normalized_seeds):
            raise ValueError("manifest seeds must be unique")
        if self.repeats_per_seed <= 0:
            raise ValueError("repeats_per_seed must be positive")
        if self.warmup_runs < 0:
            raise ValueError("warmup_runs must be non-negative")
        if self.z3_timeout_ms is not None and self.z3_timeout_ms <= 0:
            raise ValueError("z3_timeout_ms must be positive or None")
        if (
            not math.isfinite(self.target_success)
            or not 0.0 < self.target_success < 1.0
        ):
            raise ValueError("target_success must lie strictly between zero and one")
        if (
            not math.isfinite(self.bbht_growth_factor)
            or self.bbht_growth_factor <= 1.0
        ):
            raise ValueError("bbht_growth_factor must exceed one")
        if self.bbht_attempts_per_stage <= 0:
            raise ValueError("bbht_attempts_per_stage must be positive")
        if self.bbht_max_stages is not None and self.bbht_max_stages <= 0:
            raise ValueError("bbht_max_stages must be positive or None")
        object.__setattr__(self, "seeds", normalized_seeds)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "configurations": [asdict(config) for config in self.configurations],
            "seeds": list(self.seeds),
            "repeats_per_seed": self.repeats_per_seed,
            "warmup_runs": self.warmup_runs,
            "use_z3": self.use_z3,
            "z3_timeout_ms": self.z3_timeout_ms,
            "target_success": self.target_success,
            "bbht_growth_factor": self.bbht_growth_factor,
            "bbht_attempts_per_stage": self.bbht_attempts_per_stage,
            "bbht_max_stages": self.bbht_max_stages,
        }

    @classmethod
    def from_dict(
        cls, payload: dict[str, object]
    ) -> "FixedPointMLPBenchmarkManifest":
        version = str(payload.get("schema_version", ""))
        expected = "qrecon.fixed-point-mlp-manifest.v1"
        if version != expected:
            raise ValueError(f"unsupported manifest schema: {version!r}")
        raw_configs = payload.get("configurations")
        if not isinstance(raw_configs, list):
            raise ValueError("configurations must be a list")
        configs: list[FixedPointMLPBenchmarkConfig] = []
        for item in raw_configs:
            if not isinstance(item, dict):
                raise ValueError("each configuration must be an object")
            config_payload = dict(item)
            if isinstance(config_payload.get("domain_codes"), list):
                config_payload["domain_codes"] = tuple(config_payload["domain_codes"])
            configs.append(FixedPointMLPBenchmarkConfig(**config_payload))
        raw_seeds = payload.get("seeds")
        if not isinstance(raw_seeds, list):
            raise ValueError("seeds must be a list")
        return cls(
            configurations=tuple(configs),
            seeds=tuple(int(seed) for seed in raw_seeds),
            repeats_per_seed=int(payload.get("repeats_per_seed", 5)),
            warmup_runs=int(payload.get("warmup_runs", 1)),
            use_z3=bool(payload.get("use_z3", False)),
            z3_timeout_ms=(
                None
                if payload.get("z3_timeout_ms") is None
                else int(payload["z3_timeout_ms"])
            ),
            target_success=float(payload.get("target_success", 0.75)),
            bbht_growth_factor=float(payload.get("bbht_growth_factor", 8.0 / 7.0)),
            bbht_attempts_per_stage=int(payload.get("bbht_attempts_per_stage", 1)),
            bbht_max_stages=(
                None
                if payload.get("bbht_max_stages") is None
                else int(payload["bbht_max_stages"])
            ),
        )

    @classmethod
    def from_json(cls, text: str) -> "FixedPointMLPBenchmarkManifest":
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError("manifest JSON must contain an object")
        return cls.from_dict(payload)

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class BenchmarkRunRecord:
    manifest_sha256: str
    configuration_index: int
    seed: int
    phase: RunPhase
    repeat_index: int
    status: RunStatus
    total_seconds: float
    result: FixedPointMLPBenchmarkResult | None
    error_type: str | None = None
    error_message: str | None = None
    error_sha256: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest_sha256": self.manifest_sha256,
            "configuration_index": self.configuration_index,
            "seed": self.seed,
            "phase": self.phase,
            "repeat_index": self.repeat_index,
            "status": self.status,
            "total_seconds": self.total_seconds,
            "result": None if self.result is None else self.result.to_dict(),
            "error_type": self.error_type,
            "error_message": self.error_message,
            "error_sha256": self.error_sha256,
        }


@dataclass(frozen=True)
class ManifestExecution:
    manifest: FixedPointMLPBenchmarkManifest
    records: tuple[BenchmarkRunRecord, ...]
    started_at_utc: str
    finished_at_utc: str

    @property
    def measurement_records(self) -> tuple[BenchmarkRunRecord, ...]:
        return tuple(record for record in self.records if record.phase == "measurement")

    @property
    def success_count(self) -> int:
        return sum(record.status == "success" for record in self.measurement_records)

    @property
    def failure_count(self) -> int:
        return sum(record.status == "error" for record in self.measurement_records)

    @property
    def warmup_failure_count(self) -> int:
        return sum(
            record.status == "error" and record.phase == "warmup"
            for record in self.records
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest": self.manifest.to_dict(),
            "manifest_sha256": self.manifest.sha256,
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": self.finished_at_utc,
            "measurement_success_count": self.success_count,
            "measurement_failure_count": self.failure_count,
            "warmup_failure_count": self.warmup_failure_count,
            "records": [record.to_dict() for record in self.records],
        }


@dataclass(frozen=True)
class CollapsedFixedPointMLPBenchmarkResult:
    reference: FixedPointMLPBenchmarkResult
    repeat_count: int
    branch_and_bound_seconds: float
    oracle_build_seconds: float
    z3_seconds: float | None
    total_seconds: float

    @property
    def config(self) -> FixedPointMLPBenchmarkConfig:
        return self.reference.config

    @property
    def seed(self) -> int:
        return self.reference.seed

    def __getattr__(self, name: str) -> Any:
        return getattr(self.reference, name)

    def to_dict(self) -> dict[str, object]:
        payload = self.reference.to_dict()
        payload["repeat_count"] = self.repeat_count
        payload["branch_and_bound_seconds"] = self.branch_and_bound_seconds
        payload["oracle_build_seconds"] = self.oracle_build_seconds
        payload["z3_seconds"] = self.z3_seconds
        payload["total_seconds"] = self.total_seconds
        return payload


@dataclass(frozen=True)
class ManifestStatisticalReport:
    manifest_sha256: str
    expected_measurement_runs: int
    observed_measurement_runs: int
    successful_measurement_runs: int
    failed_measurement_runs: int
    warmup_failure_runs: int
    measurement_success: ProportionSummary
    complete_cells: int
    incomplete_cells: int
    semantic_consistency_failures: int
    within_seed_branch_and_bound_seconds: ScalarSummary
    within_seed_oracle_build_seconds: ScalarSummary
    within_seed_total_seconds: ScalarSummary
    matrix_summary: FixedPointMLPMatrixSummary

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest_sha256": self.manifest_sha256,
            "expected_measurement_runs": self.expected_measurement_runs,
            "observed_measurement_runs": self.observed_measurement_runs,
            "successful_measurement_runs": self.successful_measurement_runs,
            "failed_measurement_runs": self.failed_measurement_runs,
            "warmup_failure_runs": self.warmup_failure_runs,
            "measurement_success": self.measurement_success.to_dict(),
            "complete_cells": self.complete_cells,
            "incomplete_cells": self.incomplete_cells,
            "semantic_consistency_failures": self.semantic_consistency_failures,
            "within_seed_branch_and_bound_seconds": (
                self.within_seed_branch_and_bound_seconds.to_dict()
            ),
            "within_seed_oracle_build_seconds": (
                self.within_seed_oracle_build_seconds.to_dict()
            ),
            "within_seed_total_seconds": self.within_seed_total_seconds.to_dict(),
            "matrix_summary": self.matrix_summary.to_dict(),
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _error_payload(exc: BaseException) -> tuple[str, str, str]:
    error_type = type(exc).__name__
    message = str(exc)
    digest = hashlib.sha256(f"{error_type}\n{message}".encode("utf-8")).hexdigest()
    return error_type, message, digest


def _runner_kwargs(manifest: FixedPointMLPBenchmarkManifest) -> dict[str, object]:
    return {
        "use_z3": manifest.use_z3,
        "z3_timeout_ms": manifest.z3_timeout_ms,
        "target_success": manifest.target_success,
        "bbht_growth_factor": manifest.bbht_growth_factor,
        "bbht_attempts_per_stage": manifest.bbht_attempts_per_stage,
        "bbht_max_stages": manifest.bbht_max_stages,
    }


def run_fixed_point_mlp_manifest(
    manifest: FixedPointMLPBenchmarkManifest,
    *,
    runner: Callable[..., FixedPointMLPBenchmarkResult] = run_fixed_point_mlp_benchmark,
    continue_on_error: bool = True,
) -> ManifestExecution:
    """Execute warmups and repeated measurements while preserving every failure."""

    started = _utc_now()
    records: list[BenchmarkRunRecord] = []
    kwargs = _runner_kwargs(manifest)
    for config_index, config in enumerate(manifest.configurations):
        for seed in manifest.seeds:
            for phase, repetitions in (
                ("warmup", manifest.warmup_runs),
                ("measurement", manifest.repeats_per_seed),
            ):
                for repeat_index in range(repetitions):
                    start = perf_counter()
                    try:
                        result = runner(config, seed, **kwargs)
                    except Exception as exc:
                        elapsed = perf_counter() - start
                        error_type, message, digest = _error_payload(exc)
                        records.append(
                            BenchmarkRunRecord(
                                manifest_sha256=manifest.sha256,
                                configuration_index=config_index,
                                seed=seed,
                                phase=phase,
                                repeat_index=repeat_index,
                                status="error",
                                total_seconds=elapsed,
                                result=None,
                                error_type=error_type,
                                error_message=message,
                                error_sha256=digest,
                            )
                        )
                        if not continue_on_error:
                            raise
                    else:
                        elapsed = perf_counter() - start
                        records.append(
                            BenchmarkRunRecord(
                                manifest_sha256=manifest.sha256,
                                configuration_index=config_index,
                                seed=seed,
                                phase=phase,
                                repeat_index=repeat_index,
                                status="success",
                                total_seconds=elapsed,
                                result=(result if phase == "measurement" else None),
                            )
                        )
    return ManifestExecution(manifest, tuple(records), started, _utc_now())


def _semantic_payload(result: FixedPointMLPBenchmarkResult) -> dict[str, object]:
    payload = result.to_dict()
    for key in ("branch_and_bound_seconds", "oracle_build_seconds", "z3_seconds"):
        payload.pop(key, None)
    return payload


def _semantic_hash(result: FixedPointMLPBenchmarkResult) -> str:
    encoded = json.dumps(
        _semantic_payload(result),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def collapse_manifest_measurements(
    execution: ManifestExecution,
    *,
    require_complete_cells: bool = True,
) -> tuple[CollapsedFixedPointMLPBenchmarkResult, ...]:
    """Collapse repeated measurements to one median-timed result per config/seed."""

    grouped: dict[tuple[int, int], list[BenchmarkRunRecord]] = defaultdict(list)
    for record in execution.measurement_records:
        grouped[(record.configuration_index, record.seed)].append(record)

    expected_cells = {
        (config_index, seed)
        for config_index in range(len(execution.manifest.configurations))
        for seed in execution.manifest.seeds
    }
    if set(grouped) != expected_cells:
        raise ValueError("measurement records do not cover every manifest cell")

    collapsed: list[CollapsedFixedPointMLPBenchmarkResult] = []
    for key in sorted(grouped):
        records = grouped[key]
        successful = [record for record in records if record.status == "success"]
        if (
            require_complete_cells
            and len(successful) != execution.manifest.repeats_per_seed
        ):
            raise ValueError(f"incomplete measurement cell: {key}")
        if not successful:
            raise ValueError(f"measurement cell has no successful repeat: {key}")
        results = [record.result for record in successful]
        if any(result is None for result in results):
            raise ValueError("successful measurement record is missing its result")
        typed_results = [result for result in results if result is not None]
        fingerprints = {_semantic_hash(result) for result in typed_results}
        if len(fingerprints) != 1:
            raise ValueError(f"semantic mismatch across repeats in cell {key}")
        reference = typed_results[0]
        z3_values = [
            float(result.z3_seconds)
            for result in typed_results
            if result.z3_seconds is not None
        ]
        collapsed.append(
            CollapsedFixedPointMLPBenchmarkResult(
                reference=reference,
                repeat_count=len(typed_results),
                branch_and_bound_seconds=float(
                    median(result.branch_and_bound_seconds for result in typed_results)
                ),
                oracle_build_seconds=float(
                    median(result.oracle_build_seconds for result in typed_results)
                ),
                z3_seconds=(None if not z3_values else float(median(z3_values))),
                total_seconds=float(
                    median(record.total_seconds for record in successful)
                ),
            )
        )
    return tuple(collapsed)


def summarize_manifest_execution(
    execution: ManifestExecution,
    *,
    confidence_level: float = 0.95,
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 1729,
    minimum_seeds_per_config: int = 10,
    minimum_candidate_scales: int = 3,
    require_complete_cells: bool = True,
) -> ManifestStatisticalReport:
    """Hierarchical report: median within seed, then uncertainty across seeds."""

    measurements = execution.measurement_records
    expected_runs = (
        len(execution.manifest.configurations)
        * len(execution.manifest.seeds)
        * execution.manifest.repeats_per_seed
    )
    grouped: dict[tuple[int, int], list[BenchmarkRunRecord]] = defaultdict(list)
    for record in measurements:
        grouped[(record.configuration_index, record.seed)].append(record)
    complete_cells = sum(
        len([record for record in records if record.status == "success"])
        == execution.manifest.repeats_per_seed
        for records in grouped.values()
    )
    total_cells = len(execution.manifest.configurations) * len(execution.manifest.seeds)

    semantic_failures = 0
    for records in grouped.values():
        results = [record.result for record in records if record.result is not None]
        if results and len({_semantic_hash(result) for result in results}) != 1:
            semantic_failures += 1

    collapsed = collapse_manifest_measurements(
        execution, require_complete_cells=require_complete_cells
    )
    matrix = summarize_fixed_point_mlp_benchmark_matrix(
        collapsed,
        confidence_level=confidence_level,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        minimum_seeds_per_config=minimum_seeds_per_config,
        minimum_candidate_scales=minimum_candidate_scales,
    )
    timing_label = f"manifest:{execution.manifest.sha256}"
    return ManifestStatisticalReport(
        manifest_sha256=execution.manifest.sha256,
        expected_measurement_runs=expected_runs,
        observed_measurement_runs=len(measurements),
        successful_measurement_runs=execution.success_count,
        failed_measurement_runs=execution.failure_count,
        warmup_failure_runs=execution.warmup_failure_count,
        measurement_success=summarize_proportion(
            execution.success_count,
            max(1, len(measurements)),
            confidence_level=confidence_level,
        ),
        complete_cells=complete_cells,
        incomplete_cells=total_cells - complete_cells,
        semantic_consistency_failures=semantic_failures,
        within_seed_branch_and_bound_seconds=summarize_scalar(
            [result.branch_and_bound_seconds for result in collapsed],
            confidence_level=confidence_level,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
            label=f"{timing_label}:branch",
        ),
        within_seed_oracle_build_seconds=summarize_scalar(
            [result.oracle_build_seconds for result in collapsed],
            confidence_level=confidence_level,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
            label=f"{timing_label}:oracle",
        ),
        within_seed_total_seconds=summarize_scalar(
            [result.total_seconds for result in collapsed],
            confidence_level=confidence_level,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
            label=f"{timing_label}:total",
        ),
        matrix_summary=matrix,
    )
