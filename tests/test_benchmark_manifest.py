from __future__ import annotations

from dataclasses import dataclass

import pytest

from qrecon.benchmarks.fixed_point_mlp import (
    FixedPointMLPBenchmarkConfig,
    FixedPointMLPBenchmarkResult,
)
from qrecon.benchmarks.manifest import (
    FixedPointMLPBenchmarkManifest,
    collapse_manifest_measurements,
    run_fixed_point_mlp_manifest,
    summarize_manifest_execution,
)


@dataclass(frozen=True)
class DummyBranchAndBoundReport:
    leaf_fraction: float = 0.25

    def to_dict(self) -> dict[str, float]:
        return {"leaf_fraction": self.leaf_fraction}


def _config(input_dimension: int) -> FixedPointMLPBenchmarkConfig:
    return FixedPointMLPBenchmarkConfig(
        input_dimension=input_dimension,
        hidden_width=2,
        output_dimension=1,
        input_bits=2,
        fractional_bits=0,
        domain_codes=(-1, 0, 1),
        max_basis_verification_bits=6,
    )


def _result(
    config: FixedPointMLPBenchmarkConfig,
    seed: int,
    *,
    tick: float = 0.0,
    semantic_shift: int = 0,
) -> FixedPointMLPBenchmarkResult:
    candidate_count = len(config.domain_codes) ** config.input_dimension
    return FixedPointMLPBenchmarkResult(
        config=config,
        seed=seed,
        candidate_count=candidate_count,
        full_word_population=1 << (config.input_dimension * config.input_bits),
        target_codes=(1 + semantic_shift,),
        private_record=tuple(0 for _ in range(config.input_dimension)),
        solution_count=1,
        uniquely_identifiable_on_domain=True,
        branch_and_bound=DummyBranchAndBoundReport(),
        branch_and_bound_seconds=1.0 + tick,
        oracle_resources={"logical_qubits": 5, "toffoli_gates": 7},
        oracle_build_seconds=2.0 + tick,
        oracle_basis_permutation_verified=True,
        classical_oracle_solution_sets_match=True,
        bbht_certificate={"maximum_expected_phase_oracle_calls": 3.0},
        z3_report=None,
        z3_seconds=None,
        branch_and_bound_z3_solution_sets_match=None,
    )


def test_manifest_hash_roundtrip_and_seed_validation():
    manifest = FixedPointMLPBenchmarkManifest(
        configurations=(_config(1),),
        seeds=(1, 2),
        repeats_per_seed=2,
    )
    restored = FixedPointMLPBenchmarkManifest.from_json(manifest.canonical_json())
    assert restored == manifest
    assert restored.sha256 == manifest.sha256

    with pytest.raises(ValueError, match="unique"):
        FixedPointMLPBenchmarkManifest(
            configurations=manifest.configurations,
            seeds=(1, 1),
        )


def test_runner_records_warmups_repeats_and_errors():
    config = _config(1)
    manifest = FixedPointMLPBenchmarkManifest(
        configurations=(config,),
        seeds=(1, 2),
        repeats_per_seed=2,
        warmup_runs=1,
    )
    calls: dict[int, int] = {}

    def runner(config, seed, **kwargs):
        calls[seed] = calls.get(seed, 0) + 1
        if seed == 2 and calls[seed] == 2:
            raise RuntimeError("boom")
        return _result(config, seed, tick=float(calls[seed]))

    execution = run_fixed_point_mlp_manifest(manifest, runner=runner)
    assert len(execution.records) == 6
    assert execution.success_count == 3
    assert execution.failure_count == 1
    assert execution.warmup_failure_count == 0
    error = [record for record in execution.records if record.status == "error"][0]
    assert error.error_type == "RuntimeError"
    assert error.error_sha256


def test_collapse_uses_medians_and_requires_complete_cells():
    config = _config(1)
    manifest = FixedPointMLPBenchmarkManifest(
        configurations=(config,),
        seeds=(1,),
        repeats_per_seed=3,
        warmup_runs=0,
    )
    counter = {"value": 0}
    ticks = (0.0, 10.0, 2.0)

    def runner(config, seed, **kwargs):
        tick = ticks[counter["value"]]
        counter["value"] += 1
        return _result(config, seed, tick=tick)

    execution = run_fixed_point_mlp_manifest(manifest, runner=runner)
    collapsed = collapse_manifest_measurements(execution)
    assert len(collapsed) == 1
    assert collapsed[0].branch_and_bound_seconds == 3.0
    assert collapsed[0].oracle_build_seconds == 4.0
    assert collapsed[0].repeat_count == 3


def test_semantic_mismatch_across_repeats_is_rejected():
    config = _config(1)
    manifest = FixedPointMLPBenchmarkManifest(
        configurations=(config,),
        seeds=(1,),
        repeats_per_seed=2,
        warmup_runs=0,
    )
    counter = {"value": 0}

    def runner(config, seed, **kwargs):
        shift = counter["value"]
        counter["value"] += 1
        return _result(config, seed, semantic_shift=shift)

    execution = run_fixed_point_mlp_manifest(manifest, runner=runner)
    with pytest.raises(ValueError, match="semantic mismatch"):
        collapse_manifest_measurements(execution)


def test_hierarchical_summary_operates_on_collapsed_seed_cells():
    configs = (_config(1), _config(2), _config(3))
    manifest = FixedPointMLPBenchmarkManifest(
        configurations=configs,
        seeds=(1, 2),
        repeats_per_seed=2,
        warmup_runs=0,
    )
    counters: dict[tuple[int, int], int] = {}

    def runner(config, seed, **kwargs):
        key = (config.input_dimension, seed)
        counters[key] = counters.get(key, 0) + 1
        return _result(config, seed, tick=counters[key] / 100.0)

    execution = run_fixed_point_mlp_manifest(manifest, runner=runner)
    report = summarize_manifest_execution(
        execution,
        bootstrap_samples=100,
        minimum_seeds_per_config=2,
        minimum_candidate_scales=3,
    )
    assert report.failed_measurement_runs == 0
    assert report.complete_cells == 6
    assert report.semantic_consistency_failures == 0
    assert report.matrix_summary.quality_gate.passed
