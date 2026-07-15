from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from qrecon.benchmarks.statistics import (
    fit_loglog_scaling,
    summarize_fixed_point_mlp_benchmark_matrix,
    summarize_proportion,
    summarize_scalar,
)


def test_scalar_bootstrap_is_deterministic_and_contains_point_estimates():
    left = summarize_scalar(
        [1.0, 2.0, 3.0, 4.0], bootstrap_samples=500, bootstrap_seed=9
    )
    right = summarize_scalar(
        [1.0, 2.0, 3.0, 4.0], bootstrap_samples=500, bootstrap_seed=9
    )
    assert left == right
    assert left.mean == pytest.approx(2.5)
    assert left.median == pytest.approx(2.5)
    assert left.mean_interval.lower <= left.mean <= left.mean_interval.upper
    assert left.median_interval.lower <= left.median <= left.median_interval.upper


def test_wilson_interval_handles_boundary_rates_without_zero_width():
    zero = summarize_proportion(0, 10)
    one = summarize_proportion(10, 10)
    assert zero.rate == 0.0
    assert zero.interval.upper > 0.0
    assert one.rate == 1.0
    assert one.interval.lower < 1.0
    assert 0.0 <= zero.interval.lower <= zero.interval.upper <= 1.0


def test_exact_power_law_has_expected_loglog_slope():
    fit = fit_loglog_scaling([(1, 3), (2, 12), (4, 48), (8, 192)])
    assert fit is not None
    assert fit.slope == pytest.approx(2.0)
    assert fit.r_squared == pytest.approx(1.0)
    assert fit.slope_interval is not None


@dataclass(frozen=True)
class DummyConfig:
    name: str
    scale: int


def _result(config, seed, candidate_count, *, z3=True):
    return SimpleNamespace(
        config=config,
        seed=seed,
        candidate_count=candidate_count,
        full_word_population=candidate_count * 2,
        solution_count=1 if seed % 2 else 2,
        uniquely_identifiable_on_domain=bool(seed % 2),
        branch_and_bound=SimpleNamespace(leaf_fraction=0.1 * seed),
        branch_and_bound_seconds=0.001 * candidate_count * (1 + seed / 100),
        oracle_build_seconds=0.002 * candidate_count * (1 + seed / 100),
        oracle_resources={
            "logical_qubits": 8 + candidate_count,
            "toffoli_gates": 3 * candidate_count,
            "backend": "dummy",
        },
        oracle_basis_permutation_verified=True,
        classical_oracle_solution_sets_match=True,
        bbht_certificate={
            "maximum_expected_phase_oracle_calls": candidate_count**0.5
        },
        z3_report={"complete": True, "termination": "exhausted"} if z3 else None,
        z3_seconds=0.003 * candidate_count if z3 else None,
        branch_and_bound_z3_solution_sets_match=True if z3 else None,
    )


def test_matrix_summary_enforces_balanced_replicates_and_quality_gate():
    results = []
    for index, scale in enumerate((4, 16, 64)):
        config = DummyConfig(f"c{index}", scale)
        results.extend(_result(config, seed, scale) for seed in (1, 2, 3))
    report = summarize_fixed_point_mlp_benchmark_matrix(
        results,
        bootstrap_samples=200,
        minimum_seeds_per_config=3,
        minimum_candidate_scales=3,
        include_environment=False,
    )
    assert report.total_instances == 9
    assert report.configuration_count == 3
    assert report.quality_gate.passed
    assert report.quality_gate.balanced_seed_matrix
    assert report.overall_classical_oracle_agreement.rate == 1.0
    assert report.overall_z3_complete is not None
    assert report.overall_z3_complete.rate == 1.0
    assert report.branch_and_bound_scaling is not None
    assert report.branch_and_bound_scaling.slope == pytest.approx(1.0, abs=0.03)
    payload = report.to_dict()
    assert payload["quality_gate"]["passed"] is True
    assert payload["environment"] == {}


def test_duplicate_config_seed_is_rejected_as_pseudoreplication():
    config = DummyConfig("same", 4)
    result = _result(config, 1, 4)
    with pytest.raises(ValueError, match="duplicate seed"):
        summarize_fixed_point_mlp_benchmark_matrix(
            [result, result], bootstrap_samples=20
        )


def test_quality_gate_stays_false_for_underpowered_matrix():
    config = DummyConfig("tiny", 4)
    report = summarize_fixed_point_mlp_benchmark_matrix(
        [_result(config, 1, 4, z3=False)],
        bootstrap_samples=20,
        minimum_seeds_per_config=5,
        minimum_candidate_scales=3,
        include_environment=False,
    )
    assert not report.quality_gate.passed
    assert report.overall_z3_complete is None
