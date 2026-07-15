import pytest

pytest.importorskip("z3")

from qrecon.benchmarks import (
    FixedPointMLPBenchmarkConfig,
    run_fixed_point_mlp_benchmark,
)


def test_benchmark_records_complete_z3_branch_and_oracle_agreement():
    config = FixedPointMLPBenchmarkConfig(
        input_dimension=2,
        hidden_width=2,
        output_dimension=2,
        input_bits=2,
        fractional_bits=0,
        domain_codes=(-1, 0, 1),
        max_basis_verification_bits=4,
    )
    result = run_fixed_point_mlp_benchmark(config, seed=31, use_z3=True)

    assert result.z3_report is not None
    assert result.z3_report["complete"] is True
    assert result.branch_and_bound_z3_solution_sets_match is True
    assert result.classical_oracle_solution_sets_match is True
    assert result.z3_report["solution_count"] == result.solution_count
    assert result.z3_seconds is not None
    assert result.z3_seconds >= 0.0
