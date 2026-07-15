import pytest

from qrecon.benchmarks import (
    FixedPointMLPBenchmarkConfig,
    build_fixed_point_mlp_instance,
    run_fixed_point_mlp_benchmark,
    run_fixed_point_mlp_benchmark_matrix,
)


def _small_config():
    return FixedPointMLPBenchmarkConfig(
        input_dimension=2,
        hidden_width=2,
        output_dimension=2,
        input_bits=2,
        fractional_bits=0,
        domain_codes=(-1, 0, 1),
        max_basis_verification_bits=4,
    )


def test_generated_instance_is_deterministic_and_bit_exact():
    config = _small_config()
    left = build_fixed_point_mlp_instance(config, seed=17)
    right = build_fixed_point_mlp_instance(config, seed=17)

    assert left.private_record == right.private_record
    assert left.target_codes == right.target_codes
    assert left.hidden_layer == right.hidden_layer
    assert left.output_layer == right.output_layer
    assert left.target_codes == left.output_layer.evaluate_codes(
        left.hidden_layer.evaluate_codes(left.private_record)
    )


def test_benchmark_closes_classical_oracle_and_unknown_k_reports():
    config = _small_config()
    result = run_fixed_point_mlp_benchmark(config, seed=23)

    assert result.candidate_count == 9
    assert result.full_word_population == 16
    assert result.private_record in result.branch_and_bound.solutions
    assert result.solution_count >= 1
    assert result.classical_oracle_solution_sets_match
    assert result.oracle_basis_permutation_verified is True
    assert result.oracle_resources["logical_qubits"] > 0
    assert result.oracle_resources["toffoli_gates"] > 0
    assert result.bbht_certificate is not None
    assert result.bbht_certificate["certified_minimum_success"] >= 0.9
    assert result.z3_report is None
    assert result.branch_and_bound_z3_solution_sets_match is None

    payload = result.to_dict()
    assert payload["seed"] == 23
    assert payload["classical_oracle_solution_sets_match"] is True
    assert payload["branch_and_bound"]["solution_count"] == result.solution_count


def test_benchmark_matrix_preserves_config_seed_cartesian_product():
    configs = (
        _small_config(),
        FixedPointMLPBenchmarkConfig(
            input_dimension=2,
            hidden_width=3,
            output_dimension=1,
            input_bits=2,
            fractional_bits=0,
            domain_codes=(-1, 0, 1),
            max_basis_verification_bits=4,
        ),
    )
    results = run_fixed_point_mlp_benchmark_matrix(configs, seeds=(3, 5))
    assert len(results) == 4
    assert [(result.config.hidden_width, result.seed) for result in results] == [
        (2, 3),
        (2, 5),
        (3, 3),
        (3, 5),
    ]
    assert all(result.classical_oracle_solution_sets_match for result in results)


def test_larger_register_skips_exhaustive_basis_check_but_not_solution_audit():
    config = FixedPointMLPBenchmarkConfig(
        input_dimension=3,
        hidden_width=2,
        output_dimension=2,
        input_bits=3,
        fractional_bits=1,
        domain_codes=(-2, -1, 0, 1),
        max_basis_verification_bits=8,
    )
    result = run_fixed_point_mlp_benchmark(config, seed=7)
    assert result.full_word_population == 512
    assert result.oracle_basis_permutation_verified is None
    assert result.classical_oracle_solution_sets_match
    assert result.bbht_certificate is not None


def test_benchmark_configuration_contract_rejects_invalid_spaces():
    with pytest.raises(ValueError):
        FixedPointMLPBenchmarkConfig(input_dimension=0)
    with pytest.raises(ValueError):
        FixedPointMLPBenchmarkConfig(input_bits=1)
    with pytest.raises(ValueError):
        FixedPointMLPBenchmarkConfig(domain_codes=())
    with pytest.raises(ValueError):
        FixedPointMLPBenchmarkConfig(domain_codes=(0, 0))
    with pytest.raises(OverflowError):
        FixedPointMLPBenchmarkConfig(input_bits=2, domain_codes=(2,))
    with pytest.raises(ValueError):
        run_fixed_point_mlp_benchmark_matrix((), (1,))
