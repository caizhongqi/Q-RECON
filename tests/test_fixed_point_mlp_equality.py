import pytest

from qrecon.oracles import (
    FixedPointFormat,
    QuantizedAffineLayer,
    ReversibleFixedPointMLPEqualityOracle,
    compile_structure_preserving_fixed_point_mlp_equality_oracle,
    simulate_grover,
    solve_fixed_point_mlp_exact_output,
)
from qrecon.theory import grover_success, optimal_standard_grover_iterations


def _model():
    input_format = FixedPointFormat(2, 0, True)
    hidden_format = FixedPointFormat(4, 0, True)
    hidden = QuantizedAffineLayer(
        weights=((1, -1), (1, 1)),
        biases=(0, 0),
        input_format=input_format,
        weight_format=FixedPointFormat(2, 0, True),
        bias_format=FixedPointFormat(4, 0, True),
        output_format=hidden_format,
        activation="relu",
    )
    output = QuantizedAffineLayer(
        weights=((1, 2), (-1, 1)),
        biases=(0, 1),
        input_format=hidden_format,
        weight_format=FixedPointFormat(3, 0, True),
        bias_format=FixedPointFormat(5, 0, True),
        output_format=FixedPointFormat(5, 0, True),
        activation="identity",
    )
    domain = tuple(range(input_format.min_code, input_format.max_code + 1))
    return hidden, output, (domain, domain)


def test_exact_mlp_equality_oracle_matches_reference_and_cleans_all_work():
    hidden, output, domains = _model()
    private_record = (1, -2)
    target = output.evaluate_codes(hidden.evaluate_codes(private_record))
    oracle = ReversibleFixedPointMLPEqualityOracle(hidden, output, target)

    assert oracle.verify_basis_permutation()
    marked = oracle.marked_inputs()
    assert marked == (oracle.encode_inputs(private_record),)

    for word in range(1 << oracle.input_bits):
        expected = int(word in marked)
        assert oracle.evaluate_predicate(word) == expected
        assert oracle.apply(word, 0) == (word, expected, 0)
        assert oracle.apply(word, 1) == (word, 1 ^ expected, 0)
        assert oracle.phase_sign(word) == (-1 if expected else 1)


def test_same_exact_output_task_matches_complete_classical_branch_and_bound():
    hidden, output, domains = _model()
    target = output.evaluate_codes(hidden.evaluate_codes((1, -2)))
    oracle = compile_structure_preserving_fixed_point_mlp_equality_oracle(
        hidden, output, target
    )
    classical = solve_fixed_point_mlp_exact_output(
        hidden, output, target, domains=domains
    )

    encoded_classical = tuple(sorted(oracle.encode_inputs(row) for row in classical.solutions))
    assert encoded_classical == oracle.marked_inputs()
    assert classical.solutions == ((1, -2),)
    assert classical.stopped_early is False


def test_compiled_equality_oracle_drives_exact_grover_semantics():
    hidden, output, _ = _model()
    target = output.evaluate_codes(hidden.evaluate_codes((1, -2)))
    oracle = ReversibleFixedPointMLPEqualityOracle(hidden, output, target)
    population = 1 << oracle.input_bits
    marked = len(oracle.marked_inputs())
    iterations = optimal_standard_grover_iterations(population, marked)
    assert iterations is not None

    result = simulate_grover(oracle, iterations)
    assert result.success_probability == pytest.approx(
        grover_success(population, marked, iterations), abs=1e-12
    )
    assert result.most_likely_inputs == oracle.marked_inputs()


def test_equality_composition_has_auditable_nonzero_resources():
    hidden, output, _ = _model()
    target = output.evaluate_codes(hidden.evaluate_codes((1, -2)))
    oracle = ReversibleFixedPointMLPEqualityOracle(hidden, output, target)
    resources = oracle.resource_estimate(phase_kickback=True)

    assert resources.input_qubits == oracle.input_bits
    assert resources.output_qubits == 1
    assert resources.peak_clean_ancillas == len(oracle.layout.work_wires)
    assert resources.logical_qubits == oracle.circuit.num_qubits
    assert resources.toffoli_gates > 0
    assert resources.t_count_upper_bound == 7 * resources.toffoli_gates
    assert resources.logical_depth_upper_bound > 0


def test_target_contract_is_rejected_before_circuit_construction():
    hidden, output, _ = _model()
    with pytest.raises(ValueError, match="one target code"):
        ReversibleFixedPointMLPEqualityOracle(hidden, output, (0,))
    with pytest.raises(OverflowError):
        ReversibleFixedPointMLPEqualityOracle(
            hidden,
            output,
            (output.output_format.max_code + 1, 0),
        )
