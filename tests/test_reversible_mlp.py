import math

import pytest

from qrecon.oracles import (
    FixedPointFormat,
    QuantizedAffineLayer,
    QuantizedNetwork,
    ReversibleCircuit,
    ReversibleIntegerMLPPredicateOracle,
    append_signed_relu_copy,
    compile_structure_preserving_mlp_threshold_oracle,
    estimate_grover_resources,
    simulate_grover,
)


def _decode_signed(word: int, bits: int) -> int:
    return word - (1 << bits) if word >= (1 << (bits - 1)) else word


def _two_layer_model(*, fractional_bits: int = 0) -> QuantizedNetwork:
    input_format = FixedPointFormat(2, fractional_bits=fractional_bits, signed=True)
    weight_format = FixedPointFormat(3, signed=True)
    hidden_format = FixedPointFormat(4, signed=True)
    output_format = FixedPointFormat(6, signed=True)
    first = QuantizedAffineLayer(
        weights=((1, -1), (-1, 1)),
        biases=(0, 0),
        input_format=input_format,
        weight_format=weight_format,
        bias_format=hidden_format,
        output_format=hidden_format,
        activation="relu",
    )
    second = QuantizedAffineLayer(
        weights=((1, 1),),
        biases=(0,),
        input_format=hidden_format,
        weight_format=weight_format,
        bias_format=output_format,
        output_format=output_format,
        activation="identity",
    )
    return QuantizedNetwork(
        (first, second), output_mode="binary_threshold", binary_threshold=3
    )


def test_signed_relu_copy_is_exact_reversible_and_clean_exhaustively():
    for width in range(1, 6):
        source = tuple(range(width))
        target = tuple(range(width, 2 * width))
        circuit = ReversibleCircuit(2 * width)
        gates = append_signed_relu_copy(circuit, source, target)
        expected_toffolis = max(0, width - 1)
        assert sum(gate.kind == "ccx" for gate in gates) == expected_toffolis
        assert sum(gate.kind == "x" for gate in gates) == (2 if width > 1 else 0)

        mask = (1 << width) - 1
        for input_word in range(1 << width):
            relu = max(0, _decode_signed(input_word, width))
            for output_word in range(1 << width):
                state = input_word | (output_word << width)
                forward = circuit.apply_state(state)
                assert forward & mask == input_word
                assert (forward >> width) & mask == output_word ^ relu
                assert circuit.apply_inverse_state(forward) == state


def test_two_layer_integer_mlp_oracle_matches_quantized_reference_and_cleans_work():
    model = _two_layer_model()
    oracle = compile_structure_preserving_mlp_threshold_oracle(model)
    assert oracle.range_report.no_overflow
    assert oracle.verify_basis_permutation()

    for input_word in range(1 << model.input_bits):
        expected = model.evaluate_input_word(input_word)
        values = model.decode_input_word(input_word)
        assert expected == int(abs(values[0] - values[1]) >= 3)
        assert oracle.evaluate_predicate(input_word) == expected
        assert oracle.apply(input_word, 0) == (input_word, expected, 0)
        assert oracle.apply(input_word, 1) == (input_word, 1 ^ expected, 0)
        assert oracle.phase_sign(input_word) == (-1 if expected else 1)


def test_mlp_resource_breakdown_matches_composed_gate_counts_exactly():
    oracle = compile_structure_preserving_mlp_threshold_oracle(_two_layer_model())
    breakdown = oracle.resource_breakdown()
    first = breakdown.first_affine_once
    second = breakdown.second_predicate_once
    total = breakdown.total

    assert breakdown.hidden_neurons == 2
    assert breakdown.hidden_bits == 4
    assert breakdown.relu_x_gates_compute_uncompute == 8
    assert breakdown.relu_toffoli_gates_compute_uncompute == 12
    assert total.x_gates == 2 * first.x_gates + second.x_gates + 8
    assert total.cnot_gates == 2 * first.cnot_gates + second.cnot_gates
    assert total.toffoli_gates == 2 * first.toffoli_gates + second.toffoli_gates + 12
    assert total.logical_qubits == oracle.input_bits + 1 + len(oracle.layout.work_wires)
    assert total.t_count_upper_bound == 7 * total.toffoli_gates


def test_compiled_mlp_phase_oracle_drives_the_expected_grover_curve():
    oracle = compile_structure_preserving_mlp_threshold_oracle(_two_layer_model())
    marked = oracle.marked_inputs()
    assert len(marked) == 2
    iterations = 2
    result = simulate_grover(oracle, iterations)
    theta = math.asin(math.sqrt(len(marked) / (1 << oracle.input_bits)))
    assert result.success_probability == pytest.approx(
        math.sin((2 * iterations + 1) * theta) ** 2
    )
    assert set(result.most_likely_inputs) == set(marked)

    resources = estimate_grover_resources(oracle, iterations)
    assert resources.oracle_calls == iterations
    assert resources.oracle.toffoli_gates == oracle.resource_estimate().toffoli_gates
    assert resources.total_toffoli_gates > iterations * resources.oracle.toffoli_gates


def test_direct_mlp_compiler_rejects_unproved_first_layer_overflow():
    with pytest.raises(OverflowError):
        ReversibleIntegerMLPPredicateOracle(
            first_weights=((4,),),
            first_biases=(0,),
            second_weights=(1,),
            second_bias=0,
            input_bits_per_feature=3,
            hidden_bits=3,
            output_accumulator_bits=6,
        )


def test_mlp_adapter_rejects_fractional_requantization_until_it_is_compiled():
    with pytest.raises(ValueError):
        compile_structure_preserving_mlp_threshold_oracle(
            _two_layer_model(fractional_bits=1)
        )
