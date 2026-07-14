import math

import pytest

from qrecon.oracles import FixedPointFormat, QuantizedAffineLayer, QuantizedNetwork
from qrecon.oracles.grover import simulate_grover
from qrecon.oracles.mlp import (
    ReversibleIntegerMLPPredicateOracle,
    append_signed_relu_copy,
    compile_structure_preserving_mlp_threshold_oracle,
)
from qrecon.oracles.reversible import ReversibleCircuit, pack_register, unpack_register


def _manual_relu_network(x0: int, x1: int) -> int:
    first = max(0, x0 - x1)
    second = max(0, -x0 + x1)
    return int(first - second >= 0)


def test_signed_relu_copy_is_exact_clean_and_self_inverse():
    width = 4
    source = tuple(range(width))
    target = tuple(range(width, 2 * width))
    circuit = ReversibleCircuit(2 * width)
    gates = append_signed_relu_copy(circuit, source, target)
    assert len(gates) == width + 1
    counts = circuit.gate_counts()
    assert counts["x"] == 2
    assert counts["ccx"] == width - 1

    for source_word in range(1 << width):
        signed = source_word - (1 << width) if source_word >= (1 << (width - 1)) else source_word
        expected = max(0, signed)
        for target_word in range(1 << width):
            state = pack_register(0, source, source_word)
            state = pack_register(state, target, target_word)
            output = circuit.apply_state(state)
            assert unpack_register(output, source) == source_word
            assert unpack_register(output, target) == (target_word ^ expected)
            assert circuit.apply_inverse_state(output) == state


def test_two_layer_integer_mlp_predicate_is_clean_and_exact_exhaustively():
    oracle = ReversibleIntegerMLPPredicateOracle(
        first_weights=((1, -1), (-1, 1)),
        first_biases=(0, 0),
        second_weights=(1, -1),
        second_bias=0,
        threshold=0,
        input_bits_per_feature=2,
        hidden_bits=4,
        output_accumulator_bits=5,
        signed_inputs=True,
    )
    assert oracle.range_report.no_overflow
    assert oracle.verify_basis_permutation()

    for input_word in range(1 << oracle.input_bits):
        x0, x1 = oracle.decode_input_word(input_word)
        expected = _manual_relu_network(x0, x1)
        assert oracle.hidden_activations(input_word) == (
            max(0, x0 - x1),
            max(0, -x0 + x1),
        )
        assert oracle.evaluate_predicate(input_word) == expected
        assert oracle.apply(input_word, 0) == (input_word, expected, 0)
        assert oracle.apply(input_word, 1) == (input_word, 1 ^ expected, 0)
        assert oracle.phase_sign(input_word) == (-1 if expected else 1)

    result = simulate_grover(oracle, 1)
    marked = len(oracle.marked_inputs())
    theta = math.asin(math.sqrt(marked / (1 << oracle.input_bits)))
    assert result.success_probability == pytest.approx(math.sin(3 * theta) ** 2)


def test_mlp_quantized_network_adapter_matches_reference_for_every_candidate():
    input_format = FixedPointFormat(2, signed=True)
    hidden_format = FixedPointFormat(4, signed=True)
    output_format = FixedPointFormat(5, signed=True)
    first = QuantizedAffineLayer(
        weights=((1, -1), (-1, 1)),
        biases=(0, 0),
        input_format=input_format,
        weight_format=FixedPointFormat(3, signed=True),
        bias_format=hidden_format,
        output_format=hidden_format,
        activation="relu",
    )
    second = QuantizedAffineLayer(
        weights=((1, -1),),
        biases=(0,),
        input_format=hidden_format,
        weight_format=FixedPointFormat(3, signed=True),
        bias_format=output_format,
        output_format=output_format,
        activation="identity",
    )
    model = QuantizedNetwork((first, second), output_mode="binary_threshold")
    oracle = compile_structure_preserving_mlp_threshold_oracle(model)
    for input_word in range(1 << model.input_bits):
        expected = model.evaluate_input_word(input_word)
        assert oracle.evaluate_predicate(input_word) == expected
        assert oracle.apply(input_word, 0) == (input_word, expected, 0)


def test_mlp_resource_breakdown_counts_relu_compute_and_uncompute():
    hidden_neurons = 2
    hidden_bits = 4
    oracle = ReversibleIntegerMLPPredicateOracle(
        first_weights=((1, 0), (0, 1)),
        first_biases=(0, 0),
        second_weights=(1, 1),
        second_bias=0,
        input_bits_per_feature=2,
        hidden_bits=hidden_bits,
        output_accumulator_bits=6,
        signed_inputs=True,
    )
    breakdown = oracle.resource_breakdown()
    assert breakdown.hidden_neurons == hidden_neurons
    assert breakdown.hidden_bits == hidden_bits
    assert breakdown.relu_x_gates_compute_uncompute == 4 * hidden_neurons
    assert breakdown.relu_toffoli_gates_compute_uncompute == (
        2 * hidden_neurons * (hidden_bits - 1)
    )
    assert breakdown.total.toffoli_gates > breakdown.second_predicate_once.toffoli_gates
    assert breakdown.total.peak_clean_ancillas == len(oracle.layout.work_wires)


def test_mlp_lowering_rejects_wrong_activation_contract():
    input_format = FixedPointFormat(2, signed=True)
    hidden_format = FixedPointFormat(4, signed=True)
    first = QuantizedAffineLayer(
        weights=((1,),),
        biases=(0,),
        input_format=input_format,
        weight_format=FixedPointFormat(2, signed=True),
        bias_format=hidden_format,
        output_format=hidden_format,
        activation="identity",
    )
    second = QuantizedAffineLayer(
        weights=((1,),),
        biases=(0,),
        input_format=hidden_format,
        weight_format=FixedPointFormat(2, signed=True),
        bias_format=hidden_format,
        output_format=hidden_format,
    )
    model = QuantizedNetwork((first, second), output_mode="binary_threshold")
    with pytest.raises(ValueError):
        compile_structure_preserving_mlp_threshold_oracle(model)
