import math

import pytest

from qrecon.oracles import FixedPointFormat, QuantizedAffineLayer, QuantizedNetwork
from qrecon.oracles.comparators import (
    ReversibleIntegerAffineEqualityOracle,
    append_equality_to_constant,
    compile_structure_preserving_affine_equality_oracle,
)
from qrecon.oracles.grover import simulate_grover
from qrecon.oracles.reversible import ReversibleCircuit, pack_register, unpack_register


def test_constant_equality_comparator_is_clean_and_exact_for_small_widths():
    for width in range(1, 6):
        for constant in range(1 << width):
            source = tuple(range(width))
            target = width
            work = tuple(range(width + 1, width + 1 + max(0, width - 2)))
            circuit = ReversibleCircuit(width + 1 + len(work))
            append_equality_to_constant(circuit, source, target, work, constant)
            counts = circuit.gate_counts()
            assert counts["x"] == 2 * (width - constant.bit_count())
            assert counts["cx"] == (1 if width == 1 else 0)
            assert counts["ccx"] == (0 if width == 1 else 2 * width - 3)

            for source_word in range(1 << width):
                for target_word in (0, 1):
                    state = pack_register(0, source, source_word)
                    state = pack_register(state, (target,), target_word)
                    output = circuit.apply_state(state)
                    assert unpack_register(output, source) == source_word
                    assert unpack_register(output, (target,)) == (
                        target_word ^ int(source_word == constant)
                    )
                    assert all(((output >> wire) & 1) == 0 for wire in work)
                    assert circuit.apply_inverse_state(output) == state


def test_affine_exact_observation_verifier_matches_reference_and_cleans_all_work():
    value_target = (-1) & ((1 << 6) - 1)
    oracle = ReversibleIntegerAffineEqualityOracle(
        weights=((1, 2),),
        biases=(0,),
        target_word=value_target,
        input_bits_per_feature=2,
        accumulator_bits=6,
        signed_inputs=True,
        signed_accumulator=True,
    )
    assert oracle.verify_basis_permutation()
    for input_word in range(1 << oracle.input_bits):
        expected = int(oracle.value_oracle.evaluate_input_word(input_word) == value_target)
        assert oracle.evaluate_predicate(input_word) == expected
        assert oracle.apply(input_word, 0) == (input_word, expected, 0)
        assert oracle.apply(input_word, 1) == (input_word, 1 ^ expected, 0)
        assert oracle.phase_sign(input_word) == (-1 if expected else 1)

    marked = len(oracle.marked_inputs())
    result = simulate_grover(oracle, 1)
    theta = math.asin(math.sqrt(marked / (1 << oracle.input_bits)))
    assert result.success_probability == pytest.approx(math.sin(3 * theta) ** 2)


def test_affine_equality_resource_composition_is_exact():
    oracle = ReversibleIntegerAffineEqualityOracle(
        weights=((1,),),
        biases=(0,),
        target_word=0,
        input_bits_per_feature=3,
        accumulator_bits=4,
        signed_inputs=True,
        signed_accumulator=True,
    )
    value = oracle.value_oracle.resource_estimate()
    total = oracle.resource_estimate(phase_kickback=True)
    comparator_toffoli = 2 * oracle.value_oracle.output_bits - 3
    comparator_x = 2 * oracle.value_oracle.output_bits
    assert total.toffoli_gates == 2 * value.toffoli_gates + comparator_toffoli
    assert total.cnot_gates == 2 * value.cnot_gates
    assert total.x_gates == 2 * value.x_gates + comparator_x
    assert total.peak_clean_ancillas == len(oracle.layout.work_wires)


def test_quantized_affine_equality_adapter_matches_raw_model_output():
    input_format = FixedPointFormat(2, signed=True)
    output_format = FixedPointFormat(6, signed=True)
    layer = QuantizedAffineLayer(
        weights=((1, 2),),
        biases=(0,),
        input_format=input_format,
        weight_format=FixedPointFormat(3, signed=True),
        bias_format=output_format,
        output_format=output_format,
    )
    model = QuantizedNetwork((layer,), output_mode="raw")
    target_input = model.encode_input_codes((1, -1)) if hasattr(model, "encode_input_codes") else None
    # QuantizedNetwork packs feature words little-endian; (1, -1) is 01 | 11.
    target_word = model.evaluate_input_word(0b1101)
    oracle = compile_structure_preserving_affine_equality_oracle(model, target_word)
    for input_word in range(1 << model.input_bits):
        expected = int(model.evaluate_input_word(input_word) == target_word)
        assert oracle.evaluate_predicate(input_word) == expected
        assert oracle.apply(input_word, 0) == (input_word, expected, 0)
