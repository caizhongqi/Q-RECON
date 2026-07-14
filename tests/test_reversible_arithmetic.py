import math

import pytest

from qrecon.oracles import (
    FixedPointFormat,
    ReversibleCircuit,
    ReversibleIntegerAffinePredicateOracle,
    ReversibleIntegerAffineValueOracle,
    append_cdkm_fixed_adder,
    compile_structure_preserving_affine_oracle,
    compile_structure_preserving_threshold_oracle,
    estimate_grover_resources,
    pack_register,
    quantized_binary_logistic_regression,
    simulate_grover,
    unpack_register,
)
from qrecon.oracles.models import QuantizedAffineLayer, QuantizedNetwork


def test_cdkm_fixed_adder_is_clean_and_exact_exhaustively():
    for width in range(1, 5):
        addend = tuple(range(width))
        accumulator = tuple(range(width, 2 * width))
        helper = 2 * width
        circuit = ReversibleCircuit(2 * width + 1)
        append_cdkm_fixed_adder(circuit, addend, accumulator, helper)
        counts = circuit.gate_counts()
        assert counts["ccx"] == 2 * width
        assert counts["cx"] == 4 * width

        for left in range(1 << width):
            for right in range(1 << width):
                state = pack_register(0, addend, left)
                state = pack_register(state, accumulator, right)
                output = circuit.apply_state(state)
                assert unpack_register(output, addend) == left
                assert unpack_register(output, accumulator) == (
                    left + right
                ) % (1 << width)
                assert (output >> helper) & 1 == 0
                assert circuit.apply_inverse_state(output) == state


def test_structure_preserving_affine_value_oracle_is_clean_for_signed_inputs():
    oracle = ReversibleIntegerAffineValueOracle(
        weights=((2, -1), (-1, 1)),
        biases=(1, -2),
        input_bits_per_feature=2,
        accumulator_bits=6,
        signed_inputs=True,
        signed_accumulator=True,
    )
    assert oracle.range_report.no_overflow
    assert oracle.verify_basis_permutation()

    output_mask = (1 << oracle.output_bits) - 1
    for input_word in range(1 << oracle.input_bits):
        expected = oracle.evaluate_input_word(input_word)
        for output_word in (0, 1, output_mask, 0b101010101010):
            forward = oracle.apply(input_word, output_word)
            assert forward == (input_word, output_word ^ expected, 0)
            assert oracle.inverse_apply(*forward) == (input_word, output_word, 0)


def test_affine_compiler_adapter_matches_integer_quantized_network():
    input_format = FixedPointFormat(2, signed=True)
    weight_format = FixedPointFormat(3, signed=True)
    accumulator_format = FixedPointFormat(6, signed=True)
    layer = QuantizedAffineLayer(
        weights=((2, -1),),
        biases=(1,),
        input_format=input_format,
        weight_format=weight_format,
        bias_format=accumulator_format,
        output_format=accumulator_format,
    )
    model = QuantizedNetwork((layer,), output_mode="raw")
    oracle = compile_structure_preserving_affine_oracle(model)
    for input_word in range(1 << model.input_bits):
        assert oracle.evaluate_input_word(input_word) == model.evaluate_input_word(
            input_word
        )
        assert oracle.apply(input_word, 0) == (
            input_word,
            model.evaluate_input_word(input_word),
            0,
        )


def test_structure_preserving_threshold_oracle_and_phase_are_exact():
    oracle = ReversibleIntegerAffinePredicateOracle(
        weights=(2, -1),
        bias=1,
        threshold=0,
        input_bits_per_feature=2,
        accumulator_bits=6,
        signed_inputs=True,
    )
    assert oracle.range_report.no_overflow
    assert oracle.verify_basis_permutation()
    for input_word in range(1 << oracle.input_bits):
        predicate = oracle.evaluate_predicate(input_word)
        assert oracle.apply(input_word, 0) == (input_word, predicate, 0)
        assert oracle.apply(input_word, 1) == (input_word, 1 ^ predicate, 0)
        assert oracle.phase_sign(input_word) == (-1 if predicate else 1)

    marked = len(oracle.marked_inputs())
    iterations = 1
    result = simulate_grover(oracle, iterations)
    theta = math.asin(math.sqrt(marked / (1 << oracle.input_bits)))
    assert result.success_probability == pytest.approx(
        math.sin((2 * iterations + 1) * theta) ** 2
    )
    resources = estimate_grover_resources(oracle, iterations)
    assert resources.oracle_calls == 1
    assert resources.oracle.toffoli_gates > 0


def test_threshold_adapter_matches_binary_logistic_reference():
    model = quantized_binary_logistic_regression(
        (2, -1),
        1,
        input_format=FixedPointFormat(2, signed=True),
        weight_format=FixedPointFormat(3, signed=True),
        bias_format=FixedPointFormat(6, signed=True),
        logit_format=FixedPointFormat(6, signed=True),
        threshold=0,
    )
    oracle = compile_structure_preserving_threshold_oracle(model)
    for input_word in range(1 << model.input_bits):
        expected = model.evaluate_input_word(input_word)
        assert oracle.evaluate_predicate(input_word) == expected
        assert oracle.apply(input_word, 0) == (input_word, expected, 0)


def test_exact_polynomial_resource_formula_for_single_weight_one():
    width = 4
    value = ReversibleIntegerAffineValueOracle(
        weights=((1,),),
        biases=(0,),
        input_bits_per_feature=3,
        accumulator_bits=width,
        signed_inputs=True,
        signed_accumulator=True,
    )
    value_resources = value.resource_estimate()
    assert value_resources.toffoli_gates == 4 * width
    assert value_resources.cnot_gates == 13 * width

    predicate = ReversibleIntegerAffinePredicateOracle(
        weights=(1,),
        bias=0,
        threshold=0,
        input_bits_per_feature=3,
        accumulator_bits=width,
        signed_inputs=True,
    )
    predicate_resources = predicate.resource_estimate(phase_kickback=True)
    assert predicate_resources.toffoli_gates == 4 * width
    assert predicate_resources.cnot_gates == 12 * width + 1
    assert predicate_resources.x_gates == 1


def test_structure_compiler_rejects_unproved_overflow_and_fractional_scaling():
    with pytest.raises(OverflowError):
        ReversibleIntegerAffinePredicateOracle(
            weights=(4,),
            bias=0,
            threshold=0,
            input_bits_per_feature=3,
            accumulator_bits=3,
            signed_inputs=True,
        )

    fractional = quantized_binary_logistic_regression(
        (1,),
        0,
        input_format=FixedPointFormat(3, fractional_bits=1, signed=True),
        weight_format=FixedPointFormat(3, signed=True),
        bias_format=FixedPointFormat(6, signed=True),
        logit_format=FixedPointFormat(6, signed=True),
    )
    with pytest.raises(ValueError):
        compile_structure_preserving_threshold_oracle(fractional)
