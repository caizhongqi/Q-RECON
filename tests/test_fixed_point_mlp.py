import math

import pytest

from qrecon.oracles import (
    FixedPointFormat,
    QuantizedAffineLayer,
    ReversibleFixedPointAffineReLUValueOracle,
    ReversibleFixedPointMLPPredicateOracle,
    ReversibleFixedPointMLPValueOracle,
    simulate_grover,
)


def _layers():
    input_format = FixedPointFormat(2, 1, True)
    hidden_format = FixedPointFormat(4, 1, True)
    first = QuantizedAffineLayer(
        weights=((1, -1), (-1, 1)),
        biases=(0, 0),
        input_format=input_format,
        weight_format=FixedPointFormat(2, 1, True),
        bias_format=FixedPointFormat(6, 2, True),
        output_format=hidden_format,
        activation="relu",
    )
    second = QuantizedAffineLayer(
        weights=((1, -1),),
        biases=(0,),
        input_format=hidden_format,
        weight_format=FixedPointFormat(3, 1, True),
        bias_format=FixedPointFormat(8, 2, True),
        output_format=FixedPointFormat(5, 1, True),
        activation="identity",
    )
    return first, second


def test_fixed_point_affine_relu_matches_reference_exhaustively():
    hidden, _ = _layers()
    oracle = ReversibleFixedPointAffineReLUValueOracle(
        hidden,
        max_enumeration_bits=4,
    )
    assert oracle.verify_basis_permutation()
    for input_word in range(1 << oracle.input_bits):
        expected = oracle.evaluate_input_word(input_word)
        assert oracle.apply(input_word, 0) == (input_word, expected, 0)
        codes = tuple(
            hidden.output_format.word_to_code(
                (expected >> (index * hidden.output_format.bits))
                & hidden.output_format.mask
            )
            for index in range(hidden.output_dimension)
        )
        assert all(code >= 0 for code in codes)
    resources = oracle.resource_estimate()
    assert resources.toffoli_gates > 0
    assert resources.peak_clean_ancillas == len(oracle.layout.work_wires)


def test_fixed_point_two_layer_value_oracle_is_clean_and_exact():
    hidden, output = _layers()
    oracle = ReversibleFixedPointMLPValueOracle(
        hidden,
        output,
        max_enumeration_bits=4,
    )
    assert oracle.verify_basis_permutation()
    for input_word in range(1 << oracle.input_bits):
        expected = oracle.evaluate_input_word(input_word)
        for target in (0, 1, (1 << oracle.output_bits) - 1):
            forward = oracle.apply(input_word, target)
            assert forward == (input_word, target ^ expected, 0)
            assert oracle.inverse_apply(*forward) == (input_word, target, 0)
    assert oracle.resource_estimate().toffoli_gates > 0


def test_fixed_point_mlp_threshold_phase_oracle_and_grover_curve():
    hidden, output = _layers()
    oracle = ReversibleFixedPointMLPPredicateOracle(
        hidden,
        output,
        threshold_code=1,
        max_enumeration_bits=4,
    )
    assert oracle.verify_basis_permutation()
    marked = oracle.marked_inputs()
    assert 0 < len(marked) < (1 << oracle.input_bits)
    for input_word in range(1 << oracle.input_bits):
        label = oracle.evaluate_label(input_word)
        assert oracle.apply(input_word, 0) == (input_word, label, 0)
        assert oracle.phase_sign(input_word) == (-1 if label else 1)

    iterations = 1
    result = simulate_grover(oracle, iterations)
    theta = math.asin(math.sqrt(len(marked) / (1 << oracle.input_bits)))
    assert result.success_probability == pytest.approx(
        math.sin((2 * iterations + 1) * theta) ** 2
    )
    assert oracle.resource_estimate(phase_kickback=True).toffoli_gates > 0


def test_fixed_point_relu_and_mlp_reject_incompatible_layers():
    hidden, output = _layers()
    with pytest.raises(ValueError):
        ReversibleFixedPointAffineReLUValueOracle(
            QuantizedAffineLayer(
                weights=hidden.weights,
                biases=hidden.biases,
                input_format=hidden.input_format,
                weight_format=hidden.weight_format,
                bias_format=hidden.bias_format,
                output_format=hidden.output_format,
                activation="identity",
            )
        )

    bad_output = QuantizedAffineLayer(
        weights=((1,),),
        biases=(0,),
        input_format=FixedPointFormat(4, 0, True),
        weight_format=output.weight_format,
        bias_format=output.bias_format,
        output_format=output.output_format,
        activation="identity",
    )
    with pytest.raises(ValueError):
        ReversibleFixedPointMLPValueOracle(hidden, bad_output)


def test_fixed_point_mlp_threshold_validates_target_code():
    hidden, output = _layers()
    with pytest.raises(OverflowError):
        ReversibleFixedPointMLPPredicateOracle(
            hidden,
            output,
            threshold_code=1 << output.output_format.bits,
        )
