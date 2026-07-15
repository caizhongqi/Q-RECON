import pytest

from qrecon.oracles import (
    FixedPointFormat,
    QuantizedAffineLayer,
    ReversibleFixedPointAffineValueOracle,
    ReversibleFixedPointRequantizationOracle,
    append_controlled_increment,
)
from qrecon.oracles.reversible import ReversibleCircuit, pack_register, unpack_register


@pytest.mark.parametrize(
    "source,target",
    [
        (FixedPointFormat(4, 2, True), FixedPointFormat(4, 1, True)),
        (FixedPointFormat(5, 3, True), FixedPointFormat(4, 0, True)),
        (FixedPointFormat(4, 2, False), FixedPointFormat(4, 0, False)),
        (FixedPointFormat(3, 1, True), FixedPointFormat(5, 1, True)),
    ],
)
def test_requantization_matches_reference_exhaustively(source, target):
    oracle = ReversibleFixedPointRequantizationOracle(source, target)
    assert oracle.range_report.no_overflow
    assert oracle.verify_basis_permutation()
    for word in range(1 << source.bits):
        expected_code = target.requantize(
            source.word_to_code(word), source.fractional_bits
        )
        assert oracle.decode_output_word(oracle.evaluate_input_word(word)) == expected_code
        assert oracle.apply(word, 0) == (
            word,
            target.code_to_word(expected_code),
            0,
        )


def test_half_ties_round_away_from_zero():
    source = FixedPointFormat(5, 2, True)
    target = FixedPointFormat(5, 1, True)
    oracle = ReversibleFixedPointRequantizationOracle(source, target)
    assert oracle.decode_output_word(oracle.evaluate_input_word(source.code_to_word(1))) == 1
    assert oracle.decode_output_word(oracle.evaluate_input_word(source.code_to_word(-1))) == -1
    assert oracle.decode_output_word(oracle.evaluate_input_word(source.code_to_word(3))) == 2
    assert oracle.decode_output_word(oracle.evaluate_input_word(source.code_to_word(-3))) == -2


def test_controlled_increment_is_exact_and_cleans_work():
    width = 5
    control = 0
    register = tuple(range(1, 1 + width))
    work = tuple(range(1 + width, 1 + width + width - 1))
    circuit = ReversibleCircuit(1 + width + width - 1)
    append_controlled_increment(circuit, control, register, work)
    for enabled in (0, 1):
        for value in range(1 << width):
            state = pack_register(0, (control,), enabled)
            state = pack_register(state, register, value)
            result = circuit.apply_state(state)
            assert unpack_register(result, (control,)) == enabled
            assert unpack_register(result, register) == (
                value + enabled
            ) % (1 << width)
            assert unpack_register(result, work) == 0
            assert circuit.apply_inverse_state(result) == state


def test_requantization_rejects_unsafe_target_range():
    with pytest.raises(OverflowError):
        ReversibleFixedPointRequantizationOracle(
            FixedPointFormat(6, 0, True),
            FixedPointFormat(3, 0, True),
        )
    with pytest.raises(ValueError):
        ReversibleFixedPointRequantizationOracle(
            FixedPointFormat(4, 1, True),
            FixedPointFormat(4, 2, True),
        )


def test_fixed_point_affine_oracle_matches_layer_reference():
    input_format = FixedPointFormat(3, 1, True)
    weight_format = FixedPointFormat(3, 1, True)
    bias_format = FixedPointFormat(6, 2, True)
    output_format = FixedPointFormat(5, 1, True)
    layer = QuantizedAffineLayer(
        weights=((1, -2), (2, 1)),
        biases=(1, -1),
        input_format=input_format,
        weight_format=weight_format,
        bias_format=bias_format,
        output_format=output_format,
        activation="identity",
    )
    oracle = ReversibleFixedPointAffineValueOracle(layer, max_enumeration_bits=6)
    assert oracle.verify_basis_permutation()
    for word in range(1 << oracle.input_bits):
        assert oracle.apply(word, 0) == (word, oracle.evaluate_input_word(word), 0)
    resources = oracle.resource_estimate()
    assert resources.toffoli_gates > 0
    assert resources.peak_clean_ancillas == len(oracle.layout.work_wires)


def test_fixed_point_affine_rejects_relu_until_composed():
    fmt = FixedPointFormat(3, 1, True)
    layer = QuantizedAffineLayer(
        weights=((1,),),
        biases=(0,),
        input_format=fmt,
        weight_format=fmt,
        bias_format=FixedPointFormat(6, 2, True),
        output_format=fmt,
        activation="relu",
    )
    with pytest.raises(ValueError):
        ReversibleFixedPointAffineValueOracle(layer)
