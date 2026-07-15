import pytest

from qrecon.oracles.fixed_point import FixedPointFormat
from qrecon.oracles.fixed_point_deep_mlp import (
    ReversibleFixedPointDeepMLPEqualityOracle,
    ReversibleFixedPointDeepMLPPredicateOracle,
    ReversibleFixedPointDeepMLPValueOracle,
)
from qrecon.oracles.models import QuantizedAffineLayer


def _deep_layers():
    input_format = FixedPointFormat(2, 0, True)
    hidden_one_format = FixedPointFormat(4, 0, True)
    hidden_two_format = FixedPointFormat(5, 0, True)
    output_format = FixedPointFormat(6, 0, True)
    weight_format = FixedPointFormat(2, 0, True)
    bias_format = FixedPointFormat(8, 0, True)

    first = QuantizedAffineLayer(
        weights=((1, -1), (-1, 1)),
        biases=(0, 1),
        input_format=input_format,
        weight_format=weight_format,
        bias_format=bias_format,
        output_format=hidden_one_format,
        activation="relu",
    )
    second = QuantizedAffineLayer(
        weights=((1, 1), (1, -1)),
        biases=(0, 0),
        input_format=hidden_one_format,
        weight_format=weight_format,
        bias_format=bias_format,
        output_format=hidden_two_format,
        activation="relu",
    )
    third = QuantizedAffineLayer(
        weights=((1, -1),),
        biases=(0,),
        input_format=hidden_two_format,
        weight_format=weight_format,
        bias_format=bias_format,
        output_format=output_format,
        activation="identity",
    )
    return first, second, third


def test_arbitrary_depth_fixed_point_value_oracle_is_clean_and_exact():
    layers = _deep_layers()
    oracle = ReversibleFixedPointDeepMLPValueOracle(
        layers, max_enumeration_bits=4
    )

    assert oracle.reachability_certificate.no_overflow
    assert oracle.verify_basis_permutation()
    assert len(oracle.layout.hidden_wires) == 2
    assert len(oracle.layout.shared_work) == max(
        len(layer_oracle.layout.work_wires)
        for layer_oracle in oracle.layer_oracles
    )

    for input_word in range(1 << oracle.input_bits):
        expected = oracle.network.evaluate_input_word(input_word)
        assert oracle.evaluate_input_word(input_word) == expected
        for target in (0, 1, (1 << oracle.output_bits) - 1):
            forward = oracle.apply(input_word, target)
            assert forward == (input_word, target ^ expected, 0)
            assert oracle.inverse_apply(*forward) == (input_word, target, 0)

    breakdown = oracle.resource_breakdown()
    assert breakdown.layer_multiplicity == (2, 2, 1)
    assert breakdown.hidden_register_qubits == sum(
        len(register) for register in oracle.layout.hidden_wires
    )
    assert breakdown.shared_work_qubits == len(oracle.layout.shared_work)
    assert breakdown.total.peak_clean_ancillas == len(oracle.layout.work_wires)
    assert breakdown.total.toffoli_gates == sum(
        multiplicity * item.toffoli_gates
        for multiplicity, item in zip(
            breakdown.layer_multiplicity, breakdown.layer_once
        )
    )
    assert breakdown.total.cnot_gates == sum(
        multiplicity * item.cnot_gates
        for multiplicity, item in zip(
            breakdown.layer_multiplicity, breakdown.layer_once
        )
    )


def test_arbitrary_depth_threshold_and_equality_phase_oracles():
    layers = _deep_layers()
    value = ReversibleFixedPointDeepMLPValueOracle(
        layers, max_enumeration_bits=4
    )
    threshold = ReversibleFixedPointDeepMLPPredicateOracle(
        layers,
        threshold_code=1,
        max_enumeration_bits=4,
    )
    target_word = value.evaluate_input_word(0)
    target_code = layers[-1].output_format.word_to_code(target_word)
    equality = ReversibleFixedPointDeepMLPEqualityOracle(
        layers,
        (target_code,),
        max_enumeration_bits=4,
    )

    assert threshold.verify_basis_permutation()
    assert equality.verify_basis_permutation()
    assert 0 < len(threshold.marked_inputs()) < (1 << threshold.input_bits)
    assert equality.marked_inputs()

    for input_word in range(1 << value.input_bits):
        threshold_label = int(
            layers[-1].output_format.word_to_code(
                value.evaluate_input_word(input_word)
            )
            >= 1
        )
        equality_label = int(value.evaluate_input_word(input_word) == target_word)
        assert threshold.apply(input_word, 0) == (
            input_word,
            threshold_label,
            0,
        )
        assert equality.apply(input_word, 0) == (
            input_word,
            equality_label,
            0,
        )
        assert threshold.phase_sign(input_word) == (
            -1 if threshold_label else 1
        )
        assert equality.phase_sign(input_word) == (
            -1 if equality_label else 1
        )


def test_deep_fixed_point_compiler_rejects_invalid_contracts():
    first, second, third = _deep_layers()
    final_relu = QuantizedAffineLayer(
        weights=third.weights,
        biases=third.biases,
        input_format=third.input_format,
        weight_format=third.weight_format,
        bias_format=third.bias_format,
        output_format=third.output_format,
        activation="relu",
    )
    with pytest.raises(ValueError):
        ReversibleFixedPointDeepMLPValueOracle((first, second, final_relu))

    unsafe = QuantizedAffineLayer(
        weights=((1,),),
        biases=(0,),
        input_format=FixedPointFormat(3, 0, True),
        weight_format=FixedPointFormat(2, 0, True),
        bias_format=FixedPointFormat(4, 0, True),
        output_format=FixedPointFormat(2, 0, True),
        activation="identity",
    )
    with pytest.raises(OverflowError):
        ReversibleFixedPointDeepMLPValueOracle((unsafe,))

    with pytest.raises(ValueError):
        ReversibleFixedPointDeepMLPValueOracle(())
    with pytest.raises(ValueError):
        ReversibleFixedPointDeepMLPValueOracle((third,), max_enumeration_bits=0)
