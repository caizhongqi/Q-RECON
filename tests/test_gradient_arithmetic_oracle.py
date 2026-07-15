import math

import pytest

from qrecon.oracles import FixedPointFormat, ReversibleCircuit
from qrecon.oracles.gradient_arithmetic import (
    ReversibleSingleRecordGradientEqualityOracle,
    ReversibleSingleRecordGradientValueOracle,
    append_signed_modular_product,
    run_structure_preserving_gradient_reconstruction,
)
from qrecon.oracles.gradient_reconstruction import SingleRecordGradientLeakageSpec


def _decode_signed(word: int, bits: int) -> int:
    return word - (1 << bits) if word >= (1 << (bits - 1)) else word


def _spec() -> SingleRecordGradientLeakageSpec:
    word = FixedPointFormat(2, signed=True)
    return SingleRecordGradientLeakageSpec(
        weights=(1,),
        bias=0,
        input_format=word,
        target_format=word,
        gradient_format=FixedPointFormat(4, signed=True),
    )


def test_signed_variable_multiplier_is_reversible_for_equal_and_mixed_widths():
    for left_width, product_width in ((4, 4), (2, 4)):
        left = tuple(range(left_width))
        right = tuple(range(left_width, left_width + product_width))
        output = tuple(
            range(left_width + product_width, left_width + 2 * product_width)
        )
        scratch = tuple(
            range(left_width + 2 * product_width, left_width + 3 * product_width)
        )
        helper = left_width + 3 * product_width
        circuit = ReversibleCircuit(helper + 1)
        append_signed_modular_product(
            circuit, left, right, output, scratch, helper
        )
        mask = (1 << product_width) - 1
        for left_word in range(1 << left_width):
            for right_word in range(1 << product_width):
                state = left_word | (right_word << left_width)
                forward = circuit.apply_state(state)
                expected = (
                    _decode_signed(left_word, left_width)
                    * _decode_signed(right_word, product_width)
                ) & mask
                assert (
                    forward >> (left_width + product_width)
                ) & mask == expected
                assert forward >> (left_width + 2 * product_width) == 0
                assert circuit.apply_inverse_state(forward) == state


def test_structure_preserving_gradient_value_oracle_matches_reference_exhaustively():
    spec = _spec()
    reference = spec.compile_value_oracle()
    value = ReversibleSingleRecordGradientValueOracle(
        spec.weights,
        spec.bias,
        input_bits=spec.input_format.bits,
        gradient_bits=spec.gradient_format.bits,
    )
    assert value.range_report.no_overflow
    assert value.verify_basis_permutation()
    for candidate in range(spec.population):
        assert value.decode_candidate(candidate) == spec.decode_candidate(candidate)
        assert value.gradient_components(candidate) == spec.gradient_components(candidate)
        assert value.evaluate_input_word(candidate) == reference.table[candidate]
        assert value.apply(candidate, 0) == (
            candidate,
            reference.table[candidate],
            0,
        )


def test_exact_gradient_equality_oracle_has_unique_and_collision_regimes():
    spec = _spec()
    value = ReversibleSingleRecordGradientValueOracle(
        spec.weights,
        spec.bias,
        input_bits=spec.input_format.bits,
        gradient_bits=spec.gradient_format.bits,
    )

    unique_candidate = value.encode_candidate((1,), 0)
    unique = ReversibleSingleRecordGradientEqualityOracle(
        value, value.evaluate_input_word(unique_candidate)
    )
    assert unique.verify_basis_permutation()
    assert unique.marked_inputs() == (unique_candidate,)
    assert unique.phase_sign(unique_candidate) == -1

    collision_candidate = value.encode_candidate((1,), 1)
    collision = ReversibleSingleRecordGradientEqualityOracle(
        value, value.evaluate_input_word(collision_candidate)
    )
    expected = tuple(
        sorted(value.encode_candidate((item,), item) for item in (-2, -1, 0, 1))
    )
    assert collision.marked_inputs() == expected
    assert all(collision.phase_sign(candidate) == -1 for candidate in expected)


def test_structure_preserving_gradient_phase_oracle_drives_grover_curve():
    spec = _spec()
    report = run_structure_preserving_gradient_reconstruction(spec, (1,), 0)
    assert report.exact_original_identifiable
    assert report.target_fibre_size == 1
    theta = math.asin(math.sqrt(1 / spec.population))
    expected_iterations = round((math.pi / (4 * theta)) - 0.5)
    assert report.grover_success_probability == pytest.approx(
        math.sin((2 * expected_iterations + 1) * theta) ** 2
    )
    assert report.grover_resources.oracle.toffoli_gates > 0


def test_structure_gradient_compiler_rejects_underwidth_products():
    with pytest.raises(OverflowError):
        ReversibleSingleRecordGradientValueOracle(
            (1,), 0, input_bits=3, gradient_bits=3
        )
