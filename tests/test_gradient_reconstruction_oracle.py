import math

import pytest

from qrecon.oracles import FixedPointFormat
from qrecon.oracles.gradient_reconstruction import (
    SingleRecordGradientLeakageSpec,
    recover_single_record_from_full_gradient,
    run_single_record_gradient_reconstruction,
)


def _spec(*, gradient_bits: int = 5) -> SingleRecordGradientLeakageSpec:
    return SingleRecordGradientLeakageSpec(
        weights=(1,),
        bias=0,
        input_format=FixedPointFormat(2, signed=True),
        target_format=FixedPointFormat(3, signed=True),
        gradient_format=FixedPointFormat(gradient_bits, signed=True),
    )


def test_full_gradient_reference_channel_and_pack_round_trip_exhaustively():
    spec = _spec()
    oracle = spec.compile_value_oracle()
    assert oracle.verify_basis_permutation(exhaustive_output_words=False)
    assert spec.range_report().no_overflow
    for candidate in range(spec.population):
        inputs, target = spec.decode_candidate(candidate)
        residual = inputs[0] - target
        expected = (residual * inputs[0], residual)
        assert spec.gradient_components(candidate) == expected
        packed = spec.observe_word(candidate)
        assert spec.unpack_observation(packed) == expected
        assert oracle.apply(candidate, 0) == (candidate, packed, 0)
        assert spec.encode_candidate(inputs, target) == candidate


def test_nonzero_bias_gradient_is_analytically_and_globally_identifiable():
    spec = _spec()
    candidate = spec.encode_candidate((1,), 0)
    components = spec.gradient_components(candidate)
    assert components == (1, 1)
    assert recover_single_record_from_full_gradient(
        spec.weights, spec.bias, components[:-1], components[-1]
    ) == ((1,), 0)

    report = run_single_record_gradient_reconstruction(spec, (1,), 0)
    assert report.exact_original_identifiable
    assert report.target_fibre_size == 1
    assert report.marked_candidates == (candidate,)
    assert report.classical_queries is not None
    assert report.grover_queries is not None
    theta = math.asin(math.sqrt(1 / spec.population))
    expected_iterations = round((math.pi / (4 * theta)) - 0.5)
    assert report.grover_success_probability == pytest.approx(
        math.sin((2 * expected_iterations + 1) * theta) ** 2
    )


def test_zero_residual_creates_a_four_candidate_collision_fibre():
    spec = _spec()
    report = run_single_record_gradient_reconstruction(spec, (1,), 1)
    expected = tuple(
        sorted(
            spec.encode_candidate((value,), value)
            for value in range(
                spec.input_format.min_code, spec.input_format.max_code + 1
            )
        )
    )
    assert report.observed_word == 0
    assert report.marked_candidates == expected
    assert report.target_fibre_size == 4
    assert not report.exact_original_identifiable
    assert recover_single_record_from_full_gradient((1,), 0, (0,), 0) is None


def test_analytic_decoder_rejects_inconsistent_divisibility_and_dimensions():
    assert recover_single_record_from_full_gradient((1,), 0, (3,), 2) is None
    with pytest.raises(ValueError):
        recover_single_record_from_full_gradient((1, 2), 0, (1,), 1)


def test_gradient_range_contract_rejects_underwidth_observation_words():
    with pytest.raises(OverflowError):
        _spec(gradient_bits=4)
