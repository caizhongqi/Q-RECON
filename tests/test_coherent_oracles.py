import math

import pytest

from qrecon.oracles import (
    FixedPointFormat,
    QuantizedAffineLayer,
    QuantizedNetwork,
    TruthTableOracle,
    analyze_finite_oracle,
    compile_model_value_oracle,
    compile_verifier_oracle,
    estimate_grover_resources,
    quantized_binary_logistic_regression,
    round_half_away_from_zero,
    round_shift_right,
    simulate_grover,
)


def test_fixed_point_rounding_and_twos_complement_round_trip():
    assert round_half_away_from_zero(1.5) == 2
    assert round_half_away_from_zero(-1.5) == -2
    assert round_shift_right(3, 1) == 2
    assert round_shift_right(-3, 1) == -2

    fmt = FixedPointFormat(bits=4, fractional_bits=1, signed=True)
    assert fmt.quantize(1.25) == 3
    assert fmt.dequantize(3) == pytest.approx(1.5)
    for code in range(fmt.min_code, fmt.max_code + 1):
        assert fmt.word_to_code(fmt.code_to_word(code)) == code
        assert fmt.bits_to_code(fmt.code_to_bits(code)) == code


def test_quantized_logistic_reference_and_range_proof():
    input_format = FixedPointFormat(2, signed=True)
    weight_format = FixedPointFormat(3, signed=True)
    bias_format = FixedPointFormat(5, signed=True)
    logit_format = FixedPointFormat(6, signed=True)
    model = quantized_binary_logistic_regression(
        (1, -1),
        0,
        input_format=input_format,
        weight_format=weight_format,
        bias_format=bias_format,
        logit_format=logit_format,
    )
    assert model.range_report().no_overflow
    assert model.evaluate_codes((1, -1)) == (2,)
    assert model.evaluate_input_word(model.encode_input_codes((1, -1))) == 1
    assert model.evaluate_input_word(model.encode_input_codes((-1, 1))) == 0


def test_two_layer_quantized_mlp_has_bit_exact_semantics():
    fmt = FixedPointFormat(3, signed=True)
    wide = FixedPointFormat(7, signed=True)
    first = QuantizedAffineLayer(
        weights=((1, -1), (-1, 1)),
        biases=(0, 0),
        input_format=fmt,
        weight_format=fmt,
        bias_format=wide,
        output_format=wide,
        activation="relu",
    )
    second = QuantizedAffineLayer(
        weights=((1, -1),),
        biases=(0,),
        input_format=wide,
        weight_format=fmt,
        bias_format=wide,
        output_format=wide,
    )
    network = QuantizedNetwork((first, second), output_mode="binary_threshold")
    assert network.range_report().no_overflow
    assert network.evaluate_input_word(network.encode_input_codes((2, -1))) == 1
    assert network.evaluate_input_word(network.encode_input_codes((-1, 2))) == 0


def test_truth_table_value_oracle_is_clean_self_inverse_permutation():
    oracle = TruthTableOracle.from_function(2, 2, lambda value: (value + 1) % 4)
    assert oracle.verify_basis_permutation()
    for x in range(4):
        for y in range(4):
            forward = oracle.apply(x, y)
            assert forward == (x, y ^ ((x + 1) % 4), 0)
            assert oracle.inverse_apply(*forward) == (x, y, 0)
    assert len(oracle.truth_table_sha256) == 64


def test_compiled_model_oracle_matches_reference_exhaustively_and_is_clean():
    input_format = FixedPointFormat(2, signed=True)
    model = quantized_binary_logistic_regression(
        (1, -1),
        0,
        input_format=input_format,
        weight_format=FixedPointFormat(3, signed=True),
        bias_format=FixedPointFormat(5, signed=True),
        logit_format=FixedPointFormat(6, signed=True),
    )
    oracle = compile_model_value_oracle(model, max_input_bits=4)
    assert oracle.verify_basis_permutation()
    for word in range(1 << model.input_bits):
        assert oracle.apply(word, 0) == (word, model.evaluate_input_word(word), 0)

    resources = oracle.resource_estimate()
    assert resources.minterm_gates == sum(oracle.table)
    assert resources.toffoli_gates == resources.minterm_gates * (2 * model.input_bits - 3)
    assert resources.peak_clean_ancillas == model.input_bits - 2
    assert resources.t_count_upper_bound == 7 * resources.toffoli_gates


def test_verifier_identifiability_and_grover_match_closed_form():
    value = TruthTableOracle.from_function(3, 3, lambda x: x)
    verifier = compile_verifier_oracle(value, target_word=5, metric="exact")
    assert verifier.marked_inputs() == (5,)
    assert verifier.phase_sign(5) == -1
    assert verifier.phase_sign(4) == 1

    report = analyze_finite_oracle(value)
    assert report.injective
    assert report.uniform_exact_success == 1.0
    predicate_report = analyze_finite_oracle(verifier)
    assert not predicate_report.injective
    assert predicate_report.distinct_observations == 2

    result = simulate_grover(verifier, iterations=2)
    theta = math.asin(math.sqrt(1 / 8))
    expected = math.sin(5 * theta) ** 2
    assert result.success_probability == pytest.approx(expected)
    assert result.most_likely_inputs == (5,)

    resources = estimate_grover_resources(verifier, iterations=2)
    assert resources.oracle_calls == 2
    assert resources.total_toffoli_gates > 0
    assert resources.logical_qubits >= verifier.input_bits + 1


def test_range_proof_rejects_unsafe_compilation():
    tiny = FixedPointFormat(2, signed=True)
    layer = QuantizedAffineLayer(
        weights=((1, 1),),
        biases=(0,),
        input_format=tiny,
        weight_format=tiny,
        bias_format=tiny,
        output_format=tiny,
    )
    model = QuantizedNetwork((layer,), output_mode="raw")
    assert not model.range_report().no_overflow
    with pytest.raises(OverflowError):
        compile_model_value_oracle(model, max_input_bits=4)


def test_saturating_contract_is_explicit_and_compilable():
    tiny = FixedPointFormat(2, signed=True)
    saturated = FixedPointFormat(2, signed=True, overflow="saturate")
    layer = QuantizedAffineLayer(
        weights=((1, 1),),
        biases=(0,),
        input_format=tiny,
        weight_format=tiny,
        bias_format=tiny,
        output_format=saturated,
    )
    model = QuantizedNetwork((layer,), output_mode="raw")
    assert not model.range_report().no_overflow
    oracle = compile_model_value_oracle(
        model, max_input_bits=4, require_no_overflow=False
    )
    assert oracle.verify_basis_permutation()
