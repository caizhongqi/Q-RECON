import math

import pytest

from qrecon.oracles import (
    FixedPointFormat,
    QuantizedAffineLayer,
    QuantizedNetwork,
    compile_structure_preserving_deep_mlp_threshold_oracle,
    estimate_grover_resources,
    simulate_grover,
)


def _deep_model(*, fractional_bits: int = 0) -> QuantizedNetwork:
    input_format = FixedPointFormat(2, fractional_bits=fractional_bits, signed=True)
    weight_format = FixedPointFormat(3, signed=True)
    hidden_one = FixedPointFormat(4, signed=True)
    hidden_two = FixedPointFormat(4, signed=True)
    output_format = FixedPointFormat(6, signed=True)
    return QuantizedNetwork(
        (
            QuantizedAffineLayer(
                weights=((1, -1), (-1, 1)),
                biases=(0, 0),
                input_format=input_format,
                weight_format=weight_format,
                bias_format=hidden_one,
                output_format=hidden_one,
                activation="relu",
            ),
            QuantizedAffineLayer(
                weights=((1, 0), (0, 1)),
                biases=(0, 0),
                input_format=hidden_one,
                weight_format=weight_format,
                bias_format=hidden_two,
                output_format=hidden_two,
                activation="relu",
            ),
            QuantizedAffineLayer(
                weights=((1, 1),),
                biases=(0,),
                input_format=hidden_two,
                weight_format=weight_format,
                bias_format=output_format,
                output_format=output_format,
                activation="identity",
            ),
        ),
        output_mode="binary_threshold",
        binary_threshold=3,
    )


def test_arbitrary_depth_mlp_matches_reference_and_cleans_every_layer_exhaustively():
    model = _deep_model()
    oracle = compile_structure_preserving_deep_mlp_threshold_oracle(model)
    assert oracle.range_report.no_overflow
    assert oracle.verify_basis_permutation()
    for input_word in range(1 << model.input_bits):
        values = model.decode_input_word(input_word)
        expected = int(abs(values[0] - values[1]) >= 3)
        assert model.evaluate_input_word(input_word) == expected
        assert oracle.evaluate_predicate(input_word) == expected
        assert oracle.apply(input_word, 0) == (input_word, expected, 0)
        assert oracle.apply(input_word, 1) == (input_word, 1 ^ expected, 0)
        assert oracle.phase_sign(input_word) == (-1 if expected else 1)
        activations = oracle.hidden_activations(input_word)
        assert activations[0] == (
            max(0, values[0] - values[1]),
            max(0, values[1] - values[0]),
        )
        assert activations[1] == activations[0]


def test_deep_mlp_reuses_one_shared_arithmetic_work_region():
    oracle = compile_structure_preserving_deep_mlp_threshold_oracle(_deep_model())
    component_work = [
        len(component.layout.work_wires) for component in oracle.hidden_oracles
    ] + [len(oracle.final_oracle.layout.work_wires)]
    assert len(oracle.layout.shared_arithmetic_work) == max(component_work)
    assert len(oracle.layout.shared_arithmetic_work) < sum(component_work)

    breakdown = oracle.resource_breakdown()
    hidden = breakdown.hidden_affine_once
    final = breakdown.final_predicate_once
    total = breakdown.total
    assert breakdown.relu_x_gates_compute_uncompute == 16
    assert breakdown.relu_toffoli_gates_compute_uncompute == 24
    assert total.x_gates == 2 * sum(item.x_gates for item in hidden) + final.x_gates + 16
    assert total.cnot_gates == 2 * sum(item.cnot_gates for item in hidden) + final.cnot_gates
    assert total.toffoli_gates == (
        2 * sum(item.toffoli_gates for item in hidden)
        + final.toffoli_gates
        + 24
    )
    assert total.logical_qubits == oracle.input_bits + 1 + len(oracle.layout.work_wires)


def test_deep_mlp_phase_netlist_matches_standard_grover_success():
    oracle = compile_structure_preserving_deep_mlp_threshold_oracle(_deep_model())
    marked = oracle.marked_inputs()
    assert len(marked) == 2
    result = simulate_grover(oracle, 2)
    theta = math.asin(math.sqrt(2 / (1 << oracle.input_bits)))
    assert result.success_probability == pytest.approx(math.sin(5 * theta) ** 2)
    assert set(result.most_likely_inputs) == set(marked)
    resources = estimate_grover_resources(oracle, 2)
    assert resources.oracle_calls == 2
    assert resources.logical_qubits >= oracle.resource_estimate().logical_qubits


def test_deep_mlp_adapter_rejects_fractional_requantization():
    with pytest.raises(ValueError):
        compile_structure_preserving_deep_mlp_threshold_oracle(
            _deep_model(fractional_bits=1)
        )
