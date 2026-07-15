from itertools import product

import pytest

z3 = pytest.importorskip("z3")

from qrecon.oracles.fixed_point import FixedPointFormat
from qrecon.oracles.fixed_point_deep_mlp import (
    ReversibleFixedPointDeepMLPEqualityOracle,
)
from qrecon.oracles.models import QuantizedAffineLayer, QuantizedNetwork
from qrecon.oracles.z3_deep_inversion import (
    solve_fixed_point_deep_mlp_with_z3,
)


def _deep_model():
    input_format = FixedPointFormat(2, 0, True)
    weight_format = FixedPointFormat(2, 0, True)
    bias_format = FixedPointFormat(8, 0, True)
    hidden_one_format = FixedPointFormat(4, 0, True)
    hidden_two_format = FixedPointFormat(5, 0, True)
    output_format = FixedPointFormat(6, 0, True)

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
    final = QuantizedAffineLayer(
        weights=((1, -1),),
        biases=(0,),
        input_format=hidden_two_format,
        weight_format=weight_format,
        bias_format=bias_format,
        output_format=output_format,
        activation="identity",
    )
    full_domain = tuple(
        range(input_format.min_code, input_format.max_code + 1)
    )
    return (first, second, final), (full_domain, full_domain)


def _exact_solutions(layers, domains, target):
    network = QuantizedNetwork(tuple(layers), output_mode="raw")
    return tuple(
        sorted(
            candidate
            for candidate in product(*domains)
            if network.evaluate_codes(candidate) == tuple(target)
        )
    )


def test_deep_z3_matches_reference_and_clean_equality_oracle_for_every_output():
    layers, domains = _deep_model()
    network = QuantizedNetwork(layers, output_mode="raw")
    reachable = {
        network.evaluate_codes(candidate)
        for candidate in product(*domains)
    }

    for target in sorted(reachable):
        expected = _exact_solutions(layers, domains, target)
        report = solve_fixed_point_deep_mlp_with_z3(
            layers,
            target,
            domains=domains,
        )
        oracle = ReversibleFixedPointDeepMLPEqualityOracle(
            layers,
            target,
            max_enumeration_bits=network.input_bits,
        )

        assert report.complete
        assert report.termination == "exhausted"
        assert report.reason_unknown is None
        assert report.candidate_count == 1 << network.input_bits
        assert report.solutions == expected
        assert report.solver_checks == report.solution_count + 1
        assert report.encoded_constraint_count > 0
        assert oracle.verify_basis_permutation()
        assert oracle.marked_inputs() == tuple(
            sorted(network.encode_input_codes(candidate) for candidate in expected)
        )


def test_deep_z3_solution_limit_is_explicit():
    layers, domains = _deep_model()
    network = QuantizedNetwork(layers, output_mode="raw")
    target = network.evaluate_codes((0, 0))
    expected = _exact_solutions(layers, domains, target)
    assert len(expected) > 1

    report = solve_fixed_point_deep_mlp_with_z3(
        layers,
        target,
        domains=domains,
        max_solutions=1,
    )
    assert report.termination == "solution_limit"
    assert not report.complete
    assert report.solution_count == 1
    assert report.solutions[0] in expected


def test_deep_z3_certifies_impossible_output():
    layers, domains = _deep_model()
    impossible = (layers[-1].output_format.max_code,)
    assert _exact_solutions(layers, domains, impossible) == ()

    report = solve_fixed_point_deep_mlp_with_z3(
        layers,
        impossible,
        domains=domains,
    )
    assert report.complete
    assert report.solutions == ()
    assert report.solver_checks == 1


def test_deep_z3_rejects_mismatched_network_and_solver_contracts():
    layers, domains = _deep_model()
    with pytest.raises(ValueError):
        solve_fixed_point_deep_mlp_with_z3((), (0,), domains=domains)
    with pytest.raises(ValueError):
        solve_fixed_point_deep_mlp_with_z3(layers, (), domains=domains)
    with pytest.raises(ValueError):
        solve_fixed_point_deep_mlp_with_z3(
            layers,
            (0,),
            domains=(domains[0],),
        )
    with pytest.raises(ValueError):
        solve_fixed_point_deep_mlp_with_z3(
            layers,
            (0,),
            domains=domains,
            max_solutions=0,
        )
    with pytest.raises(ValueError):
        solve_fixed_point_deep_mlp_with_z3(
            layers,
            (0,),
            domains=domains,
            timeout_ms=0,
        )

    final_relu = QuantizedAffineLayer(
        weights=layers[-1].weights,
        biases=layers[-1].biases,
        input_format=layers[-1].input_format,
        weight_format=layers[-1].weight_format,
        bias_format=layers[-1].bias_format,
        output_format=layers[-1].output_format,
        activation="relu",
    )
    with pytest.raises(ValueError):
        solve_fixed_point_deep_mlp_with_z3(
            (*layers[:-1], final_relu),
            (0,),
            domains=domains,
        )
