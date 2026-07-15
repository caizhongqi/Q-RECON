import pytest

z3 = pytest.importorskip("z3")

from qrecon.oracles import (
    FixedPointFormat,
    QuantizedAffineLayer,
    exhaustive_fixed_point_mlp_solutions,
)
from qrecon.oracles.z3_inversion import solve_fixed_point_mlp_with_z3


def _fractional_model():
    input_format = FixedPointFormat(3, 1, True)
    hidden_format = FixedPointFormat(5, 1, True)
    hidden = QuantizedAffineLayer(
        weights=((2, -1, 1), (-1, 2, -2), (1, 1, -1)),
        biases=(0, 1, -1),
        input_format=input_format,
        weight_format=FixedPointFormat(3, 1, True),
        bias_format=FixedPointFormat(8, 2, True),
        output_format=hidden_format,
        activation="relu",
    )
    output = QuantizedAffineLayer(
        weights=((2, -1, 1), (-1, 1, 2)),
        biases=(0, 1),
        input_format=hidden_format,
        weight_format=FixedPointFormat(3, 1, True),
        bias_format=FixedPointFormat(9, 2, True),
        output_format=FixedPointFormat(6, 1, True),
        activation="identity",
    )
    domains = ((-2, -1, 0, 1), (-2, -1, 0, 1), (-2, -1, 0, 1))
    return hidden, output, domains


def test_z3_matches_exhaustive_for_every_reachable_fractional_output():
    hidden, output, domains = _fractional_model()
    reachable = set()
    for x0 in domains[0]:
        for x1 in domains[1]:
            for x2 in domains[2]:
                reachable.add(
                    output.evaluate_codes(hidden.evaluate_codes((x0, x1, x2)))
                )

    for target in sorted(reachable):
        expected = exhaustive_fixed_point_mlp_solutions(
            hidden, output, target, domains=domains
        )
        report = solve_fixed_point_mlp_with_z3(
            hidden, output, target, domains=domains
        )
        assert report.complete
        assert report.termination == "exhausted"
        assert report.reason_unknown is None
        assert report.candidate_count == 64
        assert report.solutions == tuple(sorted(expected))
        assert report.solver_checks == report.solution_count + 1
        assert report.encoded_constraint_count > 0


def test_z3_encoding_matches_tie_away_rounding_relu_and_saturation():
    input_format = FixedPointFormat(3, 1, True)
    hidden_format = FixedPointFormat(2, 0, True, "saturate")
    hidden = QuantizedAffineLayer(
        weights=((3,),),
        biases=(0,),
        input_format=input_format,
        weight_format=FixedPointFormat(3, 1, True),
        bias_format=FixedPointFormat(6, 2, True),
        output_format=hidden_format,
        activation="relu",
    )
    output = QuantizedAffineLayer(
        weights=((1,),),
        biases=(0,),
        input_format=hidden_format,
        weight_format=FixedPointFormat(2, 0, True),
        bias_format=FixedPointFormat(4, 0, True),
        output_format=FixedPointFormat(3, 0, True),
        activation="identity",
    )
    domains = (tuple(range(input_format.min_code, input_format.max_code + 1)),)
    target = (1,)

    expected = exhaustive_fixed_point_mlp_solutions(
        hidden, output, target, domains=domains
    )
    report = solve_fixed_point_mlp_with_z3(
        hidden, output, target, domains=domains
    )
    assert report.complete
    assert report.solutions == tuple(sorted(expected))
    assert expected
    assert (3,) in expected  # 2.25 rounds to 2, then saturates to code 1.


def test_solution_limit_is_explicit_and_not_misreported_as_complete():
    hidden, output, domains = _fractional_model()
    target = output.evaluate_codes(hidden.evaluate_codes((0, 0, 0)))
    full = exhaustive_fixed_point_mlp_solutions(
        hidden, output, target, domains=domains
    )
    assert full

    report = solve_fixed_point_mlp_with_z3(
        hidden,
        output,
        target,
        domains=domains,
        max_solutions=1,
    )
    assert report.termination == "solution_limit"
    assert not report.complete
    assert report.solution_count == 1
    assert report.solutions[0] in full


def test_impossible_target_is_certified_unsatisfiable():
    hidden, output, domains = _fractional_model()
    target = (output.output_format.max_code, output.output_format.min_code)
    assert exhaustive_fixed_point_mlp_solutions(
        hidden, output, target, domains=domains
    ) == ()

    report = solve_fixed_point_mlp_with_z3(
        hidden, output, target, domains=domains
    )
    assert report.complete
    assert report.solutions == ()
    assert report.solver_checks == 1


def test_z3_solver_contract_validates_domains_targets_and_limits():
    hidden, output, domains = _fractional_model()
    with pytest.raises(ValueError):
        solve_fixed_point_mlp_with_z3(hidden, output, (0,), domains=domains)
    with pytest.raises(ValueError):
        solve_fixed_point_mlp_with_z3(
            hidden, output, (0, 0), domains=(domains[0], domains[1])
        )
    with pytest.raises(ValueError):
        solve_fixed_point_mlp_with_z3(
            hidden, output, (0, 0), domains=(domains[0], (), domains[2])
        )
    with pytest.raises(ValueError):
        solve_fixed_point_mlp_with_z3(
            hidden, output, (0, 0), domains=domains, max_solutions=0
        )
    with pytest.raises(ValueError):
        solve_fixed_point_mlp_with_z3(
            hidden, output, (0, 0), domains=domains, timeout_ms=0
        )
