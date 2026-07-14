import pytest

from qrecon.oracles import (
    FixedPointFormat,
    QuantizedAffineLayer,
    exhaustive_fixed_point_mlp_solutions,
    fixed_point_mlp_output_bounds,
    solve_fixed_point_mlp_exact_output,
)


def _model():
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


def test_branch_and_bound_matches_exhaustive_for_every_reachable_output():
    hidden, output, domains = _model()
    reachable = {}
    for x0 in domains[0]:
        for x1 in domains[1]:
            for x2 in domains[2]:
                record = (x0, x1, x2)
                target = output.evaluate_codes(hidden.evaluate_codes(record))
                reachable.setdefault(target, []).append(record)

    for target, expected in reachable.items():
        report = solve_fixed_point_mlp_exact_output(
            hidden,
            output,
            target,
            domains=domains,
        )
        exhaustive = exhaustive_fixed_point_mlp_solutions(
            hidden,
            output,
            target,
            domains=domains,
        )
        assert report.solutions == exhaustive == tuple(expected)
        assert report.candidate_count == 64
        assert report.bound_evaluations == report.nodes_visited
        assert 0 <= report.leaf_evaluations <= report.candidate_count
        assert 0.0 <= report.exhaustive_leaf_reduction <= 1.0


def test_output_bounds_contain_all_exact_completions():
    hidden, output, domains = _model()
    partial = (1, None, -1)
    bounds = fixed_point_mlp_output_bounds(
        hidden,
        output,
        domains,
        partial,
    )
    for middle in domains[1]:
        observed = output.evaluate_codes(hidden.evaluate_codes((1, middle, -1)))
        assert all(interval.contains(value) for interval, value in zip(bounds, observed))


def test_impossible_target_prunes_without_exhaustive_leaves():
    hidden, output, domains = _model()
    target = (output.output_format.max_code, output.output_format.min_code)
    assert exhaustive_fixed_point_mlp_solutions(
        hidden,
        output,
        target,
        domains=domains,
    ) == ()
    report = solve_fixed_point_mlp_exact_output(
        hidden,
        output,
        target,
        domains=domains,
    )
    assert report.solutions == ()
    assert report.pruned_nodes > 0
    assert report.leaf_evaluations < report.candidate_count


def test_solution_limit_is_explicit_and_deterministic():
    hidden, output, domains = _model()
    target = output.evaluate_codes(hidden.evaluate_codes((0, 0, 0)))
    full = exhaustive_fixed_point_mlp_solutions(
        hidden,
        output,
        target,
        domains=domains,
    )
    assert full
    report = solve_fixed_point_mlp_exact_output(
        hidden,
        output,
        target,
        domains=domains,
        max_solutions=1,
    )
    assert report.stopped_early
    assert len(report.solutions) == 1
    assert report.solutions[0] in full


def test_domains_targets_and_layer_contracts_are_validated():
    hidden, output, domains = _model()
    with pytest.raises(ValueError):
        solve_fixed_point_mlp_exact_output(hidden, output, (0,), domains=domains)
    with pytest.raises(ValueError):
        solve_fixed_point_mlp_exact_output(
            hidden,
            output,
            (0, 0),
            domains=(domains[0], domains[1]),
        )
    with pytest.raises(ValueError):
        solve_fixed_point_mlp_exact_output(
            hidden,
            output,
            (0, 0),
            domains=(domains[0], (), domains[2]),
        )
    with pytest.raises(ValueError):
        solve_fixed_point_mlp_exact_output(
            hidden,
            output,
            (0, 0),
            domains=domains,
            max_solutions=0,
        )

    bad_output = QuantizedAffineLayer(
        weights=output.weights,
        biases=output.biases,
        input_format=output.input_format,
        weight_format=output.weight_format,
        bias_format=output.bias_format,
        output_format=output.output_format,
        activation="relu",
    )
    with pytest.raises(ValueError):
        solve_fixed_point_mlp_exact_output(
            hidden,
            bad_output,
            (0, 0),
            domains=domains,
        )
