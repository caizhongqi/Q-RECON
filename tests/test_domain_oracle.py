import pytest

from qrecon.oracles import (
    FixedPointFormat,
    QuantizedAffineLayer,
    solve_fixed_point_mlp_exact_output,
)
from qrecon.oracles.domain_oracle import (
    ReversibleDomainRestrictedMLPEqualityOracle,
    ReversibleProductDomainPredicateOracle,
)


def _model():
    input_format = FixedPointFormat(2, 0, True)
    hidden_format = FixedPointFormat(4, 0, True)
    hidden = QuantizedAffineLayer(
        weights=((1, -1), (1, 1)),
        biases=(0, 0),
        input_format=input_format,
        weight_format=FixedPointFormat(2, 0, True),
        bias_format=FixedPointFormat(4, 0, True),
        output_format=hidden_format,
        activation="relu",
    )
    output = QuantizedAffineLayer(
        weights=((1, 2), (-1, 1)),
        biases=(0, 1),
        input_format=hidden_format,
        weight_format=FixedPointFormat(3, 0, True),
        bias_format=FixedPointFormat(5, 0, True),
        output_format=FixedPointFormat(5, 0, True),
        activation="identity",
    )
    return hidden, output


def test_product_domain_membership_is_clean_for_noncontiguous_domains():
    fmt = FixedPointFormat(2, 0, True)
    domains = ((-2, 1), (-1, 0))
    oracle = ReversibleProductDomainPredicateOracle(fmt, domains)

    expected = tuple(
        sorted(
            oracle.encode_inputs(values)
            for values in ((-2, -1), (-2, 0), (1, -1), (1, 0))
        )
    )
    assert oracle.candidate_count == 4
    assert oracle.marked_inputs() == expected
    assert oracle.verify_basis_permutation()

    for word in range(1 << oracle.input_bits):
        predicate = int(word in expected)
        assert oracle.apply(word, 0) == (word, predicate, 0)
        assert oracle.apply(word, 1) == (word, 1 ^ predicate, 0)
        assert oracle.phase_sign(word) == (-1 if predicate else 1)


def test_domain_restricted_mlp_oracle_matches_complete_classical_solver():
    hidden, output = _model()
    domains = ((-2, 1), (-1, 0))
    private_record = (1, -1)
    target = output.evaluate_codes(hidden.evaluate_codes(private_record))

    classical = solve_fixed_point_mlp_exact_output(
        hidden, output, target, domains=domains
    )
    oracle = ReversibleDomainRestrictedMLPEqualityOracle(
        hidden, output, target, domains
    )
    expected_words = tuple(
        sorted(oracle.encode_inputs(solution) for solution in classical.solutions)
    )

    assert not classical.stopped_early
    assert oracle.candidate_count == classical.candidate_count == 4
    assert oracle.marked_inputs() == expected_words
    assert oracle.verify_basis_permutation()
    for word in range(1 << oracle.input_bits):
        decoded = oracle.domain.decode_input_word(word)
        in_domain = all(value in domain for value, domain in zip(decoded, domains))
        assert oracle.domain.evaluate_predicate(word) == int(in_domain)
        assert oracle.evaluate_predicate(word) == int(word in expected_words)


def test_domain_constraint_is_present_in_resource_report():
    hidden, output = _model()
    domains = ((-2, 1), (-1, 0))
    target = output.evaluate_codes(hidden.evaluate_codes((1, -1)))
    oracle = ReversibleDomainRestrictedMLPEqualityOracle(
        hidden, output, target, domains
    )
    resources = oracle.resource_estimate(phase_kickback=True)

    assert resources.input_qubits == oracle.input_bits
    assert resources.output_qubits == 1
    assert resources.peak_clean_ancillas == len(oracle.layout.work_wires)
    assert resources.logical_qubits == oracle.circuit.num_qubits
    assert resources.toffoli_gates > 0
    assert "domain-restricted" in resources.synthesis


def test_domain_oracle_rejects_empty_out_of_range_and_dimension_mismatch():
    fmt = FixedPointFormat(2, 0, True)
    with pytest.raises(ValueError):
        ReversibleProductDomainPredicateOracle(fmt, ())
    with pytest.raises(ValueError):
        ReversibleProductDomainPredicateOracle(fmt, ((0,), ()))
    with pytest.raises(OverflowError):
        ReversibleProductDomainPredicateOracle(fmt, ((2,),))

    hidden, output = _model()
    target = output.evaluate_codes(hidden.evaluate_codes((0, 0)))
    with pytest.raises(ValueError, match="dimension"):
        ReversibleDomainRestrictedMLPEqualityOracle(
            hidden, output, target, ((0,),)
        )
