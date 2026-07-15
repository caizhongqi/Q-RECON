from __future__ import annotations

import json

from qrecon.oracles import (
    FixedPointFormat,
    QuantizedAffineLayer,
    solve_fixed_point_mlp_exact_output,
)
from qrecon.oracles.domain_oracle import (
    ReversibleDomainRestrictedMLPEqualityOracle,
)
from qrecon.oracles.z3_inversion import solve_fixed_point_mlp_with_z3


def build_model():
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


def main() -> None:
    hidden, output, domains = build_model()
    private_record = (1, -2, 0)
    target = output.evaluate_codes(hidden.evaluate_codes(private_record))

    branch_and_bound = solve_fixed_point_mlp_exact_output(
        hidden, output, target, domains=domains
    )
    smt = solve_fixed_point_mlp_with_z3(
        hidden, output, target, domains=domains
    )
    oracle = ReversibleDomainRestrictedMLPEqualityOracle(
        hidden,
        output,
        target,
        domains,
        max_enumeration_bits=hidden.input_dimension * hidden.input_format.bits,
    )
    smt_words = tuple(sorted(oracle.encode_inputs(row) for row in smt.solutions))
    branch_words = tuple(
        sorted(oracle.encode_inputs(row) for row in branch_and_bound.solutions)
    )
    oracle_words = oracle.marked_inputs()

    print(
        json.dumps(
            {
                "task": "fractional fixed-point two-layer MLP exact-output inversion",
                "target_codes": list(target),
                "declared_candidate_count": branch_and_bound.candidate_count,
                "full_word_population": 1 << oracle.input_bits,
                "branch_and_bound": branch_and_bound.to_dict(),
                "z3": smt.to_dict(),
                "branch_and_bound_equals_z3": branch_words == smt_words,
                "z3_equals_domain_restricted_oracle": smt_words == oracle_words,
                "domain_restricted_oracle_marked_count": len(oracle_words),
                "domain_membership_candidate_count": oracle.candidate_count,
                "domain_membership_resources": (
                    oracle.domain.resource_estimate(phase_kickback=True).to_dict()
                ),
                "combined_oracle_resources": (
                    oracle.resource_estimate(phase_kickback=True).to_dict()
                ),
                "domain_contract": (
                    "Invalid full-word states are rejected coherently by an explicit "
                    "clean product-domain predicate. Uniform state preparation over "
                    "only the structured domain remains a separately priced option."
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
