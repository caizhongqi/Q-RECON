from __future__ import annotations

import json

from qrecon.oracles import (
    FixedPointFormat,
    QuantizedAffineLayer,
    ReversibleFixedPointMLPEqualityOracle,
    solve_fixed_point_mlp_exact_output,
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
    oracle = ReversibleFixedPointMLPEqualityOracle(
        hidden,
        output,
        target,
        max_enumeration_bits=hidden.input_dimension * hidden.input_format.bits,
    )
    smt_words = tuple(sorted(oracle.encode_inputs(row) for row in smt.solutions))
    branch_words = tuple(
        sorted(oracle.encode_inputs(row) for row in branch_and_bound.solutions)
    )
    full_word_marked = oracle.marked_inputs()
    declared_domain_marked = tuple(
        word
        for word in full_word_marked
        if oracle.value.hidden.affine.raw_affine.decode_input_word(word)
        in set(smt.solutions)
    )

    print(
        json.dumps(
            {
                "task": "fractional fixed-point two-layer MLP exact-output inversion",
                "target_codes": list(target),
                "declared_candidate_count": branch_and_bound.candidate_count,
                "branch_and_bound": branch_and_bound.to_dict(),
                "z3": smt.to_dict(),
                "branch_and_bound_equals_z3": branch_words == smt_words,
                "z3_equals_oracle_on_declared_domain": (
                    smt_words == declared_domain_marked
                ),
                "full_word_oracle_marked_count": len(full_word_marked),
                "declared_domain_marked_count": len(declared_domain_marked),
                "domain_warning": (
                    "The coherent register spans the full fixed-point word space. "
                    "A quantum comparison over the smaller declared domain must "
                    "include domain state preparation or a clean membership predicate."
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
