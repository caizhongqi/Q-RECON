from __future__ import annotations

import json

from qrecon.oracles import (
    FixedPointFormat,
    QuantizedAffineLayer,
    ReversibleFixedPointMLPEqualityOracle,
    estimate_grover_resources,
    simulate_grover,
    solve_fixed_point_mlp_exact_output,
)
from qrecon.theory import optimal_standard_grover_iterations


def build_model():
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
    domain = tuple(range(input_format.min_code, input_format.max_code + 1))
    return hidden, output, (domain, domain)


def main() -> None:
    hidden, output, domains = build_model()
    private_record = (1, -2)
    target = output.evaluate_codes(hidden.evaluate_codes(private_record))

    classical = solve_fixed_point_mlp_exact_output(
        hidden,
        output,
        target,
        domains=domains,
    )
    verifier = ReversibleFixedPointMLPEqualityOracle(hidden, output, target)
    marked = verifier.marked_inputs()
    iterations = optimal_standard_grover_iterations(
        1 << verifier.input_bits,
        len(marked),
    )
    if iterations is None:
        raise RuntimeError("the exact-output verifier has no marked candidate")
    simulation = simulate_grover(verifier, iterations)
    oracle_resources = verifier.resource_estimate(phase_kickback=True)
    search_resources = estimate_grover_resources(verifier, iterations)

    payload = {
        "task": "two-layer fixed-point MLP exact-output inversion",
        "private_record_for_regression_only": list(private_record),
        "target_codes": list(target),
        "candidate_population": 1 << verifier.input_bits,
        "marked_count": len(marked),
        "marked_input_words": list(marked),
        "classical_branch_and_bound": classical.to_dict(),
        "classical_quantum_solution_sets_match": tuple(
            sorted(verifier.encode_inputs(row) for row in classical.solutions)
        )
        == marked,
        "oracle_basis_permutation_verified": verifier.verify_basis_permutation(),
        "oracle_resources": oracle_resources.to_dict(),
        "grover": {
            "iterations": iterations,
            "success_probability": simulation.success_probability,
            "most_likely_inputs": list(simulation.most_likely_inputs),
            "resources": search_resources.to_dict(),
        },
        "claim_boundary": (
            "This report matches exact semantics, solution sets, and logical "
            "resources. It does not assign a common wall-clock or monetary cost "
            "unit and therefore does not claim end-to-end quantum advantage."
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
