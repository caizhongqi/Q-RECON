from __future__ import annotations

import json

from qrecon.oracles import (
    FaultTolerantGateCosts,
    FixedPointFormat,
    QuantizedAffineLayer,
    ReversibleFixedPointMLPEqualityOracle,
    SpecializedClassicalSolverCosts,
    UnknownKQuantumSearchCosts,
    compare_unknown_k_search_to_specialized_classical,
)
from qrecon.theory import certify_bbht_uniform_success


def build_oracle():
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
    target = output.evaluate_codes(hidden.evaluate_codes((1, -2)))
    return ReversibleFixedPointMLPEqualityOracle(hidden, output, target)


def main() -> None:
    verifier = build_oracle()
    certificate = certify_bbht_uniform_success(
        1 << verifier.input_bits,
        target_success=0.9,
    )
    # These prices are deliberately illustrative logical-operation units.  The
    # example demonstrates sensitivity and a non-forced result, not hardware
    # advantage evidence.
    quantum = UnknownKQuantumSearchCosts(
        compilation_cost=50_000.0,
        per_round_state_loading_cost=100.0,
        per_round_readout_cost=100.0,
        measured_candidate_verification_cost=250.0,
        gates=FaultTolerantGateCosts(
            x_cost=1.0,
            cnot_cost=1.0,
            h_cost=1.0,
            t_cost=10.0,
            qubit_depth_cost=0.01,
            measurement_cost=100.0,
        ),
    )
    scenarios = {}
    for name, per_instance in (
        ("strong_classical_solver", 1_000.0),
        ("expensive_classical_solver", 10_000_000.0),
    ):
        report = compare_unknown_k_search_to_specialized_classical(
            verifier,
            certificate,
            SpecializedClassicalSolverCosts(
                setup_cost=10_000.0,
                per_instance_cost=per_instance,
            ),
            quantum,
            instances=10,
        )
        scenarios[name] = report.to_dict()

    print(
        json.dumps(
            {
                "cost_unit": "illustrative logical-operation unit",
                "hardware_claim": False,
                "unknown_k_schedule": list(certificate.schedule.windows),
                "certified_minimum_success": certificate.certified_minimum_success,
                "scenarios": scenarios,
                "interpretation": (
                    "The same quantum pipeline can lose against a strong classical "
                    "solver and win only under a much larger declared classical cost. "
                    "Measured common-unit costs and uncertainty sweeps are required "
                    "before any publication claim."
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
