from __future__ import annotations

import json

from qrecon.oracles import (
    ClassicalSearchCosts,
    FaultTolerantGateCosts,
    QuantumSearchCosts,
    ReversibleIntegerAffinePredicateOracle,
    compare_end_to_end_search_costs,
    maximum_t_cost_for_fixed_plan,
    minimum_instances_for_fixed_plan_advantage,
)


def main() -> None:
    # Signed four-bit x ranges from -8 to 7. The predicate x >= 7 has K=1.
    verifier = ReversibleIntegerAffinePredicateOracle(
        weights=(1,),
        bias=0,
        threshold=7,
        input_bits_per_feature=4,
        accumulator_bits=6,
        signed_inputs=True,
    )
    classical = ClassicalSearchCosts(
        candidate_preparation_cost=0.25,
        verifier_evaluation_cost=1.0,
        readout_cost=0.25,
    )

    scenarios = {
        "unit_t": QuantumSearchCosts(
            compilation_cost=0.0,
            per_run_state_loading_cost=0.25,
            per_run_readout_cost=0.25,
            gates=FaultTolerantGateCosts(
                x_cost=0.01,
                cnot_cost=0.02,
                h_cost=0.01,
                t_cost=1.0,
                measurement_cost=0.25,
            ),
        ),
        "zero_t_sensitivity": QuantumSearchCosts(
            compilation_cost=100.0,
            per_run_state_loading_cost=0.25,
            per_run_readout_cost=0.25,
            gates=FaultTolerantGateCosts(
                x_cost=0.0,
                cnot_cost=0.0,
                h_cost=0.0,
                t_cost=0.0,
                measurement_cost=0.25,
            ),
        ),
    }

    reports: dict[str, object] = {}
    for name, quantum in scenarios.items():
        report = compare_end_to_end_search_costs(
            verifier,
            0.8,
            classical,
            quantum,
            instances=1,
        )
        reports[name] = {
            "report": report.to_dict(),
            "open_t_cost_threshold_for_selected_plan": maximum_t_cost_for_fixed_plan(
                report, quantum
            ),
            "minimum_instances_for_selected_plan_advantage": (
                minimum_instances_for_fixed_plan_advantage(
                    report, classical, quantum
                )
            ),
        }

    output = {
        "claim_boundary": (
            "All numbers are sensitivity scenarios in one abstract cost unit; "
            "they are not hardware performance claims."
        ),
        "predicate_resources": verifier.resource_estimate(
            phase_kickback=True
        ).to_dict(),
        "marked_candidates": len(verifier.marked_inputs()),
        "reports": reports,
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
