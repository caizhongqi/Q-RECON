from __future__ import annotations

import json

from qrecon.oracles import (
    FixedPointFormat,
    QuantizedAffineLayer,
    QuantizedNetwork,
    TruthTableOracle,
    compile_structure_preserving_mlp_threshold_oracle,
    estimate_grover_resources,
    simulate_grover,
)
from qrecon.theory import compare_search_queries, optimal_standard_grover_iterations


def main() -> None:
    input_format = FixedPointFormat(2, signed=True)
    weight_format = FixedPointFormat(3, signed=True)
    hidden_format = FixedPointFormat(4, signed=True)
    output_format = FixedPointFormat(6, signed=True)
    model = QuantizedNetwork(
        (
            QuantizedAffineLayer(
                weights=((1, -1), (-1, 1)),
                biases=(0, 0),
                input_format=input_format,
                weight_format=weight_format,
                bias_format=hidden_format,
                output_format=hidden_format,
                activation="relu",
            ),
            QuantizedAffineLayer(
                weights=((1, 1),),
                biases=(0,),
                input_format=hidden_format,
                weight_format=weight_format,
                bias_format=output_format,
                output_format=output_format,
            ),
        ),
        output_mode="binary_threshold",
        binary_threshold=3,
    )
    oracle = compile_structure_preserving_mlp_threshold_oracle(model)
    truth_table = TruthTableOracle.from_function(
        oracle.input_bits,
        1,
        oracle.evaluate_predicate,
        max_input_bits=oracle.input_bits,
        name="mlp_reference_predicate",
    )
    marked = oracle.marked_inputs()
    iterations = optimal_standard_grover_iterations(len(truth_table.table), len(marked)) or 0
    simulation = simulate_grover(oracle, iterations)
    report = {
        "architecture": {
            "input_features": 2,
            "input_bits_per_feature": 2,
            "hidden_neurons": oracle.hidden_neurons,
            "hidden_bits": oracle.hidden_bits,
            "output": "binary threshold",
        },
        "correctness": {
            "no_overflow": oracle.range_report.no_overflow,
            "clean_basis_permutation": oracle.verify_basis_permutation(),
            "marked_candidates": list(marked),
        },
        "resources": {
            "structure_preserving": oracle.resource_breakdown().to_dict(),
            "truth_table_baseline": truth_table.resource_estimate(
                phase_kickback=True
            ).to_dict(),
        },
        "search": {
            "query_comparison": compare_search_queries(
                len(truth_table.table), len(marked), target_success=0.8
            ).__dict__,
            "grover_iterations": iterations,
            "grover_success": simulation.success_probability,
            "grover_resources": estimate_grover_resources(
                oracle, iterations
            ).to_dict(),
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
