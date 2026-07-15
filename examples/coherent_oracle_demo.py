from __future__ import annotations

import json

from qrecon.oracles import (
    ANFOracle,
    FixedPointFormat,
    QuantizedAffineLayer,
    QuantizedNetwork,
    analyze_finite_oracle,
    compare_exact_syntheses,
    compile_model_value_oracle,
    compile_verifier_oracle,
    estimate_grover_resources,
    simulate_grover,
)
from qrecon.theory import compare_search_queries, optimal_standard_grover_iterations


def main() -> None:
    input_format = FixedPointFormat(2, signed=True)
    weight_format = FixedPointFormat(3, signed=True)
    accumulator_format = FixedPointFormat(6, signed=True)
    layer = QuantizedAffineLayer(
        weights=((1, 2),),
        biases=(0,),
        input_format=input_format,
        weight_format=weight_format,
        bias_format=accumulator_format,
        output_format=accumulator_format,
    )
    model = QuantizedNetwork((layer,), output_mode="raw")
    value_oracle = compile_model_value_oracle(model, max_input_bits=4)

    target_input = model.encode_input_codes((1, -2))
    target_output = model.evaluate_input_word(target_input)
    verifier = compile_verifier_oracle(value_oracle, target_output, metric="exact")
    anf_verifier = ANFOracle.from_truth_table(verifier)
    synthesis = compare_exact_syntheses(verifier)
    selected_verifier = anf_verifier if synthesis.selected == "anf" else verifier
    marked = len(selected_verifier.marked_inputs())
    iterations = optimal_standard_grover_iterations(len(verifier.table), marked) or 0
    simulation = simulate_grover(selected_verifier, iterations)

    report = {
        "model": {
            "input_bits": model.input_bits,
            "output_bits": model.output_bits,
            "range_proof": {
                "no_overflow": model.range_report().no_overflow,
            },
        },
        "value_oracle": {
            "truth_table_sha256": value_oracle.truth_table_sha256,
            "resources": value_oracle.resource_estimate().to_dict(),
            "finite_identifiability": analyze_finite_oracle(value_oracle).to_dict(),
        },
        "verifier": {
            "target_input": target_input,
            "target_output": target_output,
            "marked_candidates": marked,
            "selected_synthesis": synthesis.selected,
            "minterm_resources": synthesis.minterm.to_dict(),
            "anf_resources": synthesis.anf.to_dict(),
        },
        "query_comparison": compare_search_queries(
            len(verifier.table), marked, target_success=0.8
        ).__dict__,
        "grover": {
            "iterations": iterations,
            "success_probability": simulation.success_probability,
            "most_likely_inputs": simulation.most_likely_inputs,
            "resources": estimate_grover_resources(
                selected_verifier, iterations
            ).to_dict(),
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
