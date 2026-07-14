from __future__ import annotations

import json

from qrecon.oracles import (
    FixedPointFormat,
    SingleRecordGradientLeakageSpec,
    compile_structure_preserving_gradient_value_oracle,
    recover_single_record_from_full_gradient,
    run_single_record_gradient_reconstruction,
    run_structure_preserving_gradient_reconstruction,
)


def main() -> None:
    word = FixedPointFormat(2, signed=True)
    spec = SingleRecordGradientLeakageSpec(
        weights=(1,),
        bias=0,
        input_format=word,
        target_format=word,
        gradient_format=FixedPointFormat(4, signed=True),
    )
    compiled_value = compile_structure_preserving_gradient_value_oracle(spec)

    cases = {}
    for name, inputs, target in (
        ("unique_nonzero_residual", (1,), 0),
        ("zero_residual_collision", (1,), 1),
    ):
        finite = run_single_record_gradient_reconstruction(spec, inputs, target)
        arithmetic = run_structure_preserving_gradient_reconstruction(
            spec, inputs, target
        )
        components = spec.gradient_components(spec.encode_candidate(inputs, target))
        cases[name] = {
            "candidate": {"inputs": inputs, "target": target},
            "gradient_components": components,
            "analytic_recovery": recover_single_record_from_full_gradient(
                spec.weights,
                spec.bias,
                components[:-1],
                components[-1],
            ),
            "finite_reference": finite.to_dict(),
            "structure_preserving_search": arithmetic.to_dict(),
        }

    report = {
        "task": "single-record biased linear-regression full-gradient reconstruction",
        "range_certificate": compiled_value.range_report.to_dict(),
        "compiled_value_resources": compiled_value.resource_estimate().to_dict(),
        "cases": cases,
        "interpretation": (
            "nonzero residual is classically invertible in linear time; zero residual "
            "creates an observation collision. Grover is exercised as a coherent "
            "compiler check, not claimed as an end-to-end advantage."
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
