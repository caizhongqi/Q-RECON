from __future__ import annotations

import json

from qrecon.oracles import (
    ReversibleBatchGradientValueOracle,
    analyze_finite_oracle,
    run_batch_gradient_reconstruction,
)


def main() -> None:
    public = ReversibleBatchGradientValueOracle(
        (1,),
        0,
        batch_size=2,
        input_bits=2,
        gradient_bits=5,
        public_targets=(0, 1),
    )
    public_report = run_batch_gradient_reconstruction(
        public, ((1,), (-2,))
    )

    private = ReversibleBatchGradientValueOracle(
        (1,),
        0,
        batch_size=2,
        input_bits=2,
        gradient_bits=5,
    )
    private_reference = private.compile_reference_oracle()
    private_candidate = private.encode_candidate(((1,), (-1,)), (0, 1))
    private_observed = private.evaluate_input_word(private_candidate)
    private_fibre = [
        candidate
        for candidate, observation in enumerate(private_reference.table)
        if observation == private_observed
    ]

    report = {
        "task": "two-record biased linear-regression aggregate-gradient reconstruction",
        "public_targets": {
            "targets": [0, 1],
            "range_certificate": public.range_report.to_dict(),
            "value_resources": public.resource_estimate().to_dict(),
            "finite_identifiability": analyze_finite_oracle(
                public.compile_reference_oracle()
            ).to_dict(),
            "reconstruction": public_report,
        },
        "private_targets": {
            "range_certificate": private.range_report.to_dict(),
            "value_resources": private.resource_estimate().to_dict(),
            "finite_identifiability": analyze_finite_oracle(
                private_reference
            ).to_dict(),
            "selected_candidate": private_candidate,
            "selected_observation": private_observed,
            "selected_fibre": private_fibre,
        },
        "interpretation": (
            "the public-target finite domain is globally injective, while private "
            "targets create nontrivial collision fibres; neither fact alone proves "
            "end-to-end quantum advantage over specialized classical solvers"
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
