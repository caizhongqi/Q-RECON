from __future__ import annotations

import json

from qrecon.oracles import (
    ReversibleBatchGradientValueOracle,
    analyze_finite_oracle,
    balanced_mitm_partial_state_count,
    ideal_unstructured_search_scale,
    run_batch_gradient_reconstruction,
    solve_batch_gradient_meet_in_the_middle,
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
    public_inputs = ((1,), (-2,))
    public_candidate = public.encode_candidate(public_inputs)
    public_observed = public.evaluate_input_word(public_candidate)
    public_report = run_batch_gradient_reconstruction(public, public_inputs)
    public_mitm = solve_batch_gradient_meet_in_the_middle(
        public, public_observed
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
    private_mitm = solve_batch_gradient_meet_in_the_middle(
        private, private_observed
    )

    report = {
        "task": "two-record biased linear-regression aggregate-gradient reconstruction",
        "public_targets": {
            "targets": [0, 1],
            "range_certificate": public.range_report.to_dict(),
            "value_resources": public.resource_estimate().to_dict(),
            "finite_identifiability": analyze_finite_oracle(
                public.compile_reference_oracle()
            ).to_dict(),
            "quantum_search": public_report,
            "meet_in_the_middle": public_mitm.to_dict(),
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
            "meet_in_the_middle": private_mitm.to_dict(),
        },
        "scaling_warning": {
            "local_domain_size": 4,
            "batch_size": 2,
            "full_candidates": 16,
            "balanced_mitm_partial_states": balanced_mitm_partial_state_count(4, 2),
            "ideal_unstructured_grover_scale": ideal_unstructured_search_scale(4, 2),
        },
        "interpretation": (
            "the public-target finite domain is globally injective, but balanced "
            "meet-in-the-middle recovers it after eight partial states and matches "
            "Grover's exponent for an even additive batch; private targets create "
            "nontrivial collision fibres"
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
