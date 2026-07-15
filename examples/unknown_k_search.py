from __future__ import annotations

import json

from qrecon.theory import (
    certify_bbht_uniform_success,
    evaluate_bbht_schedule,
)


def main() -> None:
    certificate = certify_bbht_uniform_success(
        population=64,
        target_success=0.9,
        minimum_marked=1,
    )
    evaluations = {
        str(marked): evaluate_bbht_schedule(
            certificate.schedule, marked
        ).achieved_success
        for marked in (1, 2, 4, 8, 16, 32, 64)
    }
    payload = {
        "protocol": "BBHT randomized-iteration search with unknown marked count",
        "population": certificate.schedule.population,
        "minimum_marked": certificate.minimum_marked,
        "target_success": certificate.target_success,
        "growth_factor": certificate.schedule.growth_factor,
        "windows": list(certificate.schedule.windows),
        "rounds": certificate.schedule.rounds,
        "worst_success_marked": certificate.worst_success_marked,
        "certified_minimum_success": certificate.certified_minimum_success,
        "maximum_expected_phase_oracle_calls": (
            certificate.maximum_expected_phase_oracle_calls
        ),
        "maximum_expected_verification_queries": (
            certificate.maximum_expected_verification_queries
        ),
        "maximum_expected_total_oracle_calls": (
            certificate.maximum_expected_total_oracle_calls
        ),
        "worst_case_phase_oracle_calls": (
            certificate.schedule.worst_case_phase_oracle_calls
        ),
        "worst_case_verification_queries": (
            certificate.schedule.worst_case_verification_queries
        ),
        "selected_marked_count_success": evaluations,
        "claim_boundary": (
            "The online windows do not use the actual marked count. This is an "
            "exact finite query certificate, not an end-to-end cost advantage."
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
