from __future__ import annotations

import math

from .unknown_k import (
    BBHTSchedule,
    BBHTUniformCertificate,
    build_bbht_schedule,
    randomized_grover_round_success,
)


def certify_staged_bbht_uniform_success(
    population: int,
    target_success: float,
    *,
    minimum_marked: int = 1,
    growth_factor: float = 8.0 / 7.0,
    attempts_per_stage: int = 1,
    max_stages: int = 256,
    max_exact_population: int = 4096,
) -> BBHTUniformCertificate:
    """Certify a public staged BBHT schedule for every allowed positive ``K``.

    A stage has one geometrically selected window and repeats that randomized
    measurement/verification attempt ``attempts_per_stage`` times before the
    window grows. Certification is checked only after complete stages, so the
    returned schedule exactly implements the declared manifest contract.
    """

    count = int(population)
    if count <= 0:
        raise ValueError("population must be positive")
    target = float(target_success)
    if not math.isfinite(target) or target <= 0.0 or target >= 1.0:
        raise ValueError("target_success must lie strictly between zero and one")
    lower = int(minimum_marked)
    if lower <= 0 or lower > count:
        raise ValueError("minimum_marked must lie in [1, population]")
    attempts = int(attempts_per_stage)
    if attempts <= 0:
        raise ValueError("attempts_per_stage must be positive")
    stages = int(max_stages)
    if stages <= 0:
        raise ValueError("max_stages must be positive")
    exact_limit = int(max_exact_population)
    if exact_limit <= 0:
        raise ValueError("max_exact_population must be positive")
    if count > exact_limit:
        raise ValueError(
            "population exceeds max_exact_population for exhaustive K certification"
        )

    base = build_bbht_schedule(
        count,
        stages,
        growth_factor=growth_factor,
    )
    marked_values = tuple(range(lower, count + 1))
    failure = {marked: 1.0 for marked in marked_values}
    expected_phase = {marked: 0.0 for marked in marked_values}
    expected_verify = {marked: 0.0 for marked in marked_values}
    windows: list[int] = []

    for window in base.windows:
        for _ in range(attempts):
            windows.append(window)
            for marked in marked_values:
                reach = failure[marked]
                expected_phase[marked] += reach * (window - 1) / 2.0
                expected_verify[marked] += reach
                failure[marked] *= 1.0 - randomized_grover_round_success(
                    count,
                    marked,
                    window,
                )

        success = {marked: 1.0 - failure[marked] for marked in marked_values}
        worst_marked = min(marked_values, key=lambda value: (success[value], value))
        minimum_success = success[worst_marked]
        if minimum_success + 1e-15 >= target:
            schedule = BBHTSchedule(
                count,
                float(growth_factor),
                tuple(windows),
            )
            return BBHTUniformCertificate(
                schedule=schedule,
                minimum_marked=lower,
                target_success=target,
                worst_success_marked=worst_marked,
                certified_minimum_success=minimum_success,
                maximum_expected_phase_oracle_calls=max(expected_phase.values()),
                maximum_expected_verification_queries=max(expected_verify.values()),
            )

    raise ValueError(
        "max_stages is insufficient to certify the requested uniform success"
    )
