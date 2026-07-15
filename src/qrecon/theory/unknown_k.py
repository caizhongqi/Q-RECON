from __future__ import annotations

import math
from dataclasses import dataclass


def _validate_population(population: int, marked: int | None = None) -> None:
    if population <= 0:
        raise ValueError("population must be positive")
    if marked is not None and (marked < 0 or marked > population):
        raise ValueError("marked must lie in [0, population]")


def _validate_growth(growth_factor: float) -> float:
    value = float(growth_factor)
    if not math.isfinite(value) or value <= 1.0 or value >= 4.0 / 3.0:
        raise ValueError("growth_factor must lie strictly between 1 and 4/3")
    return value


@dataclass(frozen=True)
class BBHTSchedule:
    """A marked-count-independent randomized Grover schedule.

    In round ``r``, choose an integer ``j`` uniformly from
    ``{0, ..., windows[r]-1}``, perform ``j`` Grover iterations, measure, and
    verify the measured candidate.  The windows depend only on the public search
    population and growth factor, never on the unknown marked count.
    """

    population: int
    growth_factor: float
    windows: tuple[int, ...]

    def __post_init__(self) -> None:
        _validate_population(self.population)
        _validate_growth(self.growth_factor)
        cap = max(1, math.ceil(math.sqrt(self.population)))
        if any(window <= 0 or window > cap for window in self.windows):
            raise ValueError("every BBHT window must lie in [1, ceil(sqrt(population))]")
        if any(left > right for left, right in zip(self.windows, self.windows[1:])):
            raise ValueError("BBHT windows must be non-decreasing")

    @property
    def rounds(self) -> int:
        return len(self.windows)

    @property
    def worst_case_phase_oracle_calls(self) -> int:
        return sum(window - 1 for window in self.windows)

    @property
    def worst_case_verification_queries(self) -> int:
        return self.rounds

    @property
    def worst_case_total_oracle_calls(self) -> int:
        return self.worst_case_phase_oracle_calls + self.worst_case_verification_queries


@dataclass(frozen=True)
class BBHTRoundReport:
    round_index: int
    window: int
    reach_probability: float
    conditional_success: float
    success_mass: float
    cumulative_success: float
    expected_phase_oracle_calls: float


@dataclass(frozen=True)
class BBHTEvaluation:
    schedule: BBHTSchedule
    marked: int
    achieved_success: float
    expected_phase_oracle_calls: float
    expected_verification_queries: float
    rounds: tuple[BBHTRoundReport, ...]

    @property
    def expected_total_oracle_calls(self) -> float:
        return self.expected_phase_oracle_calls + self.expected_verification_queries


@dataclass(frozen=True)
class BBHTUniformCertificate:
    """Exact finite certificate over every allowed positive marked count."""

    schedule: BBHTSchedule
    minimum_marked: int
    target_success: float
    worst_success_marked: int
    certified_minimum_success: float
    maximum_expected_phase_oracle_calls: float
    maximum_expected_verification_queries: float

    @property
    def maximum_expected_total_oracle_calls(self) -> float:
        return (
            self.maximum_expected_phase_oracle_calls
            + self.maximum_expected_verification_queries
        )


def randomized_grover_round_success(
    population: int, marked: int, window: int
) -> float:
    """Exact success of one randomized-iteration Grover round.

    The round chooses ``j`` uniformly from ``0, ..., window-1`` and therefore has

    ``(1/window) * sum_j sin^2((2*j+1)*theta)``, where
    ``theta = asin(sqrt(marked/population))``.

    The closed form is used away from its removable endpoint singularities.
    """

    _validate_population(population, marked)
    if window <= 0:
        raise ValueError("window must be positive")
    if marked == 0:
        return 0.0
    if marked == population:
        return 1.0

    theta = math.asin(math.sqrt(marked / population))
    denominator = 4.0 * window * math.sin(2.0 * theta)
    success = 0.5 - math.sin(4.0 * window * theta) / denominator
    return min(1.0, max(0.0, float(success)))


def build_bbht_schedule(
    population: int,
    rounds: int,
    *,
    growth_factor: float = 8.0 / 7.0,
) -> BBHTSchedule:
    """Build the standard geometrically growing BBHT window schedule.

    This function intentionally receives no marked-count argument.  The real
    window scale starts at one, grows by ``growth_factor``, and is capped at
    ``sqrt(population)``.  The implemented integer window is its ceiling.
    """

    _validate_population(population)
    growth = _validate_growth(growth_factor)
    if rounds < 0:
        raise ValueError("rounds must be non-negative")

    cap = math.sqrt(population)
    integer_cap = max(1, math.ceil(cap))
    scale = 1.0
    windows: list[int] = []
    for _ in range(rounds):
        windows.append(max(1, min(integer_cap, math.ceil(scale))))
        scale = min(cap, growth * scale)
    return BBHTSchedule(population, growth, tuple(windows))


def evaluate_bbht_schedule(schedule: BBHTSchedule, marked: int) -> BBHTEvaluation:
    """Exactly evaluate success and expected queries for a fixed marked count."""

    _validate_population(schedule.population, marked)
    failure_probability = 1.0
    expected_phase_calls = 0.0
    expected_verifications = 0.0
    reports: list[BBHTRoundReport] = []

    for index, window in enumerate(schedule.windows, start=1):
        reach = failure_probability
        conditional = randomized_grover_round_success(
            schedule.population, marked, window
        )
        success_mass = reach * conditional
        expected_round_phase_calls = reach * (window - 1) / 2.0
        expected_phase_calls += expected_round_phase_calls
        expected_verifications += reach
        failure_probability *= 1.0 - conditional
        cumulative = 1.0 - failure_probability
        reports.append(
            BBHTRoundReport(
                round_index=index,
                window=window,
                reach_probability=reach,
                conditional_success=conditional,
                success_mass=success_mass,
                cumulative_success=cumulative,
                expected_phase_oracle_calls=expected_round_phase_calls,
            )
        )

    return BBHTEvaluation(
        schedule=schedule,
        marked=marked,
        achieved_success=1.0 - failure_probability,
        expected_phase_oracle_calls=expected_phase_calls,
        expected_verification_queries=expected_verifications,
        rounds=tuple(reports),
    )


def certify_bbht_uniform_success(
    population: int,
    target_success: float,
    *,
    minimum_marked: int = 1,
    growth_factor: float = 8.0 / 7.0,
    max_rounds: int = 256,
    max_exact_population: int = 4096,
) -> BBHTUniformCertificate:
    """Find the shortest schedule certified for every allowed marked count.

    The online schedule is independent of ``marked``.  Offline, this function
    exhaustively evaluates every integer marked count in
    ``[minimum_marked, population]`` and returns the first schedule prefix whose
    minimum success reaches ``target_success``.  This is an exact finite-space
    certificate, not an asymptotic or Monte Carlo estimate.
    """

    _validate_population(population)
    target = float(target_success)
    if not math.isfinite(target) or target <= 0.0 or target >= 1.0:
        raise ValueError("target_success must lie strictly between 0 and 1")
    if minimum_marked <= 0 or minimum_marked > population:
        raise ValueError("minimum_marked must lie in [1, population]")
    growth = _validate_growth(growth_factor)
    if max_rounds <= 0:
        raise ValueError("max_rounds must be positive")
    if max_exact_population <= 0:
        raise ValueError("max_exact_population must be positive")
    if population > max_exact_population:
        raise ValueError(
            "population exceeds max_exact_population for exhaustive K certification"
        )

    marked_values = tuple(range(minimum_marked, population + 1))
    failure = {marked: 1.0 for marked in marked_values}
    expected_phase = {marked: 0.0 for marked in marked_values}
    expected_verify = {marked: 0.0 for marked in marked_values}
    complete_schedule = build_bbht_schedule(
        population, max_rounds, growth_factor=growth
    )

    for round_index, window in enumerate(complete_schedule.windows, start=1):
        for marked in marked_values:
            reach = failure[marked]
            expected_phase[marked] += reach * (window - 1) / 2.0
            expected_verify[marked] += reach
            conditional = randomized_grover_round_success(
                population, marked, window
            )
            failure[marked] *= 1.0 - conditional

        success = {marked: 1.0 - failure[marked] for marked in marked_values}
        worst_marked = min(marked_values, key=lambda value: (success[value], value))
        minimum_success = success[worst_marked]
        if minimum_success + 1e-15 >= target:
            schedule = BBHTSchedule(
                population,
                growth,
                complete_schedule.windows[:round_index],
            )
            return BBHTUniformCertificate(
                schedule=schedule,
                minimum_marked=minimum_marked,
                target_success=target,
                worst_success_marked=worst_marked,
                certified_minimum_success=minimum_success,
                maximum_expected_phase_oracle_calls=max(expected_phase.values()),
                maximum_expected_verification_queries=max(expected_verify.values()),
            )

    raise ValueError(
        "max_rounds is insufficient to certify the requested uniform success"
    )
