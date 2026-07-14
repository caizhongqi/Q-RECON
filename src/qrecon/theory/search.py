from __future__ import annotations

import math
from dataclasses import dataclass


def _validate_population(population: int, marked: int) -> None:
    if population <= 0:
        raise ValueError("population must be positive")
    if marked < 0 or marked > population:
        raise ValueError("marked must lie in [0, population]")


def classical_success_without_replacement(
    population: int, marked: int, queries: int
) -> float:
    """Success of uniformly querying distinct candidates without replacement."""

    _validate_population(population, marked)
    if queries < 0 or queries > population:
        raise ValueError("queries must lie in [0, population]")
    if queries == 0 or marked == 0:
        return 0.0
    if queries > population - marked:
        return 1.0
    failure = math.comb(population - marked, queries) / math.comb(population, queries)
    return 1.0 - failure


def classical_queries_for_success(
    population: int, marked: int, target_success: float
) -> int | None:
    """Minimum distinct classical queries needed for a target success probability."""

    _validate_population(population, marked)
    target = float(target_success)
    if not math.isfinite(target) or target < 0.0 or target > 1.0:
        raise ValueError("target_success must lie in [0, 1]")
    if target <= 0.0:
        return 0
    if marked == 0:
        return None

    low, high = 1, population
    while low < high:
        middle = (low + high) // 2
        if classical_success_without_replacement(population, marked, middle) >= target:
            high = middle
        else:
            low = middle + 1
    return low


def expected_classical_queries(population: int, marked: int) -> float:
    """Expected queries to the first marked item under a random permutation."""

    _validate_population(population, marked)
    if marked == 0:
        return math.inf
    return (population + 1.0) / (marked + 1.0)


def grover_success(population: int, marked: int, iterations: int) -> float:
    """Ideal standard-Grover success after ``iterations`` oracle calls."""

    _validate_population(population, marked)
    if iterations < 0:
        raise ValueError("iterations must be non-negative")
    if marked == 0:
        return 0.0
    theta = math.asin(math.sqrt(marked / population))
    return math.sin((2 * iterations + 1) * theta) ** 2


def optimal_standard_grover_iterations(population: int, marked: int) -> int | None:
    """Integer iteration count maximizing the first standard-Grover peak."""

    _validate_population(population, marked)
    if marked == 0:
        return None
    theta = math.asin(math.sqrt(marked / population))
    real_optimum = math.pi / (4.0 * theta) - 0.5
    upper = max(0, math.ceil(real_optimum) + 1)
    candidates = range(0, upper + 1)
    return max(candidates, key=lambda value: (grover_success(population, marked, value), -value))


def grover_queries_for_success(
    population: int, marked: int, target_success: float
) -> int | None:
    """Minimum standard-Grover iterations reaching a target before its first peak.

    ``None`` means either no marked item exists or fixed-phase standard Grover
    cannot reach the requested target at an integer iteration. Exact-amplitude
    variants are outside this helper's scope.
    """

    _validate_population(population, marked)
    target = float(target_success)
    if not math.isfinite(target) or target < 0.0 or target > 1.0:
        raise ValueError("target_success must lie in [0, 1]")
    if target <= 0.0:
        return 0
    optimum = optimal_standard_grover_iterations(population, marked)
    if optimum is None:
        return None
    for iterations in range(optimum + 1):
        if grover_success(population, marked, iterations) >= target:
            return iterations
    return None


@dataclass(frozen=True)
class SearchComparison:
    population: int
    marked: int
    target_success: float
    classical_queries: int | None
    grover_queries: int | None
    classical_achieved_success: float
    grover_achieved_success: float

    @property
    def query_ratio(self) -> float | None:
        if self.classical_queries is None or self.grover_queries in (None, 0):
            return None
        return self.classical_queries / self.grover_queries


def compare_search_queries(
    population: int, marked: int, target_success: float
) -> SearchComparison:
    """Compare exact classical sampling and ideal standard-Grover query counts."""

    classical = classical_queries_for_success(population, marked, target_success)
    grover = grover_queries_for_success(population, marked, target_success)
    classical_success = (
        0.0
        if classical is None
        else classical_success_without_replacement(population, marked, classical)
    )
    grover_achieved = 0.0 if grover is None else grover_success(population, marked, grover)
    return SearchComparison(
        population=population,
        marked=marked,
        target_success=float(target_success),
        classical_queries=classical,
        grover_queries=grover,
        classical_achieved_success=classical_success,
        grover_achieved_success=grover_achieved,
    )
