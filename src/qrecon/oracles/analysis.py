from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass

from .compiler import TruthTableOracle


@dataclass(frozen=True)
class FiniteIdentifiabilityReport:
    population: int
    distinct_observations: int
    injective: bool
    colliding_candidates: int
    largest_fibre: int
    fibre_histogram: tuple[tuple[int, int], ...]
    uniform_exact_success: float
    conditional_min_entropy_bits: float

    def to_dict(self) -> dict[str, int | bool | float | list[list[int]]]:
        result = asdict(self)
        result["fibre_histogram"] = [list(item) for item in self.fibre_histogram]
        return result


def analyze_finite_oracle(oracle: TruthTableOracle) -> FiniteIdentifiabilityReport:
    fibres: dict[int, list[int]] = defaultdict(list)
    for candidate, observation in enumerate(oracle.table):
        fibres[observation].append(candidate)
    sizes = tuple(len(values) for values in fibres.values())
    histogram = tuple(sorted(Counter(sizes).items()))
    population = len(oracle.table)
    distinct = len(fibres)
    success = distinct / population
    return FiniteIdentifiabilityReport(
        population=population,
        distinct_observations=distinct,
        injective=distinct == population,
        colliding_candidates=sum(size for size in sizes if size > 1),
        largest_fibre=max(sizes),
        fibre_histogram=histogram,
        uniform_exact_success=success,
        conditional_min_entropy_bits=-math.log2(success),
    )
