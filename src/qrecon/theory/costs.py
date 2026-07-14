from __future__ import annotations

import math
from dataclasses import dataclass


def _non_negative_finite(name: str, value: float) -> float:
    converted = float(value)
    if not math.isfinite(converted) or converted < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return converted


@dataclass(frozen=True)
class AlgorithmCost:
    """A transparent setup-plus-per-instance cost model.

    ``setup_cost`` is paid once and may include compilation. Per instance, the
    model pays ``fixed_instance_cost`` plus ``queries * cost_per_query``. All
    values are abstract non-negative cost units chosen consistently by callers.
    """

    setup_cost: float = 0.0
    fixed_instance_cost: float = 0.0
    queries: int = 0
    cost_per_query: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "setup_cost", _non_negative_finite("setup_cost", self.setup_cost))
        object.__setattr__(
            self,
            "fixed_instance_cost",
            _non_negative_finite("fixed_instance_cost", self.fixed_instance_cost),
        )
        object.__setattr__(
            self, "cost_per_query", _non_negative_finite("cost_per_query", self.cost_per_query)
        )
        if self.queries < 0:
            raise ValueError("queries must be non-negative")

    @property
    def variable_cost(self) -> float:
        return self.fixed_instance_cost + self.queries * self.cost_per_query

    def total(self, instances: int = 1) -> float:
        if instances <= 0:
            raise ValueError("instances must be positive")
        return self.setup_cost + instances * self.variable_cost


@dataclass(frozen=True)
class CostComparison:
    instances: int
    classical_total: float
    quantum_total: float

    @property
    def quantum_advantage(self) -> bool:
        return self.quantum_total < self.classical_total

    @property
    def absolute_saving(self) -> float:
        return self.classical_total - self.quantum_total

    @property
    def speedup(self) -> float:
        if self.quantum_total == 0.0:
            return math.inf if self.classical_total > 0.0 else 1.0
        return self.classical_total / self.quantum_total


def compare_algorithm_costs(
    classical: AlgorithmCost, quantum: AlgorithmCost, instances: int = 1
) -> CostComparison:
    """Compare end-to-end costs under one common cost unit."""

    return CostComparison(
        instances=instances,
        classical_total=classical.total(instances),
        quantum_total=quantum.total(instances),
    )


def minimum_instances_for_quantum_advantage(
    classical: AlgorithmCost, quantum: AlgorithmCost
) -> int | None:
    """Smallest positive workload for strict quantum cost advantage, if one exists."""

    variable_gap = classical.variable_cost - quantum.variable_cost
    setup_gap = quantum.setup_cost - classical.setup_cost

    if variable_gap <= 0.0:
        return 1 if quantum.total(1) < classical.total(1) else None
    if setup_gap < 0.0:
        return 1

    instances = max(1, math.floor(setup_gap / variable_gap) + 1)
    while quantum.total(instances) >= classical.total(instances):
        instances += 1
    return instances


def maximum_quantum_query_cost_for_advantage(
    classical: AlgorithmCost,
    quantum: AlgorithmCost,
    instances: int = 1,
) -> float | None:
    """Largest quantum per-query cost compatible with strict total-cost advantage.

    The returned threshold is an open bound: advantage requires the actual
    quantum query cost to be strictly smaller. ``None`` is returned when the
    quantum algorithm uses no queries.
    """

    if instances <= 0:
        raise ValueError("instances must be positive")
    if quantum.queries == 0:
        return None
    budget = (
        classical.total(instances)
        - quantum.setup_cost
        - instances * quantum.fixed_instance_cost
    )
    return budget / (instances * quantum.queries)


def oracle_error_success_lower_bound(
    ideal_success: float, queries: int, per_query_operational_error: float
) -> float:
    """Hybrid-argument lower bound for an approximate coherent oracle.

    ``per_query_operational_error`` is defined as the maximum trace-distance
    change caused by replacing one ideal oracle call with one implemented call,
    including arbitrary reference systems. A ``queries``-call algorithm then
    loses at most ``queries * error`` in any event probability.
    """

    success = float(ideal_success)
    error = float(per_query_operational_error)
    if not math.isfinite(success) or success < 0.0 or success > 1.0:
        raise ValueError("ideal_success must lie in [0, 1]")
    if queries < 0:
        raise ValueError("queries must be non-negative")
    if not math.isfinite(error) or error < 0.0 or error > 1.0:
        raise ValueError("per_query_operational_error must lie in [0, 1]")
    return max(0.0, success - queries * error)
