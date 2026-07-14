from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Protocol

from qrecon.theory.search import (
    classical_queries_for_success,
    classical_success_without_replacement,
    grover_success,
    optimal_standard_grover_iterations,
)

from .compiler import OracleResourceEstimate
from .grover import GroverResourceEstimate, estimate_grover_resources


class CostedPredicateOracle(Protocol):
    input_bits: int
    output_bits: int

    def marked_inputs(self) -> tuple[int, ...]: ...

    def phase_sign(self, input_word: int) -> int: ...

    def resource_estimate(self, *, phase_kickback: bool = False) -> OracleResourceEstimate: ...


def _non_negative(name: str, value: float) -> float:
    converted = float(value)
    if not math.isfinite(converted) or converted < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return converted


@dataclass(frozen=True)
class FaultTolerantGateCosts:
    """One common abstract cost unit for a logical quantum execution.

    ``t_cost`` prices the decomposed T count, so Toffoli gates are not charged a
    second time. ``qubit_depth_cost`` optionally prices the logical-qubit by
    logical-depth space-time volume. All other fields price their named logical
    gate or final measurement/readout event.
    """

    x_cost: float = 0.0
    cnot_cost: float = 0.0
    h_cost: float = 0.0
    z_cost: float = 0.0
    t_cost: float = 1.0
    qubit_depth_cost: float = 0.0
    measurement_cost: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "x_cost",
            "cnot_cost",
            "h_cost",
            "z_cost",
            "t_cost",
            "qubit_depth_cost",
            "measurement_cost",
        ):
            object.__setattr__(self, name, _non_negative(name, getattr(self, name)))

    def execution_cost(self, resources: GroverResourceEstimate) -> float:
        return self.execution_cost_without_t(resources) + (
            resources.total_t_count_upper_bound * self.t_cost
        )

    def execution_cost_without_t(self, resources: GroverResourceEstimate) -> float:
        return (
            resources.total_x_gates * self.x_cost
            + resources.total_cnot_gates * self.cnot_cost
            + resources.total_h_gates * self.h_cost
            + resources.oracle.z_gates * resources.iterations * self.z_cost
            + resources.logical_qubits
            * max(1, resources.total_t_depth_upper_bound)
            * self.qubit_depth_cost
            + self.measurement_cost
        )


@dataclass(frozen=True)
class ClassicalSearchCosts:
    setup_cost: float = 0.0
    candidate_preparation_cost: float = 0.0
    verifier_evaluation_cost: float = 1.0
    readout_cost: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "setup_cost",
            "candidate_preparation_cost",
            "verifier_evaluation_cost",
            "readout_cost",
        ):
            object.__setattr__(self, name, _non_negative(name, getattr(self, name)))

    @property
    def cost_per_query(self) -> float:
        return self.candidate_preparation_cost + self.verifier_evaluation_cost


@dataclass(frozen=True)
class QuantumSearchCosts:
    compilation_cost: float = 0.0
    per_run_state_loading_cost: float = 0.0
    per_run_readout_cost: float = 0.0
    gates: FaultTolerantGateCosts = FaultTolerantGateCosts()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "compilation_cost", _non_negative("compilation_cost", self.compilation_cost)
        )
        object.__setattr__(
            self,
            "per_run_state_loading_cost",
            _non_negative("per_run_state_loading_cost", self.per_run_state_loading_cost),
        )
        object.__setattr__(
            self,
            "per_run_readout_cost",
            _non_negative("per_run_readout_cost", self.per_run_readout_cost),
        )


@dataclass(frozen=True)
class QuantumSearchPlan:
    iterations: int
    repetitions: int
    one_run_success: float
    achieved_success: float
    one_run_resources: GroverResourceEstimate
    variable_cost_per_instance: float
    variable_cost_without_t_per_instance: float
    t_gates_per_instance: int

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["one_run_resources"] = self.one_run_resources.to_dict()
        return result


@dataclass(frozen=True)
class EndToEndSearchCostReport:
    population: int
    marked: int
    target_success: float
    instances: int
    classical_queries: int
    classical_achieved_success: float
    classical_total_cost: float
    quantum_plan: QuantumSearchPlan
    quantum_total_cost: float

    @property
    def quantum_advantage(self) -> bool:
        return self.quantum_total_cost < self.classical_total_cost

    @property
    def speedup(self) -> float:
        if self.quantum_total_cost == 0.0:
            return math.inf if self.classical_total_cost > 0.0 else 1.0
        return self.classical_total_cost / self.quantum_total_cost

    @property
    def absolute_saving(self) -> float:
        return self.classical_total_cost - self.quantum_total_cost

    def to_dict(self) -> dict[str, object]:
        return {
            "population": self.population,
            "marked": self.marked,
            "target_success": self.target_success,
            "instances": self.instances,
            "classical_queries": self.classical_queries,
            "classical_achieved_success": self.classical_achieved_success,
            "classical_total_cost": self.classical_total_cost,
            "quantum_plan": self.quantum_plan.to_dict(),
            "quantum_total_cost": self.quantum_total_cost,
            "quantum_advantage": self.quantum_advantage,
            "speedup": self.speedup,
            "absolute_saving": self.absolute_saving,
        }


def _repetitions_for_target(one_run_success: float, target_success: float) -> int | None:
    probability = float(one_run_success)
    target = float(target_success)
    if target <= 0.0:
        return 0
    if probability <= 0.0:
        return None
    if probability >= target or probability >= 1.0:
        return 1
    if target >= 1.0:
        return None
    return max(1, math.ceil(math.log1p(-target) / math.log1p(-probability)))


def optimize_quantum_search_plan(
    verifier: CostedPredicateOracle,
    target_success: float,
    costs: QuantumSearchCosts,
    *,
    max_iterations: int | None = None,
) -> QuantumSearchPlan:
    """Choose iterations and independent repetitions with minimum modeled cost."""

    target = float(target_success)
    if not math.isfinite(target) or target <= 0.0 or target > 1.0:
        raise ValueError("target_success must lie in (0, 1]")
    marked = len(verifier.marked_inputs())
    population = 1 << verifier.input_bits
    if marked == 0:
        raise ValueError("the verifier has no marked candidate")
    optimum = optimal_standard_grover_iterations(population, marked)
    if optimum is None:
        raise ValueError("the verifier has no searchable marked candidate")
    limit = optimum if max_iterations is None else min(optimum, int(max_iterations))
    if limit < 0:
        raise ValueError("max_iterations must be non-negative")

    best: QuantumSearchPlan | None = None
    for iterations in range(limit + 1):
        one_run_success = grover_success(population, marked, iterations)
        repetitions = _repetitions_for_target(one_run_success, target)
        if repetitions is None:
            continue
        resources = estimate_grover_resources(verifier, iterations)
        one_run_without_t = (
            costs.per_run_state_loading_cost
            + costs.gates.execution_cost_without_t(resources)
            + costs.per_run_readout_cost
        )
        one_run_cost = (
            costs.per_run_state_loading_cost
            + costs.gates.execution_cost(resources)
            + costs.per_run_readout_cost
        )
        variable = repetitions * one_run_cost
        variable_without_t = repetitions * one_run_without_t
        t_gates = repetitions * resources.total_t_count_upper_bound
        achieved = 1.0 - (1.0 - one_run_success) ** repetitions
        candidate = QuantumSearchPlan(
            iterations=iterations,
            repetitions=repetitions,
            one_run_success=one_run_success,
            achieved_success=achieved,
            one_run_resources=resources,
            variable_cost_per_instance=variable,
            variable_cost_without_t_per_instance=variable_without_t,
            t_gates_per_instance=t_gates,
        )
        if best is None or (
            candidate.variable_cost_per_instance,
            candidate.t_gates_per_instance,
            candidate.iterations,
        ) < (
            best.variable_cost_per_instance,
            best.t_gates_per_instance,
            best.iterations,
        ):
            best = candidate
    if best is None:
        raise ValueError("no finite standard-Grover repetition plan reaches the target")
    return best


def compare_end_to_end_search_costs(
    verifier: CostedPredicateOracle,
    target_success: float,
    classical: ClassicalSearchCosts,
    quantum: QuantumSearchCosts,
    *,
    instances: int = 1,
    max_quantum_iterations: int | None = None,
) -> EndToEndSearchCostReport:
    """Compare matched-success classical and coherent search in one cost unit."""

    if instances <= 0:
        raise ValueError("instances must be positive")
    marked = len(verifier.marked_inputs())
    population = 1 << verifier.input_bits
    classical_queries = classical_queries_for_success(
        population, marked, target_success
    )
    if classical_queries is None:
        raise ValueError("the verifier has no marked candidate")
    classical_success = classical_success_without_replacement(
        population, marked, classical_queries
    )
    classical_variable = (
        classical_queries * classical.cost_per_query + classical.readout_cost
    )
    classical_total = classical.setup_cost + instances * classical_variable

    plan = optimize_quantum_search_plan(
        verifier,
        target_success,
        quantum,
        max_iterations=max_quantum_iterations,
    )
    quantum_total = quantum.compilation_cost + instances * plan.variable_cost_per_instance
    return EndToEndSearchCostReport(
        population=population,
        marked=marked,
        target_success=float(target_success),
        instances=instances,
        classical_queries=classical_queries,
        classical_achieved_success=classical_success,
        classical_total_cost=classical_total,
        quantum_plan=plan,
        quantum_total_cost=quantum_total,
    )


def maximum_t_cost_for_fixed_plan(
    report: EndToEndSearchCostReport,
    quantum: QuantumSearchCosts,
) -> float | None:
    """Open T-cost threshold for advantage under the report's selected plan."""

    t_gates = report.instances * report.quantum_plan.t_gates_per_instance
    if t_gates == 0:
        return None
    base = quantum.compilation_cost + report.instances * (
        report.quantum_plan.variable_cost_without_t_per_instance
    )
    return (report.classical_total_cost - base) / t_gates


def minimum_instances_for_fixed_plan_advantage(
    report: EndToEndSearchCostReport,
    classical: ClassicalSearchCosts,
    quantum: QuantumSearchCosts,
) -> int | None:
    """Minimum workload amortizing setup under the selected query plan."""

    classical_variable = (
        report.classical_queries * classical.cost_per_query + classical.readout_cost
    )
    quantum_variable = report.quantum_plan.variable_cost_per_instance
    gap = classical_variable - quantum_variable
    setup_gap = quantum.compilation_cost - classical.setup_cost
    if gap <= 0.0:
        return 1 if quantum.compilation_cost + quantum_variable < classical.setup_cost + classical_variable else None
    if setup_gap < 0.0:
        return 1
    return max(1, math.floor(setup_gap / gap) + 1)
