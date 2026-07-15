from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Protocol

from qrecon.theory.unknown_k import (
    BBHTEvaluation,
    BBHTSchedule,
    BBHTUniformCertificate,
    evaluate_bbht_schedule,
)

from .compiler import OracleResourceEstimate
from .costing import FaultTolerantGateCosts
from .grover import estimate_grover_resources


class UnknownKCostedPredicateOracle(Protocol):
    input_bits: int
    output_bits: int

    def resource_estimate(
        self, *, phase_kickback: bool = False
    ) -> OracleResourceEstimate: ...


def _non_negative(name: str, value: float) -> float:
    converted = float(value)
    if not math.isfinite(converted) or converted < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return converted


@dataclass(frozen=True)
class SpecializedClassicalSolverCosts:
    """Measured or modeled cost of the strongest matched classical solver."""

    setup_cost: float = 0.0
    per_instance_cost: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "setup_cost", _non_negative("setup_cost", self.setup_cost)
        )
        object.__setattr__(
            self,
            "per_instance_cost",
            _non_negative("per_instance_cost", self.per_instance_cost),
        )

    def total(self, instances: int) -> float:
        if instances <= 0:
            raise ValueError("instances must be positive")
        return self.setup_cost + instances * self.per_instance_cost


@dataclass(frozen=True)
class UnknownKQuantumSearchCosts:
    """Common-unit costs for one BBHT round and reusable compilation.

    `gates.measurement_cost` prices the final quantum measurement event in a run.
    `per_round_readout_cost` may additionally price decoding the measured bit word.
    `measured_candidate_verification_cost` prices the required post-measurement
    check that decides whether the sampled candidate is genuinely marked.
    """

    compilation_cost: float = 0.0
    per_round_state_loading_cost: float = 0.0
    per_round_readout_cost: float = 0.0
    measured_candidate_verification_cost: float = 0.0
    gates: FaultTolerantGateCosts = FaultTolerantGateCosts()

    def __post_init__(self) -> None:
        for name in (
            "compilation_cost",
            "per_round_state_loading_cost",
            "per_round_readout_cost",
            "measured_candidate_verification_cost",
        ):
            object.__setattr__(
                self, name, _non_negative(name, getattr(self, name))
            )


@dataclass(frozen=True)
class UnknownKMarkedCostEvaluation:
    marked: int
    search: BBHTEvaluation
    expected_variable_cost: float
    expected_t_count: float

    def to_dict(self) -> dict[str, object]:
        return {
            "marked": self.marked,
            "achieved_success": self.search.achieved_success,
            "expected_phase_oracle_calls": self.search.expected_phase_oracle_calls,
            "expected_verification_queries": self.search.expected_verification_queries,
            "expected_total_oracle_calls": self.search.expected_total_oracle_calls,
            "expected_variable_cost": self.expected_variable_cost,
            "expected_t_count": self.expected_t_count,
        }


@dataclass(frozen=True)
class UnknownKEndToEndCostReport:
    certificate: BBHTUniformCertificate
    instances: int
    classical: SpecializedClassicalSolverCosts
    quantum: UnknownKQuantumSearchCosts
    worst_cost_marked: int
    worst_expected_quantum_variable_cost: float
    maximum_expected_t_count: float
    classical_total_cost: float
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

    @property
    def minimum_instances_for_advantage(self) -> int | None:
        variable_gap = (
            self.classical.per_instance_cost
            - self.worst_expected_quantum_variable_cost
        )
        setup_gap = self.quantum.compilation_cost - self.classical.setup_cost
        if variable_gap <= 0.0:
            return (
                1
                if self.quantum.compilation_cost
                + self.worst_expected_quantum_variable_cost
                < self.classical.setup_cost + self.classical.per_instance_cost
                else None
            )
        if setup_gap < 0.0:
            return 1
        return max(1, math.floor(setup_gap / variable_gap) + 1)

    def to_dict(self) -> dict[str, object]:
        return {
            "certificate": {
                "population": self.certificate.schedule.population,
                "minimum_marked": self.certificate.minimum_marked,
                "target_success": self.certificate.target_success,
                "windows": list(self.certificate.schedule.windows),
                "certified_minimum_success": (
                    self.certificate.certified_minimum_success
                ),
            },
            "instances": self.instances,
            "classical": asdict(self.classical),
            "quantum": {
                **asdict(self.quantum),
                "gates": asdict(self.quantum.gates),
            },
            "worst_cost_marked": self.worst_cost_marked,
            "worst_expected_quantum_variable_cost": (
                self.worst_expected_quantum_variable_cost
            ),
            "maximum_expected_t_count": self.maximum_expected_t_count,
            "classical_total_cost": self.classical_total_cost,
            "quantum_total_cost": self.quantum_total_cost,
            "quantum_advantage": self.quantum_advantage,
            "speedup": self.speedup,
            "absolute_saving": self.absolute_saving,
            "minimum_instances_for_advantage": self.minimum_instances_for_advantage,
        }


def evaluate_unknown_k_quantum_cost(
    verifier: UnknownKCostedPredicateOracle,
    schedule: BBHTSchedule,
    marked: int,
    costs: UnknownKQuantumSearchCosts,
) -> UnknownKMarkedCostEvaluation:
    """Price one unknown-`K` schedule exactly for a fixed audit value of `K`.

    The schedule itself remains independent of `marked`.  The parameter is used
    only to evaluate reach probabilities and expected work after the fact.
    """

    if verifier.output_bits != 1:
        raise ValueError("unknown-K search requires a one-bit predicate oracle")
    if (1 << verifier.input_bits) != schedule.population:
        raise ValueError("verifier input width does not match schedule population")

    search = evaluate_bbht_schedule(schedule, marked)
    resource_cache: dict[int, tuple[float, int]] = {}

    def priced_iterations(iterations: int) -> tuple[float, int]:
        if iterations not in resource_cache:
            resources = estimate_grover_resources(verifier, iterations)
            resource_cache[iterations] = (
                costs.gates.execution_cost(resources),
                resources.total_t_count_upper_bound,
            )
        return resource_cache[iterations]

    expected_cost = 0.0
    expected_t_count = 0.0
    for round_report in search.rounds:
        window = round_report.window
        gate_costs = [priced_iterations(iterations) for iterations in range(window)]
        average_gate_cost = sum(value[0] for value in gate_costs) / window
        average_t_count = sum(value[1] for value in gate_costs) / window
        per_reached_round = (
            costs.per_round_state_loading_cost
            + average_gate_cost
            + costs.per_round_readout_cost
            + costs.measured_candidate_verification_cost
        )
        expected_cost += round_report.reach_probability * per_reached_round
        expected_t_count += round_report.reach_probability * average_t_count

    return UnknownKMarkedCostEvaluation(
        marked=marked,
        search=search,
        expected_variable_cost=expected_cost,
        expected_t_count=expected_t_count,
    )


def compare_unknown_k_search_to_specialized_classical(
    verifier: UnknownKCostedPredicateOracle,
    certificate: BBHTUniformCertificate,
    classical: SpecializedClassicalSolverCosts,
    quantum: UnknownKQuantumSearchCosts,
    *,
    instances: int = 1,
) -> UnknownKEndToEndCostReport:
    """Robust matched-cost comparison over the certificate's full `K` range.

    Quantum variable cost is the maximum *expected* cost over every allowed
    marked count.  This prevents a favorable hidden `K` from being selected after
    observing results.  The supplied classical cost must represent the strongest
    solver with the same candidate prior, observation, target equivalence, and
    target success.
    """

    if instances <= 0:
        raise ValueError("instances must be positive")
    if (1 << verifier.input_bits) != certificate.schedule.population:
        raise ValueError("verifier input width does not match certificate population")

    evaluations = tuple(
        evaluate_unknown_k_quantum_cost(
            verifier,
            certificate.schedule,
            marked,
            quantum,
        )
        for marked in range(
            certificate.minimum_marked,
            certificate.schedule.population + 1,
        )
    )
    worst = max(
        evaluations,
        key=lambda evaluation: (evaluation.expected_variable_cost, -evaluation.marked),
    )
    maximum_t_count = max(
        evaluation.expected_t_count for evaluation in evaluations
    )
    classical_total = classical.total(instances)
    quantum_total = (
        quantum.compilation_cost
        + instances * worst.expected_variable_cost
    )
    return UnknownKEndToEndCostReport(
        certificate=certificate,
        instances=instances,
        classical=classical,
        quantum=quantum,
        worst_cost_marked=worst.marked,
        worst_expected_quantum_variable_cost=worst.expected_variable_cost,
        maximum_expected_t_count=maximum_t_count,
        classical_total_cost=classical_total,
        quantum_total_cost=quantum_total,
    )
