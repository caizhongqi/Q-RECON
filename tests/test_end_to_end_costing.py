import pytest

from qrecon.oracles import ReversibleIntegerAffinePredicateOracle
from qrecon.oracles.costing import (
    ClassicalSearchCosts,
    FaultTolerantGateCosts,
    QuantumSearchCosts,
    compare_end_to_end_search_costs,
    maximum_t_cost_for_fixed_plan,
    minimum_instances_for_fixed_plan_advantage,
    optimize_quantum_search_plan,
)


def _unique_signed_target_oracle():
    # Four-bit signed candidates are -8,...,7; x >= 7 marks exactly x=7.
    return ReversibleIntegerAffinePredicateOracle(
        weights=(1,),
        bias=0,
        threshold=7,
        input_bits_per_feature=4,
        accumulator_bits=6,
        signed_inputs=True,
    )


def test_cost_optimizer_matches_target_and_prefers_two_grover_iterations_when_gates_are_cheap():
    oracle = _unique_signed_target_oracle()
    quantum = QuantumSearchCosts(
        gates=FaultTolerantGateCosts(measurement_cost=1.0, t_cost=0.0)
    )
    plan = optimize_quantum_search_plan(oracle, 0.8, quantum)
    assert plan.iterations == 2
    assert plan.repetitions == 1
    assert plan.achieved_success >= 0.8
    assert plan.t_gates_per_instance > 0


def test_end_to_end_report_can_show_advantage_or_no_advantage_in_one_cost_unit():
    oracle = _unique_signed_target_oracle()
    classical = ClassicalSearchCosts(verifier_evaluation_cost=1.0)

    cheap_quantum = QuantumSearchCosts(
        gates=FaultTolerantGateCosts(measurement_cost=1.0, t_cost=0.0)
    )
    cheap_report = compare_end_to_end_search_costs(
        oracle, 0.8, classical, cheap_quantum
    )
    assert cheap_report.classical_queries == 13
    assert cheap_report.classical_achieved_success >= 0.8
    assert cheap_report.quantum_plan.achieved_success >= 0.8
    assert cheap_report.quantum_advantage
    assert cheap_report.speedup > 1.0
    assert maximum_t_cost_for_fixed_plan(cheap_report, cheap_quantum) > 0.0

    expensive_quantum = QuantumSearchCosts(
        gates=FaultTolerantGateCosts(measurement_cost=1.0, t_cost=100.0)
    )
    expensive_report = compare_end_to_end_search_costs(
        oracle, 0.8, classical, expensive_quantum
    )
    assert not expensive_report.quantum_advantage


def test_compilation_cost_is_amortized_only_after_strict_break_even_workload():
    oracle = _unique_signed_target_oracle()
    classical = ClassicalSearchCosts(verifier_evaluation_cost=1.0)
    quantum = QuantumSearchCosts(
        compilation_cost=100.0,
        gates=FaultTolerantGateCosts(measurement_cost=1.0, t_cost=0.0),
    )
    report = compare_end_to_end_search_costs(oracle, 0.8, classical, quantum)
    assert not report.quantum_advantage
    threshold = minimum_instances_for_fixed_plan_advantage(
        report, classical, quantum
    )
    assert threshold == 9

    amortized = compare_end_to_end_search_costs(
        oracle, 0.8, classical, quantum, instances=threshold
    )
    assert amortized.quantum_advantage


def test_invalid_or_empty_search_cost_inputs_are_rejected():
    with pytest.raises(ValueError):
        FaultTolerantGateCosts(t_cost=-1.0)
    empty = ReversibleIntegerAffinePredicateOracle(
        weights=(0,),
        bias=-1,
        threshold=0,
        input_bits_per_feature=2,
        accumulator_bits=3,
        signed_inputs=True,
    )
    with pytest.raises(ValueError):
        optimize_quantum_search_plan(empty, 0.9, QuantumSearchCosts())
