import pytest

from qrecon.oracles import (
    FaultTolerantGateCosts,
    OracleResourceEstimate,
    SpecializedClassicalSolverCosts,
    UnknownKQuantumSearchCosts,
    compare_unknown_k_search_to_specialized_classical,
    evaluate_unknown_k_quantum_cost,
)
from qrecon.theory import certify_bbht_uniform_success


class StubPredicateOracle:
    input_bits = 4
    output_bits = 1

    def resource_estimate(self, *, phase_kickback: bool = False):
        return OracleResourceEstimate(
            input_qubits=4,
            output_qubits=1,
            peak_clean_ancillas=3,
            logical_qubits=8,
            controlled_x_terms=5,
            negative_control_x_gates=0,
            x_gates=1,
            cnot_gates=2,
            toffoli_gates=3,
            h_gates=0,
            z_gates=0,
            t_count_upper_bound=21,
            t_depth_upper_bound=9,
            logical_depth_upper_bound=12,
            synthesis="test resource contract",
        )


def _costs(compilation_cost=0.0):
    return UnknownKQuantumSearchCosts(
        compilation_cost=compilation_cost,
        per_round_state_loading_cost=13.0,
        per_round_readout_cost=17.0,
        measured_candidate_verification_cost=19.0,
        gates=FaultTolerantGateCosts(
            x_cost=3.0,
            cnot_cost=5.0,
            h_cost=2.0,
            t_cost=7.0,
            measurement_cost=11.0,
        ),
    )


def test_one_round_full_marked_cost_has_no_hidden_phase_query():
    oracle = StubPredicateOracle()
    certificate = certify_bbht_uniform_success(
        16, 0.9, minimum_marked=16
    )
    assert certificate.schedule.windows == (1,)

    evaluation = evaluate_unknown_k_quantum_cost(
        oracle, certificate.schedule, marked=16, costs=_costs()
    )
    # j=0 still prepares |+>^4 and |->: five H, one X, then measures.
    gate_cost = 5 * 2.0 + 1 * 3.0 + 11.0
    expected = 13.0 + gate_cost + 17.0 + 19.0
    assert evaluation.expected_variable_cost == pytest.approx(expected)
    assert evaluation.expected_t_count == 0.0
    assert evaluation.search.expected_phase_oracle_calls == 0.0
    assert evaluation.search.expected_verification_queries == 1.0


def test_uniform_cost_report_uses_maximum_expected_cost_over_all_k():
    oracle = StubPredicateOracle()
    certificate = certify_bbht_uniform_success(16, 0.9)
    quantum = _costs(compilation_cost=500.0)

    evaluations = [
        evaluate_unknown_k_quantum_cost(
            oracle, certificate.schedule, marked, quantum
        )
        for marked in range(1, 17)
    ]
    worst = max(evaluations, key=lambda item: (item.expected_variable_cost, -item.marked))
    report = compare_unknown_k_search_to_specialized_classical(
        oracle,
        certificate,
        SpecializedClassicalSolverCosts(
            setup_cost=100.0,
            per_instance_cost=1_000_000.0,
        ),
        quantum,
        instances=3,
    )

    assert report.worst_cost_marked == worst.marked
    assert report.worst_expected_quantum_variable_cost == pytest.approx(
        worst.expected_variable_cost
    )
    assert report.quantum_total_cost == pytest.approx(
        500.0 + 3 * worst.expected_variable_cost
    )
    assert report.classical_total_cost == pytest.approx(100.0 + 3_000_000.0)
    assert report.quantum_advantage
    assert report.maximum_expected_t_count >= worst.expected_t_count


def test_cost_report_does_not_force_an_advantage_against_a_strong_solver():
    oracle = StubPredicateOracle()
    certificate = certify_bbht_uniform_success(16, 0.9)
    quantum = _costs(compilation_cost=500.0)
    report = compare_unknown_k_search_to_specialized_classical(
        oracle,
        certificate,
        SpecializedClassicalSolverCosts(setup_cost=0.0, per_instance_cost=1.0),
        quantum,
    )

    assert not report.quantum_advantage
    assert report.speedup < 1.0
    assert report.minimum_instances_for_advantage is None
    payload = report.to_dict()
    assert payload["quantum_advantage"] is False
    assert payload["certificate"]["target_success"] == pytest.approx(0.9)


def test_setup_amortization_threshold_is_strict():
    oracle = StubPredicateOracle()
    certificate = certify_bbht_uniform_success(16, 0.9)
    quantum = _costs(compilation_cost=10_000.0)
    probe = compare_unknown_k_search_to_specialized_classical(
        oracle,
        certificate,
        SpecializedClassicalSolverCosts(
            setup_cost=0.0,
            per_instance_cost=100_000.0,
        ),
        quantum,
    )
    threshold = probe.minimum_instances_for_advantage
    assert threshold is not None

    before = max(1, threshold - 1)
    report_before = compare_unknown_k_search_to_specialized_classical(
        oracle,
        certificate,
        probe.classical,
        quantum,
        instances=before,
    )
    report_at = compare_unknown_k_search_to_specialized_classical(
        oracle,
        certificate,
        probe.classical,
        quantum,
        instances=threshold,
    )
    if threshold > 1:
        assert not report_before.quantum_advantage
    assert report_at.quantum_advantage


def test_cost_contract_rejects_mismatched_population_and_negative_costs():
    oracle = StubPredicateOracle()
    certificate = certify_bbht_uniform_success(8, 0.9)
    with pytest.raises(ValueError, match="input width"):
        compare_unknown_k_search_to_specialized_classical(
            oracle,
            certificate,
            SpecializedClassicalSolverCosts(per_instance_cost=1.0),
            _costs(),
        )
    with pytest.raises(ValueError):
        SpecializedClassicalSolverCosts(per_instance_cost=-1.0)
    with pytest.raises(ValueError):
        UnknownKQuantumSearchCosts(measured_candidate_verification_cost=-1.0)
