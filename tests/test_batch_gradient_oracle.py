import math

import pytest

from qrecon.oracles import (
    ReversibleBatchGradientEqualityOracle,
    ReversibleBatchGradientValueOracle,
    analyze_finite_oracle,
    run_batch_gradient_reconstruction,
)
from qrecon.theory import optimal_standard_grover_iterations


def _public_target_oracle() -> ReversibleBatchGradientValueOracle:
    return ReversibleBatchGradientValueOracle(
        (1,),
        0,
        batch_size=2,
        input_bits=2,
        gradient_bits=5,
        public_targets=(0, 1),
    )


def test_public_target_aggregate_gradient_is_globally_injective_on_small_domain():
    value = _public_target_oracle()
    assert value.range_report.no_overflow
    assert value.verify_basis_permutation()
    reference = value.compile_reference_oracle()
    report = analyze_finite_oracle(reference)
    assert report.injective
    assert report.population == 16
    assert report.uniform_exact_success == 1.0

    for candidate in range(value.population):
        inputs, targets = value.decode_candidate(candidate)
        assert targets == (0, 1)
        assert value.encode_candidate(inputs) == candidate
        expected = reference.table[candidate]
        assert value.apply(candidate, 0) == (candidate, expected, 0)
        assert value.inverse_apply(candidate, expected, 0) == (candidate, 0, 0)


def test_public_target_batch_gradient_equality_oracle_is_clean_and_unique():
    value = _public_target_oracle()
    candidate = value.encode_candidate(((1,), (-2,)))
    observed = value.evaluate_input_word(candidate)
    verifier = ReversibleBatchGradientEqualityOracle(value, observed)
    assert verifier.verify_basis_permutation()
    assert verifier.marked_inputs() == (candidate,)
    assert verifier.apply(candidate, 0) == (candidate, 1, 0)
    assert verifier.apply(candidate, 1) == (candidate, 0, 0)
    assert verifier.phase_sign(candidate) == -1

    report = run_batch_gradient_reconstruction(value, ((1,), (-2,)))
    assert report["exact_original_identifiable"]
    assert report["marked_candidates"] == [candidate]
    iterations = optimal_standard_grover_iterations(value.population, 1)
    assert iterations is not None
    theta = math.asin(math.sqrt(1 / value.population))
    assert report["grover_success_probability"] == pytest.approx(
        math.sin((2 * iterations + 1) * theta) ** 2
    )


def test_private_target_aggregate_gradient_has_nontrivial_collision_fibres():
    value = ReversibleBatchGradientValueOracle(
        (1,),
        0,
        batch_size=2,
        input_bits=2,
        gradient_bits=5,
    )
    reference = value.compile_reference_oracle()
    report = analyze_finite_oracle(reference)
    assert not report.injective
    assert report.population == 256
    assert report.distinct_observations < report.population
    assert report.largest_fibre > 1
    assert report.uniform_exact_success < 1.0

    candidate = value.encode_candidate(((1,), (-1,)), (0, 1))
    inputs, targets = value.decode_candidate(candidate)
    assert inputs == ((1,), (-1,))
    assert targets == (0, 1)
    observed = value.evaluate_input_word(candidate)
    verifier = ReversibleBatchGradientEqualityOracle(value, observed)
    assert len(verifier.marked_inputs()) > 1


def test_batch_compiler_reuses_one_record_work_region_and_reports_resources():
    value = _public_target_oracle()
    assert len(value.layout.record_work) == len(value.record_oracle.layout.work_wires)
    assert len(value.layout.public_target_register) == value.input_bits_per_word
    resources = value.resource_estimate()
    assert resources.input_qubits == value.input_bits
    assert resources.output_qubits == value.output_bits
    assert resources.peak_clean_ancillas == len(value.layout.work_wires)
    assert resources.logical_qubits == value.circuit.num_qubits
    assert resources.toffoli_gates > 0
    assert resources.t_count_upper_bound == 7 * resources.toffoli_gates


def test_batch_range_contract_rejects_underwidth_aggregate_words():
    with pytest.raises(OverflowError):
        ReversibleBatchGradientValueOracle(
            (1,),
            0,
            batch_size=2,
            input_bits=2,
            gradient_bits=4,
            public_targets=(0, 1),
        )
