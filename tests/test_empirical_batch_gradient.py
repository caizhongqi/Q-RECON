import math

import numpy as np
import pytest

from qrecon.benchmarks.empirical_batch_gradient import (
    run_real_batch_gradient_phase_diagram,
    solve_vector_two_sum,
)
from qrecon.benchmarks.real_candidate_manifest import (
    CandidateQuantizationSpec,
    RealBatchGradientManifest,
)


def _manifest(quantization, *, target_pair=(0, 1)):
    return RealBatchGradientManifest(
        dataset="synthetic_forecasting",
        candidate_count=4,
        feature_count=1,
        feature_selection="prefix",
        target_coordinate=0,
        target_batch_indices=target_pair,
        quantizations=(quantization,),
        model_weights=(1,),
        model_bias=0,
        gradient_bits=8,
        target_success=0.75,
        bbht_attempts_per_stage=2,
        bbht_max_stages=64,
        max_exact_population=64,
        max_exact_batches=64,
        max_basis_verification_bits=4,
        max_minterm_table_bits=1024,
        reusable_instances=(1, 4),
    )


def test_vector_two_sum_returns_complete_unordered_fibre():
    contributions = ((0, 0), (1, 1), (4, 2), (9, 3))
    report = solve_vector_two_sum(contributions, (5, 3))
    assert report.solutions == ((1, 2),)
    assert report.solution_count == 1
    assert report.elementary_index_operations == 8
    assert report.complete


def test_unique_real_candidate_pair_has_no_query_exponent_separation():
    manifest = _manifest(CandidateQuantizationSpec(4, 0, True, "raise"))
    inputs = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    targets = np.zeros((4, 1), dtype=np.float32)
    report = run_real_batch_gradient_phase_diagram(
        manifest,
        loader_overrides={"synthetic_forecasting": lambda _: (inputs, targets)},
    )
    assert report.complete
    point = report.points[0]
    assert point.unordered_batch_count == math.comb(4, 2)
    assert point.padded_pair_population == 8
    assert point.target_fibre_pairs == ((0, 1),)
    assert point.target_uniquely_identifiable
    assert point.global_exact_batch_bayes_success_uniform == 1.0
    assert point.two_sum_matches_exhaustive_fibre
    assert point.predicate_reference_matches_fibre
    assert point.predicate_reference_basis_verified
    assert point.bbht_certificate is not None
    assert point.bbht_target_evaluation is not None
    assert point.record_table_one_shot_boundary is not None
    assert len(point.record_loading_amortization) == 2
    assert "no_query_exponent_separation" in point.asymptotic_verdict
    assert point.semantic_cross_checks_passed


def test_quantization_collision_creates_information_limited_pair_fibre():
    manifest = _manifest(
        CandidateQuantizationSpec(4, 0, True, "raise"),
        target_pair=(0, 2),
    )
    inputs = np.array([[0.1], [0.2], [1.0], [2.0]], dtype=np.float32)
    targets = np.zeros((4, 1), dtype=np.float32)
    report = run_real_batch_gradient_phase_diagram(
        manifest,
        loader_overrides={"synthetic_forecasting": lambda _: (inputs, targets)},
    )
    point = report.points[0]
    assert point.feature_quantization.quantization_induced_collision_count == 1
    assert point.record_loading.duplicate_candidate_count == 1
    assert point.target_fibre_pairs == ((0, 2), (1, 2))
    assert point.target_fibre_size == 2
    assert not point.target_uniquely_identifiable
    assert point.target_conditional_exact_batch_success_uniform == pytest.approx(0.5)
    assert point.two_sum_matches_exhaustive_fibre
    assert "information_limited_exact_batch" in point.asymptotic_verdict


def test_failed_precision_is_preserved_in_phase_diagram():
    manifest = _manifest(CandidateQuantizationSpec(2, 0, True, "raise"))
    inputs = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    targets = np.zeros((4, 1), dtype=np.float32)
    report = run_real_batch_gradient_phase_diagram(
        manifest,
        loader_overrides={"synthetic_forecasting": lambda _: (inputs, targets)},
    )
    assert not report.complete
    assert not report.points
    assert len(report.failures) == 1
    assert report.failures[0].error_type == "OverflowError"
    assert len(report.failures[0].error_sha256) == 64


def test_classical_only_contract_skips_quantum_search_and_loading_claims():
    base = _manifest(CandidateQuantizationSpec(4, 0, True, "raise"))
    payload = base.to_dict()
    payload["access_contract"] = "classical_only"
    manifest = RealBatchGradientManifest.from_dict(payload)
    inputs = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    targets = np.zeros((4, 1), dtype=np.float32)
    point = run_real_batch_gradient_phase_diagram(
        manifest,
        loader_overrides={"synthetic_forecasting": lambda _: (inputs, targets)},
    ).points[0]
    assert point.bbht_certificate is None
    assert point.bbht_target_evaluation is None
    assert point.record_table_one_shot_boundary is None
    assert point.record_loading_amortization == ()
