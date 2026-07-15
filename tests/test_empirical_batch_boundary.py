import math

import pytest

from qrecon.theory.empirical_batch_boundary import batch_two_recovery_dichotomy


def test_unique_two_record_target_has_matching_linear_exponents():
    report = batch_two_recovery_dichotomy(100, 1)
    assert report.unordered_pair_population == math.comb(100, 2)
    assert report.classical_hash_index_operations == 200
    assert report.ideal_grover_verifier_call_scale == pytest.approx(math.sqrt(4950))
    assert report.exact_original_pair_identifiable
    assert not report.identifiable_query_exponent_separation
    assert "no_exponent_separation" in report.verdict


def test_nonunique_two_record_target_is_information_limited():
    report = batch_two_recovery_dichotomy(10, 4)
    assert not report.exact_original_pair_identifiable
    assert report.target_conditional_exact_index_bayes_success_uniform == pytest.approx(0.25)
    assert "information_limited" in report.verdict


def test_invalid_two_record_boundary_parameters_are_rejected():
    with pytest.raises(ValueError):
        batch_two_recovery_dichotomy(1, 1)
    with pytest.raises(ValueError):
        batch_two_recovery_dichotomy(3, 0)
    with pytest.raises(ValueError):
        batch_two_recovery_dichotomy(3, 4)
