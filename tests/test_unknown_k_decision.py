import pytest

from qrecon.theory.unknown_k_decision import (
    certify_bbht_existence_decision,
    evaluate_bbht_existence_decision,
)


def test_bbht_existence_certificate_covers_zero_and_every_positive_k():
    certificate = certify_bbht_existence_decision(
        16,
        0.9,
        max_exact_population=32,
    )

    assert certificate.minimum_positive_marked == 1
    assert certificate.zero_case_success == 1.0
    assert certificate.certified_positive_detection >= 0.9
    assert certificate.certified_worst_case_decision_success >= 0.9
    assert certificate.false_empty_upper_bound <= 0.1 + 1e-12

    zero = evaluate_bbht_existence_decision(certificate, 0)
    assert zero.correct_decision_probability == 1.0
    assert zero.false_empty_probability == 0.0
    assert zero.false_present_probability == 0.0
    assert zero.covered_by_certificate
    assert zero.search_evaluation.achieved_success == 0.0
    assert zero.search_evaluation.expected_verification_queries == pytest.approx(
        certificate.schedule.rounds
    )
    assert zero.expected_total_oracle_calls == pytest.approx(
        certificate.zero_case_expected_total_oracle_calls
    )

    for marked in range(1, 17):
        evaluation = evaluate_bbht_existence_decision(certificate, marked)
        assert evaluation.covered_by_certificate
        assert evaluation.correct_decision_probability >= 0.9 - 1e-12
        assert evaluation.false_empty_probability <= 0.1 + 1e-12
        assert evaluation.false_present_probability == 0.0


def test_bbht_existence_certificate_exposes_promised_positive_gap():
    certificate = certify_bbht_existence_decision(
        16,
        0.8,
        minimum_positive_marked=3,
        max_exact_population=32,
    )
    assert evaluate_bbht_existence_decision(certificate, 0).covered_by_certificate
    assert not evaluate_bbht_existence_decision(certificate, 1).covered_by_certificate
    assert not evaluate_bbht_existence_decision(certificate, 2).covered_by_certificate
    assert evaluate_bbht_existence_decision(certificate, 3).covered_by_certificate


def test_bbht_existence_certificate_rejects_invalid_population_bound():
    with pytest.raises(ValueError):
        certify_bbht_existence_decision(
            32,
            0.9,
            max_exact_population=16,
        )
