import pytest

from qrecon.theory.unknown_k import evaluate_bbht_schedule
from qrecon.theory.unknown_k_staged import certify_staged_bbht_uniform_success


def test_staged_bbht_repeats_each_window_and_certifies_all_positive_k():
    certificate = certify_staged_bbht_uniform_success(
        16,
        0.9,
        attempts_per_stage=2,
        max_stages=32,
        max_exact_population=32,
    )
    windows = certificate.schedule.windows
    assert len(windows) % 2 == 0
    assert all(
        windows[index] == windows[index + 1]
        for index in range(0, len(windows), 2)
    )
    for marked in range(1, 17):
        assert (
            evaluate_bbht_schedule(certificate.schedule, marked).achieved_success
            >= 0.9 - 1e-12
        )


def test_one_attempt_staged_certificate_matches_public_contract():
    certificate = certify_staged_bbht_uniform_success(
        8,
        0.8,
        attempts_per_stage=1,
        max_stages=32,
        max_exact_population=32,
    )
    assert certificate.certified_minimum_success >= 0.8
    assert certificate.schedule.rounds <= 32


def test_staged_bbht_rejects_invalid_or_insufficient_contracts():
    with pytest.raises(ValueError):
        certify_staged_bbht_uniform_success(16, 0.9, attempts_per_stage=0)
    with pytest.raises(ValueError):
        certify_staged_bbht_uniform_success(16, 0.9, max_stages=0)
    with pytest.raises(ValueError):
        certify_staged_bbht_uniform_success(
            16,
            0.999999,
            max_stages=1,
            max_exact_population=32,
        )
    with pytest.raises(ValueError):
        certify_staged_bbht_uniform_success(
            64,
            0.9,
            max_exact_population=32,
        )
