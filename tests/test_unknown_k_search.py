import math

import pytest

from qrecon.theory import (
    BBHTSchedule,
    build_bbht_schedule,
    certify_bbht_uniform_success,
    evaluate_bbht_schedule,
    grover_success,
    randomized_grover_round_success,
)


def test_randomized_round_matches_explicit_grover_average():
    population = 16
    for marked in (1, 3, 8, 15, 16):
        theta = math.asin(math.sqrt(marked / population))
        for window in (1, 2, 3, 4):
            explicit = sum(
                math.sin((2 * iterations + 1) * theta) ** 2
                for iterations in range(window)
            ) / window
            assert randomized_grover_round_success(
                population, marked, window
            ) == pytest.approx(explicit, abs=1e-12)


def test_schedule_is_independent_of_marked_count_and_respects_cap():
    schedule = build_bbht_schedule(64, 20)
    assert schedule.windows == build_bbht_schedule(64, 20).windows
    assert schedule.windows[0] == 1
    assert tuple(sorted(schedule.windows)) == schedule.windows
    assert max(schedule.windows) <= math.ceil(math.sqrt(schedule.population))
    assert schedule.worst_case_phase_oracle_calls == sum(
        window - 1 for window in schedule.windows
    )
    assert schedule.worst_case_total_oracle_calls == (
        schedule.worst_case_phase_oracle_calls + schedule.rounds
    )


def test_exact_schedule_evaluation_counts_reach_and_queries():
    schedule = BBHTSchedule(4, 8.0 / 7.0, (1, 2))
    evaluation = evaluate_bbht_schedule(schedule, marked=1)

    # First round succeeds with 1/4.  In the second round, the randomized choice
    # averages zero- and one-iteration Grover success: (1/4 + 1) / 2 = 5/8.
    expected_success = 1.0 - (1.0 - 0.25) * (1.0 - 0.625)
    assert evaluation.achieved_success == pytest.approx(expected_success)
    assert evaluation.expected_phase_oracle_calls == pytest.approx(0.75 * 0.5)
    assert evaluation.expected_verification_queries == pytest.approx(1.0 + 0.75)
    assert evaluation.expected_total_oracle_calls == pytest.approx(2.125)
    assert evaluation.rounds[-1].cumulative_success == pytest.approx(expected_success)


def test_uniform_certificate_covers_every_positive_marked_count():
    certificate = certify_bbht_uniform_success(16, 0.9)
    assert certificate.schedule.rounds > 0
    assert certificate.certified_minimum_success >= 0.9
    assert certificate.maximum_expected_total_oracle_calls <= (
        certificate.schedule.worst_case_total_oracle_calls
    )

    achieved = []
    for marked in range(1, 17):
        evaluation = evaluate_bbht_schedule(certificate.schedule, marked)
        achieved.append(evaluation.achieved_success)
        assert evaluation.achieved_success + 1e-12 >= 0.9
    assert min(achieved) == pytest.approx(certificate.certified_minimum_success)


def test_full_marked_space_succeeds_without_phase_iterations():
    schedule = build_bbht_schedule(32, 1)
    evaluation = evaluate_bbht_schedule(schedule, marked=32)
    assert evaluation.achieved_success == 1.0
    assert evaluation.expected_phase_oracle_calls == 0.0
    assert evaluation.expected_verification_queries == 1.0


def test_bbht_contract_rejects_invalid_or_intractable_certificates():
    with pytest.raises(ValueError):
        build_bbht_schedule(16, 4, growth_factor=1.0)
    with pytest.raises(ValueError):
        build_bbht_schedule(16, -1)
    with pytest.raises(ValueError):
        randomized_grover_round_success(16, 17, 2)
    with pytest.raises(ValueError):
        evaluate_bbht_schedule(build_bbht_schedule(16, 2), marked=-1)
    with pytest.raises(ValueError):
        certify_bbht_uniform_success(16, 1.0)
    with pytest.raises(ValueError):
        certify_bbht_uniform_success(32, 0.9, max_exact_population=16)


def test_randomized_window_is_not_confused_with_known_k_optimum():
    population = 64
    marked = 1
    schedule = build_bbht_schedule(population, 12)
    evaluation = evaluate_bbht_schedule(schedule, marked)

    # The schedule never reads K, whereas this comparison curve does.  Both are
    # retained so reports cannot silently price a known-K Grover plan as an
    # unknown-K implementation.
    best_known_k = max(
        grover_success(population, marked, iterations)
        for iterations in range(math.ceil(math.sqrt(population)) + 1)
    )
    assert evaluation.achieved_success > 0.0
    assert best_known_k > 0.9
