from qrecon.oracles import ReversibleBatchGradientValueOracle
from qrecon.oracles.batch_baselines import (
    balanced_mitm_partial_state_count,
    batch_gradient_contribution_tables,
    ideal_unstructured_search_scale,
    meet_in_the_middle_additive_solutions,
    solve_batch_gradient_meet_in_the_middle,
)


def _public_oracle() -> ReversibleBatchGradientValueOracle:
    return ReversibleBatchGradientValueOracle(
        (1,),
        0,
        batch_size=2,
        input_bits=2,
        gradient_bits=5,
        public_targets=(0, 1),
    )


def test_public_target_mitm_recovers_the_unique_batch_with_eight_partial_states():
    value = _public_oracle()
    candidate = value.encode_candidate(((1,), (-2,)))
    observed = value.evaluate_input_word(candidate)
    report = solve_batch_gradient_meet_in_the_middle(value, observed)
    assert report.exact_original_identifiable
    assert report.candidate_words == (candidate,)
    assert report.mitm.solution_count == 1
    assert report.mitm.left_states == 4
    assert report.mitm.right_states == 4
    assert report.mitm.enumerated_partial_states == 8
    assert value.population == 16


def test_private_target_mitm_matches_the_complete_brute_force_fibre():
    value = ReversibleBatchGradientValueOracle(
        (1,),
        0,
        batch_size=2,
        input_bits=2,
        gradient_bits=5,
    )
    candidate = value.encode_candidate(((1,), (-1,)), (0, 1))
    observed = value.evaluate_input_word(candidate)
    brute_fibre = tuple(
        word
        for word in range(value.population)
        if value.evaluate_input_word(word) == observed
    )
    report = solve_batch_gradient_meet_in_the_middle(value, observed)
    assert report.mitm.solution_count == len(brute_fibre)
    assert tuple(sorted(report.candidate_words)) == brute_fibre
    assert not report.exact_original_identifiable
    assert report.mitm.left_states == 16
    assert report.mitm.right_states == 16
    assert report.mitm.enumerated_partial_states == 32
    assert value.population == 256


def test_generic_mitm_matches_three_position_additive_problem():
    tables = (
        ((0, 0), (1, 0), (0, 1)),
        ((0, 0), (2, 0), (0, 2)),
        ((0, 0), (3, 0), (0, 3)),
    )
    target = (3, 2)
    report = meet_in_the_middle_additive_solutions(tables, target)
    brute = []
    for first in range(3):
        for second in range(3):
            for third in range(3):
                vector = tuple(
                    tables[0][first][index]
                    + tables[1][second][index]
                    + tables[2][third][index]
                    for index in range(2)
                )
                if vector == target:
                    brute.append((first, second, third))
    assert tuple(sorted(report.local_solutions)) == tuple(sorted(brute))
    assert report.solution_count == len(brute)
    assert report.left_states == 3
    assert report.right_states == 9
    assert report.enumerated_partial_states == 12


def test_batch_contribution_tables_sum_to_the_public_evaluator():
    value = _public_oracle()
    tables = batch_gradient_contribution_tables(value)
    for candidate in range(value.population):
        inputs, _ = value.decode_candidate(candidate)
        local_words = tuple((row[0] & 0b11) for row in inputs)
        summed = tuple(
            tables[0][local_words[0]][index]
            + tables[1][local_words[1]][index]
            for index in range(value.feature_count + 1)
        )
        assert summed == value.gradient_components(candidate)


def test_balanced_mitm_matches_grover_exponent_for_even_batch_size():
    domain = 8
    batch = 4
    partial_states = balanced_mitm_partial_state_count(domain, batch)
    ideal_quantum_scale = ideal_unstructured_search_scale(domain, batch)
    assert partial_states == 2 * domain ** (batch // 2)
    assert ideal_quantum_scale == domain ** (batch // 2)
    assert partial_states < domain**batch
