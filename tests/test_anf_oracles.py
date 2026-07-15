import math

import pytest

from qrecon.oracles import (
    ANFOracle,
    TruthTableOracle,
    compare_exact_syntheses,
    simulate_grover,
)


def test_anf_mobius_transform_matches_reference_for_multibit_function():
    reference = TruthTableOracle.from_function(
        3, 2, lambda x: ((x.bit_count() & 1) << 0) | (((x >> 1) & 1) << 1)
    )
    anf = ANFOracle.from_truth_table(reference)
    assert anf.verify_reference_equivalence()
    assert anf.verify_basis_permutation()
    for x in range(8):
        for y in range(4):
            assert anf.apply(x, y) == (x, y ^ reference.table[x], 0)


def test_anf_eliminates_toffolis_for_affine_boolean_parity():
    parity = TruthTableOracle.from_function(5, 1, lambda x: x.bit_count() & 1)
    anf = ANFOracle.from_truth_table(parity)
    resources = anf.resource_estimate(phase_kickback=True)
    assert resources.controlled_x_terms == 5
    assert resources.cnot_gates == 5
    assert resources.toffoli_gates == 0
    comparison = compare_exact_syntheses(parity)
    assert comparison.selected == "anf"
    assert comparison.anf.t_count_upper_bound < comparison.minterm.t_count_upper_bound


def test_anf_phase_oracle_drives_same_grover_curve():
    verifier = TruthTableOracle.from_function(3, 1, lambda x: int(x == 6))
    anf = ANFOracle.from_truth_table(verifier)
    result = simulate_grover(anf, 2)
    theta = math.asin(math.sqrt(1 / 8))
    assert result.success_probability == pytest.approx(math.sin(5 * theta) ** 2)
    assert result.most_likely_inputs == (6,)
