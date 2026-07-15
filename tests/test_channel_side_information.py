import math

import pytest

from qrecon.theory.channel_side_information import (
    channel_side_information_bound,
    channel_side_information_phase_diagram,
)


def test_private_distinct_seven_channel_labels_leave_full_factorial_orbit():
    signatures = tuple(f"private-{index}" for index in range(7))
    bound = channel_side_information_bound(signatures, ("private",) * 7)

    assert bound.public_classes == 1
    assert bound.public_group_sizes == (7,)
    assert bound.within_group_signature_multiplicities == ((1, 1, 1, 1, 1, 1, 1),)
    assert bound.residual_orbit_size == math.factorial(7)
    assert bound.uniform_exact_ordered_recovery_ceiling == pytest.approx(1 / 5040)
    assert bound.ambiguity_bits == pytest.approx(math.log2(5040))
    assert not bound.permutation_ambiguity_eliminated


def test_public_channel_families_leave_only_within_family_permutations():
    signatures = tuple(f"private-{index}" for index in range(7))
    labels = ("high", "high", "medium", "medium", "low", "low", "target")
    bound = channel_side_information_bound(signatures, labels)

    assert bound.public_group_sizes == (2, 2, 2, 1)
    assert bound.residual_orbit_size == 8
    assert bound.uniform_exact_ordered_recovery_ceiling == pytest.approx(1 / 8)


def test_unique_public_labels_eliminate_only_the_permutation_ambiguity():
    signatures = tuple(f"private-{index}" for index in range(7))
    labels = tuple(f"semantic-{index}" for index in range(7))
    bound = channel_side_information_bound(signatures, labels)

    assert bound.residual_orbit_size == 1
    assert bound.uniform_exact_ordered_recovery_ceiling == 1.0
    assert bound.permutation_ambiguity_eliminated
    assert bound.uniform_recovery_modulo_residual_permutation_ceiling == 1.0


def test_duplicate_private_signatures_reduce_the_residual_orbit():
    bound = channel_side_information_bound(
        ("same", "same", "different"),
        ("private", "private", "private"),
    )
    assert bound.within_group_signature_multiplicities == ((2, 1),)
    assert bound.residual_orbit_size == 3
    assert bound.uniform_exact_ordered_recovery_ceiling == pytest.approx(1 / 3)


def test_phase_diagram_is_named_and_machine_readable():
    signatures = ("a", "b", "c")
    result = channel_side_information_phase_diagram(
        signatures,
        {
            "private": (0, 0, 0),
            "public": (0, 1, 2),
        },
    )
    assert tuple(result) == ("private", "public")
    assert result["private"].residual_orbit_size == 6
    assert result["public"].residual_orbit_size == 1
    assert result["private"].to_dict()["public_group_sizes"] == [3]


@pytest.mark.parametrize(
    "signatures, labels, message",
    [
        ((), (), "non-empty"),
        (("a", "b"), ("only-one",), "one public label"),
    ],
)
def test_invalid_side_information_contracts_are_rejected(signatures, labels, message):
    with pytest.raises(ValueError, match=message):
        channel_side_information_bound(signatures, labels)


def test_empty_or_unnamed_phase_diagram_is_rejected():
    with pytest.raises(ValueError, match="at least one"):
        channel_side_information_phase_diagram(("a",), {})
    with pytest.raises(ValueError, match="non-empty"):
        channel_side_information_phase_diagram(("a",), {" ": (0,)})
