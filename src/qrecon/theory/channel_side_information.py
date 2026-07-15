from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Hashable, Mapping, Sequence


@dataclass(frozen=True)
class ChannelSideInformationBound:
    """Residual channel-permutation ambiguity after public side information.

    ``public_labels`` define which channel permutations remain compatible with the
    declared side-information channel. Channels carrying different public labels
    cannot be exchanged. Inside each public-label class, identical complete private
    signatures do not create distinct private objects.
    """

    channels: int
    public_classes: int
    public_group_sizes: tuple[int, ...]
    within_group_signature_multiplicities: tuple[tuple[int, ...], ...]
    residual_orbit_size: int
    uniform_exact_ordered_recovery_ceiling: float
    uniform_recovery_modulo_residual_permutation_ceiling: float
    ambiguity_bits: float

    @property
    def permutation_ambiguity_eliminated(self) -> bool:
        return self.residual_orbit_size == 1

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["public_group_sizes"] = list(self.public_group_sizes)
        payload["within_group_signature_multiplicities"] = [
            list(values) for values in self.within_group_signature_multiplicities
        ]
        payload["permutation_ambiguity_eliminated"] = (
            self.permutation_ambiguity_eliminated
        )
        return payload


def _multiset_permutations(multiplicities: Sequence[int]) -> int:
    counts = tuple(int(value) for value in multiplicities)
    if not counts or any(value <= 0 for value in counts):
        raise ValueError("multiplicities must be non-empty positive integers")
    return math.factorial(sum(counts)) // math.prod(
        math.factorial(value) for value in counts
    )


def channel_side_information_bound(
    channel_signatures: Sequence[Hashable],
    public_labels: Sequence[Hashable],
) -> ChannelSideInformationBound:
    """Compute the exact residual orbit induced by a public-label partition.

    The observation is assumed invariant under every simultaneous channel
    permutation. Public side information restricts the admissible permutations to
    those preserving ``public_labels``. For each public class ``a`` with private
    signature multiplicities ``m_(a,1), ..., m_(a,r)``, its contribution is

    ``n_a! / product_j m_(a,j)!``.

    The residual orbit size is the product across public classes. Under a uniform
    prior on that residual orbit, exact labeled-order recovery is bounded by the
    reciprocal orbit size for both classical and coherent quantum estimators.
    Recovery defined modulo every residual permutation collapses the orbit to one
    declared target class, so the information-theoretic ceiling for that quotient
    target is one; this does not assert computational recoverability of a
    representative.
    """

    signatures = tuple(channel_signatures)
    labels = tuple(public_labels)
    if not signatures:
        raise ValueError("channel_signatures must be non-empty")
    if len(labels) != len(signatures):
        raise ValueError("one public label is required per channel signature")

    grouped: dict[Hashable, list[Hashable]] = {}
    for signature, label in zip(signatures, labels):
        grouped.setdefault(label, []).append(signature)

    canonical_groups: list[tuple[int, tuple[int, ...], int]] = []
    for values in grouped.values():
        multiplicities = tuple(
            sorted(Counter(values).values(), reverse=True)
        )
        group_orbit = _multiset_permutations(multiplicities)
        canonical_groups.append((len(values), multiplicities, group_orbit))

    # Canonicalize output independently of hash-map insertion order and label type.
    canonical_groups.sort(key=lambda item: (item[0], item[1]), reverse=True)
    group_sizes = tuple(item[0] for item in canonical_groups)
    group_multiplicities = tuple(item[1] for item in canonical_groups)
    residual_orbit = math.prod(item[2] for item in canonical_groups)

    return ChannelSideInformationBound(
        channels=len(signatures),
        public_classes=len(canonical_groups),
        public_group_sizes=group_sizes,
        within_group_signature_multiplicities=group_multiplicities,
        residual_orbit_size=residual_orbit,
        uniform_exact_ordered_recovery_ceiling=1.0 / residual_orbit,
        uniform_recovery_modulo_residual_permutation_ceiling=1.0,
        ambiguity_bits=math.log2(residual_orbit),
    )


def channel_side_information_phase_diagram(
    channel_signatures: Sequence[Hashable],
    regimes: Mapping[str, Sequence[Hashable]],
) -> dict[str, ChannelSideInformationBound]:
    """Evaluate several predeclared public-side-information regimes."""

    if not regimes:
        raise ValueError("at least one side-information regime is required")
    result: dict[str, ChannelSideInformationBound] = {}
    for raw_name, labels in regimes.items():
        name = str(raw_name).strip()
        if not name:
            raise ValueError("side-information regime names must be non-empty")
        if name in result:
            raise ValueError("side-information regime names must be unique")
        result[name] = channel_side_information_bound(channel_signatures, labels)
    return result
