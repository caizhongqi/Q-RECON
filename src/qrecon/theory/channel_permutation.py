from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Hashable, Sequence


@dataclass(frozen=True)
class ChannelPermutationFibreBound:
    """Exact orbit size and uniform-prior recovery ceiling for labeled channels.

    ``multiplicities`` groups channels whose complete private records, including
    every released target coordinate, are identical. Permuting within an identical
    group does not create a distinct private object. The number of distinct ordered
    objects in the simultaneous channel-permutation orbit is therefore

    ``channels! / product_j multiplicities[j]!``.
    """

    channels: int
    multiplicities: tuple[int, ...]
    orbit_size: int
    uniform_exact_ordered_recovery_ceiling: float

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["multiplicities"] = list(self.multiplicities)
        return payload


def channel_permutation_orbit_size(multiplicities: Sequence[int]) -> int:
    """Number of distinct permutations of a multiset of channel records."""

    counts = tuple(int(value) for value in multiplicities)
    if not counts or any(value <= 0 for value in counts):
        raise ValueError("multiplicities must be non-empty positive integers")
    channels = sum(counts)
    denominator = math.prod(math.factorial(value) for value in counts)
    return math.factorial(channels) // denominator


def channel_permutation_fibre_bound(
    channel_signatures: Sequence[Hashable],
) -> ChannelPermutationFibreBound:
    """Build the exact simultaneous channel-permutation orbit bound.

    The signatures must encode every private quantity whose channel identity is
    part of the recovery target, normally the concatenated input history and
    forecast target for each variable. Under a uniform prior on the orbit, no
    classical or quantum estimator observing a permutation-invariant channel can
    recover the original labeled ordering with probability above ``1/orbit_size``.
    """

    signatures = tuple(channel_signatures)
    if not signatures:
        raise ValueError("channel_signatures must be non-empty")
    multiplicities = tuple(sorted(Counter(signatures).values(), reverse=True))
    orbit = channel_permutation_orbit_size(multiplicities)
    return ChannelPermutationFibreBound(
        channels=len(signatures),
        multiplicities=multiplicities,
        orbit_size=orbit,
        uniform_exact_ordered_recovery_ceiling=1.0 / orbit,
    )
