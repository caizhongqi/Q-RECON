from __future__ import annotations

import math
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass


def _validate_population(marked: int, population: int) -> None:
    if population <= 0:
        raise ValueError("population must be positive")
    if marked < 0 or marked > population:
        raise ValueError("marked must lie in [0, population]")


def bayes_reconstruction_success(
    prior: Mapping[Hashable, float], observation: Mapping[Hashable, Hashable]
) -> float:
    """Bayes-optimal exact reconstruction success from a deterministic observation.

    For a candidate x with prior pi(x) and deterministic observation g(x), the
    optimal estimator chooses the most likely candidate in each observation
    fibre. Its success probability