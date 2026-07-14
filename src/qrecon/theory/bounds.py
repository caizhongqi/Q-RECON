from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Hashable, Mapping

import numpy as np


ProbabilityMap = Mapping[Hashable, float]
DeterministicObservation = Mapping[Hashable, Hashable]
ObservationChannel = Mapping[Hashable, Mapping[Hashable, float]]


def _normalized_prior(prior: ProbabilityMap) -> dict[Hashable, float]:
    if not prior:
        raise ValueError("prior must contain at least one candidate")
    normalized: dict[Hashable, float] = {}
    total = 0.0
    for candidate, probability in prior.items():
        value = float(probability)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("prior probabilities must be finite and non-negative")
        normalized[candidate] = value
        total += value
    if total <= 0.0:
        raise ValueError("prior must have positive total mass")
    return {candidate: probability / total for candidate, probability in normalized.items()}


def _validated_channel(
    candidates: set[Hashable], channel: ObservationChannel, *, atol: float = 1e-12
) -> dict[Hashable, dict[Hashable, float]]:
    missing = candidates.difference(channel)
    if missing:
        raise KeyError(f"channel is missing candidates: {sorted(map(str, missing))}")

    validated: dict[Hashable, dict[Hashable, float]] = {}
    for candidate in candidates:
        row = channel[candidate]
        if not row:
            raise ValueError(f"channel row for {candidate!r} is empty")
        converted: dict[Hashable, float] = {}
        total = 0.0
        for observation, probability in row.items():
            value = float(probability)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("channel probabilities must be finite and non-negative")
            converted[observation] = value
            total += value
        if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=atol):
            raise ValueError(
                f"channel row for {candidate!r} must sum to one; observed {total}"
            )
        validated[candidate] = converted
    return validated


def observation_fibres(observation: DeterministicObservation) -> dict[Hashable, tuple[Hashable, ...]]:
    """Partition candidates into fibres of a deterministic observation map."""

    fibres: dict[Hashable, list[Hashable]] = defaultdict(list)
    for candidate, value in observation.items():
        fibres[value].append(candidate)
    return {value: tuple(candidates) for value, candidates in fibres.items()}


def bayes_reconstruction_success(
    prior: ProbabilityMap, observation: DeterministicObservation
) -> float:
    """Return the Bayes-optimal exact-reconstruction probability.

    For candidate ``x`` with prior ``pi(x)`` and deterministic observation
    ``g(x)``, every estimator is constant on each observation fibre. The optimal
    estimator therefore chooses a maximum-posterior candidate in every fibre,
    giving ``sum_y max_{x:g(x)=y} pi(x)``.

    The prior is normalized internally. Extra entries in ``observation`` are
    ignored, while a missing observation for a prior-supported candidate is an
    error.
    """

    normalized = _normalized_prior(prior)
    missing = set(normalized).difference(observation)
    if missing:
        raise KeyError(f"observation is missing candidates: {sorted(map(str, missing))}")

    best_mass: dict[Hashable, float] = {}
    for candidate, probability in normalized.items():
        value = observation[candidate]
        best_mass[value] = max(best_mass.get(value, 0.0), probability)
    return float(sum(best_mass.values()))


def uniform_fibre_success(observation: DeterministicObservation) -> float:
    """Bayes success for a uniform prior: number of fibres divided by candidates."""

    if not observation:
        raise ValueError("observation must contain at least one candidate")
    return len(set(observation.values())) / len(observation)


def channel_bayes_reconstruction_success(
    prior: ProbabilityMap, channel: ObservationChannel
) -> float:
    """Bayes-optimal exact-reconstruction probability for a noisy channel.

    If ``W(y|x)`` is the observation channel, the exact optimum over all
    estimators is ``sum_y max_x pi(x) W(y|x)``.
    """

    normalized = _normalized_prior(prior)
    validated = _validated_channel(set(normalized), channel)
    observations = {
        observation
        for candidate in normalized
        for observation in validated[candidate]
    }
    success = 0.0
    for observation in observations:
        success += max(
            normalized[candidate] * validated[candidate].get(observation, 0.0)
            for candidate in normalized
        )
    return float(success)


def postprocess_channel(
    channel: ObservationChannel,
    kernel: Mapping[Hashable, Mapping[Hashable, float]],
) -> dict[Hashable, dict[Hashable, float]]:
    """Compose an observation channel with a stochastic post-processing kernel.

    This helper makes the data-processing inequality executable: Bayes recovery
    after calling this function cannot exceed recovery from the input channel.
    """

    candidates = set(channel)
    validated_channel = _validated_channel(candidates, channel)
    intermediate = {value for row in validated_channel.values() for value in row}
    validated_kernel = _validated_channel(intermediate, kernel)

    result: dict[Hashable, dict[Hashable, float]] = {}
    for candidate, row in validated_channel.items():
        output: dict[Hashable, float] = defaultdict(float)
        for middle, first_probability in row.items():
            for final, second_probability in validated_kernel[middle].items():
                output[final] += first_probability * second_probability
        result[candidate] = dict(output)
    return result


def conditional_min_entropy_bits(guessing_probability: float) -> float:
    """Classical conditional min-entropy ``-log2(P_guess)`` in bits."""

    value = float(guessing_probability)
    if not math.isfinite(value) or value <= 0.0 or value > 1.0:
        raise ValueError("guessing_probability must lie in (0, 1]")
    return -math.log2(value)


def binary_helstrom_success(
    rho0: np.ndarray,
    rho1: np.ndarray,
    prior0: float = 0.5,
    *,
    atol: float = 1e-9,
) -> float:
    """Optimal success for discriminating two finite-dimensional quantum states.

    The returned value is
    ``(1 + ||p0*rho0 - p1*rho1||_1) / 2``. Inputs are validated as density
    matrices up to ``atol``. This is an information bound, not an implementation
    of the optimal measurement.
    """

    p0 = float(prior0)
    if not math.isfinite(p0) or p0 < 0.0 or p0 > 1.0:
        raise ValueError("prior0 must lie in [0, 1]")
    left = np.asarray(rho0, dtype=np.complex128)
    right = np.asarray(rho1, dtype=np.complex128)
    if left.ndim != 2 or left.shape[0] != left.shape[1] or left.shape != right.shape:
        raise ValueError("rho0 and rho1 must be square matrices of equal shape")

    for name, matrix in (("rho0", left), ("rho1", right)):
        if not np.allclose(matrix, matrix.conj().T, atol=atol, rtol=0.0):
            raise ValueError(f"{name} must be Hermitian")
        if not np.isclose(np.trace(matrix), 1.0, atol=atol, rtol=0.0):
            raise ValueError(f"{name} must have unit trace")
        if float(np.linalg.eigvalsh(matrix).min()) < -atol:
            raise ValueError(f"{name} must be positive semidefinite")

    delta = p0 * left - (1.0 - p0) * right
    trace_norm = float(np.abs(np.linalg.eigvalsh(delta)).sum())
    success = 0.5 * (1.0 + trace_norm)
    return min(1.0, max(0.0, success))


def all_pairs_epsilon_private_uniform_bound(population: int, epsilon: float) -> float:
    """Upper bound on exact guessing under all-pairs pure epsilon privacy.

    Assume a uniform prior on ``population`` candidates and a channel satisfying
    ``W(y|x) <= exp(epsilon) W(y|x')`` for every pair of candidates. Then every
    posterior maximum, and hence the total Bayes success, is at most
    ``exp(epsilon) / (exp(epsilon) + population - 1)``.

    This strong all-pairs assumption is *not* the usual neighbouring-dataset DP
    relation; callers must not apply the bound to a sparse adjacency graph.
    """

    if population <= 0:
        raise ValueError("population must be positive")
    value = float(epsilon)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("epsilon must be finite and non-negative")
    if population == 1:
        return 1.0
    return 1.0 / (1.0 + (population - 1) * math.exp(-value))
