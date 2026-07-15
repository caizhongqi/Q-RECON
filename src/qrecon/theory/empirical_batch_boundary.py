from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class BatchTwoRecoveryDichotomy:
    """Information/query boundary for additive two-record reconstruction.

    ``record_candidates`` is the number of local records and ``target_fibre_size``
    is the number of unordered record pairs producing the released additive
    observation. The classical upper bound is the complete vector two-sum hash
    algorithm: index every local contribution and perform one complement lookup
    per record. The Grover scale is the square root of the unordered pair
    population divided by the marked count.
    """

    record_candidates: int
    target_fibre_size: int
    unordered_pair_population: int
    target_conditional_exact_index_bayes_success_uniform: float
    classical_hash_index_operations: int
    classical_expected_time_scale: str
    classical_memory_scale: str
    ideal_grover_verifier_call_scale: float
    exact_original_pair_identifiable: bool
    identifiable_query_exponent_separation: bool
    verdict: str

    def to_dict(self) -> dict[str, int | float | bool | str]:
        return asdict(self)


def batch_two_recovery_dichotomy(
    record_candidates: int,
    target_fibre_size: int,
) -> BatchTwoRecoveryDichotomy:
    """Return the information/algorithmic dichotomy for batch size two.

    For additive leakage ``g(i,j)=h(i)+h(j)`` over unordered pairs, a hash index
    over ``h(0),...,h(N-1)`` finds the complete target fibre in expected
    ``O(N+K)`` time and ``O(N)`` memory. If ``K>1``, exact original-pair recovery
    is information-theoretically ambiguous under a uniform prior. If ``K=1``,
    unstructured Grover search over ``binom(N,2)`` pairs uses ``Theta(N)``
    verifier calls, matching the classical linear exponent rather than improving
    it.
    """

    candidates = int(record_candidates)
    marked = int(target_fibre_size)
    if candidates < 2:
        raise ValueError("record_candidates must be at least two")
    population = math.comb(candidates, 2)
    if marked <= 0 or marked > population:
        raise ValueError("target_fibre_size must lie in [1, choose(N, 2)]")

    identifiable = marked == 1
    if identifiable:
        verdict = (
            "unique_target_no_exponent_separation: complete vector two-sum uses "
            "O(N) expected time and O(N) memory, while ideal Grover search over "
            "the unordered pair domain uses Theta(N) verifier calls"
        )
    else:
        verdict = (
            "nonunique_target_information_limited: the released observation has "
            f"{marked} compatible unordered pairs, so conditional uniform-prior "
            f"exact-index success is at most 1/{marked}"
        )

    return BatchTwoRecoveryDichotomy(
        record_candidates=candidates,
        target_fibre_size=marked,
        unordered_pair_population=population,
        target_conditional_exact_index_bayes_success_uniform=1.0 / marked,
        classical_hash_index_operations=2 * candidates,
        classical_expected_time_scale="O(N + K)",
        classical_memory_scale="O(N)",
        ideal_grover_verifier_call_scale=math.sqrt(population / marked),
        exact_original_pair_identifiable=identifiable,
        identifiable_query_exponent_separation=False,
        verdict=verdict,
    )
