from __future__ import annotations

import itertools
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Sequence


@dataclass(frozen=True)
class MeetInTheMiddleResult:
    target: tuple[int, ...]
    split_index: int
    local_domain_sizes: tuple[int, ...]
    left_states: int
    right_states: int
    hash_buckets: int
    solution_count: int
    local_solutions: tuple[tuple[int, ...], ...]
    truncated: bool

    @property
    def enumerated_partial_states(self) -> int:
        return self.left_states + self.right_states

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["target"] = list(self.target)
        result["local_domain_sizes"] = list(self.local_domain_sizes)
        result["local_solutions"] = [list(solution) for solution in self.local_solutions]
        result["enumerated_partial_states"] = self.enumerated_partial_states
        return result


@dataclass(frozen=True)
class BatchGradientMITMReport:
    observed_components: tuple[int, ...]
    candidate_words: tuple[int, ...]
    mitm: MeetInTheMiddleResult

    @property
    def exact_original_identifiable(self) -> bool:
        return self.mitm.solution_count == 1

    def to_dict(self) -> dict[str, object]:
        return {
            "observed_components": list(self.observed_components),
            "candidate_words": list(self.candidate_words),
            "exact_original_identifiable": self.exact_original_identifiable,
            "mitm": self.mitm.to_dict(),
        }


def _vector_add(left: Sequence[int], right: Sequence[int]) -> tuple[int, ...]:
    if len(left) != len(right):
        raise ValueError("vector dimensions differ")
    return tuple(int(a) + int(b) for a, b in zip(left, right))


def _vector_subtract(left: Sequence[int], right: Sequence[int]) -> tuple[int, ...]:
    if len(left) != len(right):
        raise ValueError("vector dimensions differ")
    return tuple(int(a) - int(b) for a, b in zip(left, right))


def _sum_assignment(
    tables: Sequence[Sequence[tuple[int, ...]]], assignment: Sequence[int]
) -> tuple[int, ...]:
    if len(tables) != len(assignment):
        raise ValueError("assignment length does not match contribution tables")
    if not tables:
        return ()
    dimension = len(tables[0][0])
    total = (0,) * dimension
    for table, index in zip(tables, assignment):
        total = _vector_add(total, table[int(index)])
    return total


def meet_in_the_middle_additive_solutions(
    contribution_tables: Sequence[Sequence[Sequence[int]]],
    target: Sequence[int],
    *,
    max_solutions: int | None = None,
) -> MeetInTheMiddleResult:
    """Solve an additive product-space reconstruction problem exactly.

    ``contribution_tables[j][u]`` is the observation contribution of local word
    ``u`` at record position ``j``. The solver returns ordered local assignments
    whose componentwise sum equals ``target``. It enumerates balanced partial sums
    rather than all Cartesian-product candidates.
    """

    tables = tuple(
        tuple(tuple(int(value) for value in contribution) for contribution in table)
        for table in contribution_tables
    )
    target_vector = tuple(int(value) for value in target)
    if not tables:
        raise ValueError("at least one contribution table is required")
    if not target_vector:
        raise ValueError("target must contain at least one component")
    if max_solutions is not None and max_solutions <= 0:
        raise ValueError("max_solutions must be positive when supplied")
    for table in tables:
        if not table:
            raise ValueError("every local contribution table must be non-empty")
        if any(len(item) != len(target_vector) for item in table):
            raise ValueError("all contributions must match the target dimension")

    split = len(tables) // 2
    left_tables = tables[:split]
    right_tables = tables[split:]
    left_domains = tuple(range(len(table)) for table in left_tables)
    right_domains = tuple(range(len(table)) for table in right_tables)

    left_map: dict[tuple[int, ...], list[tuple[int, ...]]] = defaultdict(list)
    left_states = 0
    for assignment in itertools.product(*left_domains):
        total = (0,) * len(target_vector) if not left_tables else _sum_assignment(
            left_tables, assignment
        )
        left_map[total].append(tuple(int(index) for index in assignment))
        left_states += 1

    solutions: list[tuple[int, ...]] = []
    solution_count = 0
    right_states = 0
    for assignment in itertools.product(*right_domains):
        right_total = _sum_assignment(right_tables, assignment)
        complement = _vector_subtract(target_vector, right_total)
        matches = left_map.get(complement, ())
        solution_count += len(matches)
        for left_assignment in matches:
            if max_solutions is None or len(solutions) < max_solutions:
                solutions.append(left_assignment + tuple(int(index) for index in assignment))
        right_states += 1

    return MeetInTheMiddleResult(
        target=target_vector,
        split_index=split,
        local_domain_sizes=tuple(len(table) for table in tables),
        left_states=left_states,
        right_states=right_states,
        hash_buckets=len(left_map),
        solution_count=solution_count,
        local_solutions=tuple(solutions),
        truncated=max_solutions is not None and solution_count > len(solutions),
    )


def balanced_mitm_partial_state_count(local_domain_size: int, batch_size: int) -> int:
    """Number of partial assignments enumerated by a balanced equal-domain split."""

    domain = int(local_domain_size)
    records = int(batch_size)
    if domain <= 0 or records <= 0:
        raise ValueError("local_domain_size and batch_size must be positive")
    left = records // 2
    right = records - left
    return domain**left + domain**right


def ideal_unstructured_search_scale(local_domain_size: int, batch_size: int) -> float:
    """Square root of the full product-space population ``M**B``."""

    domain = int(local_domain_size)
    records = int(batch_size)
    if domain <= 0 or records <= 0:
        raise ValueError("local_domain_size and batch_size must be positive")
    return math.sqrt(domain**records)


def _decode_signed_word(word: int, bits: int) -> int:
    return int(word) - (1 << bits) if int(word) >= (1 << (bits - 1)) else int(word)


def batch_gradient_contribution_tables(
    value_oracle: object,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Build exact per-position contribution tables for a batch-gradient oracle."""

    batch_size = int(getattr(value_oracle, "batch_size"))
    features = int(getattr(value_oracle, "feature_count"))
    bits = int(getattr(value_oracle, "input_bits_per_word"))
    weights = tuple(int(value) for value in getattr(value_oracle, "weights"))
    bias = int(getattr(value_oracle, "bias"))
    public_targets = getattr(value_oracle, "public_targets")
    private_target = public_targets is None
    local_words = 1 << ((features + int(private_target)) * bits)
    mask = (1 << bits) - 1

    tables: list[tuple[tuple[int, ...], ...]] = []
    for position in range(batch_size):
        contributions: list[tuple[int, ...]] = []
        for local_word in range(local_words):
            offset = 0
            inputs: list[int] = []
            for _ in range(features):
                inputs.append(_decode_signed_word((local_word >> offset) & mask, bits))
                offset += bits
            target = (
                _decode_signed_word((local_word >> offset) & mask, bits)
                if private_target
                else int(public_targets[position])
            )
            residual = bias + sum(
                weight * value for weight, value in zip(weights, inputs)
            ) - target
            contributions.append(
                tuple(residual * value for value in inputs) + (residual,)
            )
        tables.append(tuple(contributions))
    return tuple(tables)


def unpack_batch_gradient_observation(
    value_oracle: object, observed_word: int
) -> tuple[int, ...]:
    bits = int(getattr(value_oracle, "gradient_bits"))
    components = int(getattr(value_oracle, "feature_count")) + 1
    word = int(observed_word)
    if word < 0 or word >= (1 << (bits * components)):
        raise ValueError("observed_word is outside the aggregate-gradient register")
    mask = (1 << bits) - 1
    return tuple(
        _decode_signed_word((word >> (index * bits)) & mask, bits)
        for index in range(components)
    )


def solve_batch_gradient_meet_in_the_middle(
    value_oracle: object,
    observed_word: int,
    *,
    max_solutions: int | None = None,
) -> BatchGradientMITMReport:
    """Recover all ordered batches matching an aggregate gradient by MITM."""

    tables = batch_gradient_contribution_tables(value_oracle)
    target = unpack_batch_gradient_observation(value_oracle, observed_word)
    result = meet_in_the_middle_additive_solutions(
        tables, target, max_solutions=max_solutions
    )
    features = int(getattr(value_oracle, "feature_count"))
    bits = int(getattr(value_oracle, "input_bits_per_word"))
    private_target = getattr(value_oracle, "public_targets") is None
    local_bits = (features + int(private_target)) * bits
    candidates = tuple(
        sum(local_word << (position * local_bits) for position, local_word in enumerate(solution))
        for solution in result.local_solutions
    )
    for candidate in candidates:
        if int(getattr(value_oracle, "evaluate_input_word")(candidate)) != int(observed_word):
            raise RuntimeError("MITM solution failed the public aggregate-gradient evaluator")
    return BatchGradientMITMReport(target, candidates, result)
