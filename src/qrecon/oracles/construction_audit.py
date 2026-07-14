from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Hashable, Mapping, Protocol

from .anf import ANFOracle
from .compiler import TruthTableOracle


class StructurePreservingPredicate(Protocol):
    input_bits: int
    output_bits: int

    def resource_estimate(self, *, phase_kickback: bool = False): ...


@dataclass(frozen=True)
class OracleConstructionAudit:
    backend: str
    candidate_bits: int
    candidate_space_size: int
    output_bits: int
    materialized_truth_table_entries: int
    reference_evaluations_to_materialize: int
    artifact_payload_bits_lower_bound: int
    stores_complete_preimage_information: bool
    marked_count: int | None
    unique_answer_recoverable_from_artifact: bool
    ideal_first_peak_grover_iterations: int | None
    enumerative_setup_exceeds_ideal_grover: bool
    controlled_x_terms: int
    toffoli_gates: int
    claim_boundary: str

    def to_dict(self) -> dict[str, int | bool | str | None]:
        return asdict(self)


@dataclass(frozen=True)
class PreimageIndex:
    output_bits: int
    mapping: Mapping[int, tuple[int, ...]]

    def preimages(self, output_word: int) -> tuple[int, ...]:
        value = int(output_word)
        if value < 0 or value >= (1 << self.output_bits):
            raise ValueError("output_word is outside the compiled output register")
        return self.mapping.get(value, ())


def build_truth_table_preimage_index(oracle: TruthTableOracle) -> PreimageIndex:
    """Build the complete classical inverse index exposed by a stored truth table."""

    buckets: dict[int, list[int]] = {}
    for candidate, output in enumerate(oracle.table):
        buckets.setdefault(int(output), []).append(candidate)
    return PreimageIndex(
        output_bits=oracle.output_bits,
        mapping={output: tuple(candidates) for output, candidates in buckets.items()},
    )


def _first_peak_iterations(population: int, marked: int | None) -> int | None:
    if marked is None or marked <= 0:
        return None
    theta = math.asin(math.sqrt(marked / population))
    real_optimum = max(0.0, math.pi / (4.0 * theta) - 0.5)
    candidates = {0, math.floor(real_optimum), math.ceil(real_optimum)}

    def success(iterations: int) -> float:
        return math.sin((2 * iterations + 1) * theta) ** 2

    return max(candidates, key=lambda value: (success(value), -value))


def _predicate_marked_count(table: tuple[int, ...], output_bits: int) -> int | None:
    if output_bits != 1 or any(value not in (0, 1) for value in table):
        return None
    return sum(int(value) for value in table)


def audit_truth_table_oracle(oracle: TruthTableOracle) -> OracleConstructionAudit:
    """Audit the enumerative and answer-disclosure cost of a truth-table compiler."""

    population = 1 << oracle.input_bits
    marked = _predicate_marked_count(oracle.table, oracle.output_bits)
    grover = _first_peak_iterations(population, marked)
    resources = oracle.resource_estimate(phase_kickback=oracle.output_bits == 1)
    return OracleConstructionAudit(
        backend="truth_table_minterm",
        candidate_bits=oracle.input_bits,
        candidate_space_size=population,
        output_bits=oracle.output_bits,
        materialized_truth_table_entries=population,
        reference_evaluations_to_materialize=population,
        artifact_payload_bits_lower_bound=population * oracle.output_bits,
        stores_complete_preimage_information=True,
        marked_count=marked,
        unique_answer_recoverable_from_artifact=marked == 1,
        ideal_first_peak_grover_iterations=grover,
        enumerative_setup_exceeds_ideal_grover=(
            grover is not None and population > grover
        ),
        controlled_x_terms=resources.controlled_x_terms,
        toffoli_gates=resources.toffoli_gates,
        claim_boundary=(
            "The stored table permits a complete classical preimage index. Excluding "
            "table materialization or index construction from a search comparison is "
            "circular for one-shot reconstruction."
        ),
    )


def audit_anf_oracle(oracle: ANFOracle) -> OracleConstructionAudit:
    """Audit the current ANF backend, including its truth-table construction input."""

    population = 1 << oracle.input_bits
    marked = _predicate_marked_count(oracle.table, oracle.output_bits)
    grover = _first_peak_iterations(population, marked)
    resources = oracle.resource_estimate(phase_kickback=oracle.output_bits == 1)
    return OracleConstructionAudit(
        backend="truth_table_to_anf",
        candidate_bits=oracle.input_bits,
        candidate_space_size=population,
        output_bits=oracle.output_bits,
        materialized_truth_table_entries=population,
        reference_evaluations_to_materialize=population,
        artifact_payload_bits_lower_bound=population * oracle.output_bits,
        stores_complete_preimage_information=True,
        marked_count=marked,
        unique_answer_recoverable_from_artifact=marked == 1,
        ideal_first_peak_grover_iterations=grover,
        enumerative_setup_exceeds_ideal_grover=(
            grover is not None and population > grover
        ),
        controlled_x_terms=resources.controlled_x_terms,
        toffoli_gates=resources.toffoli_gates,
        claim_boundary=(
            "ANF may reduce circuit gates, but the present compiler first materializes "
            "all 2^n outputs and retains the table. Gate compression does not erase "
            "the enumerative setup or the available classical inverse index."
        ),
    )


def audit_structure_preserving_oracle(
    oracle: StructurePreservingPredicate,
    *,
    family: str,
    compiler_reference_evaluations: int = 0,
) -> OracleConstructionAudit:
    """Record that a structural compiler emits gates without enumerating candidates.

    This audit certifies absence of the specific truth-table circularity. It does
    not prove that the compiled predicate is classically hard to invert.
    """

    if oracle.input_bits <= 0 or oracle.output_bits <= 0:
        raise ValueError("oracle bit widths must be positive")
    if compiler_reference_evaluations < 0:
        raise ValueError("compiler_reference_evaluations must be non-negative")
    resources = oracle.resource_estimate(phase_kickback=oracle.output_bits == 1)
    return OracleConstructionAudit(
        backend=f"structure_preserving:{family}",
        candidate_bits=oracle.input_bits,
        candidate_space_size=1 << oracle.input_bits,
        output_bits=oracle.output_bits,
        materialized_truth_table_entries=0,
        reference_evaluations_to_materialize=int(compiler_reference_evaluations),
        artifact_payload_bits_lower_bound=0,
        stores_complete_preimage_information=False,
        marked_count=None,
        unique_answer_recoverable_from_artifact=False,
        ideal_first_peak_grover_iterations=None,
        enumerative_setup_exceeds_ideal_grover=False,
        controlled_x_terms=resources.controlled_x_terms,
        toffoli_gates=resources.toffoli_gates,
        claim_boundary=(
            "The compiler avoids candidate enumeration and does not store a preimage "
            "table. A separate reduction or lower bound is still required before "
            "claiming hardness against structure-aware classical inversion."
        ),
    )
