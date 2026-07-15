from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .data_loading import explicit_table_compiler_bit_probe_lower_bound
from .search import grover_success, optimal_standard_grover_iterations


@dataclass(frozen=True)
class OneShotExplicitTableBoundary:
    """One-shot input-access boundary for an arbitrary explicit candidate table.

    The comparison uses *table-description bit probes* only. It does not convert
    quantum gates, fault-tolerant cycles, or wall-clock time into that unit. Its
    purpose is narrower: an exact compiler receiving a literal table cannot make
    the end-to-end input-processing cost sublinear merely because the subsequent
    coherent search uses fewer verifier calls.
    """

    entries: int
    word_bits: int
    marked: int
    table_description_bits: int
    exact_quantum_compiler_probe_lower_bound: int
    classical_full_scan_probe_upper_bound: int
    ideal_grover_iterations: int
    ideal_grover_success: float
    quantum_one_shot_total_probe_lower_bound: int
    classical_one_shot_total_probe_upper_bound: int
    rules_out_sublinear_one_shot_input_processing: bool
    assumption: str

    def to_dict(self) -> dict[str, int | float | bool | str]:
        return asdict(self)


@dataclass(frozen=True)
class AmortizedExplicitTableProbeFloor:
    entries: int
    word_bits: int
    instances: int
    table_description_bits: int
    total_compiler_probe_lower_bound: int
    amortized_probe_lower_bound_per_instance: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def one_shot_explicit_table_boundary(
    entries: int,
    word_bits: int,
    *,
    marked: int = 1,
) -> OneShotExplicitTableBoundary:
    """Compare one-shot table-input complexity with ideal Grover query count.

    Model assumptions:

    1. the candidate table is supplied as a literal classical ``entries*word_bits``
       description to the experiment;
    2. a data-dependent coherent lookup/verifier is compiled exactly from that
       description;
    3. no already-loaded QRAM or externally supplied coherent oracle is free;
    4. the classical reference may perform a direct worst-case full table scan.

    Under these assumptions, exact compilation needs at least one probe to every
    table bit in the worst case, while a classical scan uses at most the same
    number of table-bit probes. Therefore the one-shot total input-processing
    complexity is ``Theta(entries*word_bits)`` even if the *post-compilation*
    verifier-query count is ``Theta(sqrt(entries/marked))``.
    """

    count = int(entries)
    width = int(word_bits)
    solutions = int(marked)
    if count <= 0:
        raise ValueError("entries must be positive")
    if width <= 0:
        raise ValueError("word_bits must be positive")
    if solutions <= 0 or solutions > count:
        raise ValueError("marked must lie in [1, entries]")

    table_bits = explicit_table_compiler_bit_probe_lower_bound(count, width)
    iterations = optimal_standard_grover_iterations(count, solutions)
    if iterations is None:  # unreachable because marked is positive
        raise RuntimeError("positive marked count unexpectedly produced no Grover plan")
    success = grover_success(count, solutions, iterations)
    return OneShotExplicitTableBoundary(
        entries=count,
        word_bits=width,
        marked=solutions,
        table_description_bits=table_bits,
        exact_quantum_compiler_probe_lower_bound=table_bits,
        classical_full_scan_probe_upper_bound=table_bits,
        ideal_grover_iterations=iterations,
        ideal_grover_success=success,
        quantum_one_shot_total_probe_lower_bound=table_bits,
        classical_one_shot_total_probe_upper_bound=table_bits,
        rules_out_sublinear_one_shot_input_processing=True,
        assumption=(
            "Literal classical table input; exact data-dependent coherent compiler; "
            "no free preloaded QRAM or externally supplied coherent oracle."
        ),
    )


def amortized_explicit_table_probe_floor(
    entries: int,
    word_bits: int,
    instances: int,
) -> AmortizedExplicitTableProbeFloor:
    """Amortized lower bound from consuming one explicit table description once."""

    uses = int(instances)
    if uses <= 0:
        raise ValueError("instances must be positive")
    table_bits = explicit_table_compiler_bit_probe_lower_bound(entries, word_bits)
    return AmortizedExplicitTableProbeFloor(
        entries=int(entries),
        word_bits=int(word_bits),
        instances=uses,
        table_description_bits=table_bits,
        total_compiler_probe_lower_bound=table_bits,
        amortized_probe_lower_bound_per_instance=table_bits / uses,
    )


def minimum_instances_for_amortized_probe_budget(
    entries: int,
    word_bits: int,
    maximum_probes_per_instance: float,
) -> int:
    """Smallest reuse count making explicit-table input cost fit a per-task budget."""

    budget = float(maximum_probes_per_instance)
    if not math.isfinite(budget) or budget <= 0.0:
        raise ValueError("maximum_probes_per_instance must be positive and finite")
    table_bits = explicit_table_compiler_bit_probe_lower_bound(entries, word_bits)
    return max(1, math.ceil(table_bits / budget))
