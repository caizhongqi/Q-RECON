from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from .costs import AlgorithmCost, minimum_instances_for_quantum_advantage


def _positive_integer(name: str, value: int) -> int:
    converted = int(value)
    if converted <= 0:
        raise ValueError(f"{name} must be positive")
    return converted


def _non_negative_finite(name: str, value: float) -> float:
    converted = float(value)
    if not math.isfinite(converted) or converted < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return converted


def _ceil_log2(value: int) -> int:
    integer = _positive_integer("value", value)
    return 0 if integer == 1 else (integer - 1).bit_length()


@dataclass(frozen=True)
class ExplicitLookupDescription:
    entries: int
    word_bits: int
    index_bits: int
    table_description_bits: int
    ancilla_qubits: int
    total_circuit_qubits: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class TypicalLookupCircuitLowerBound:
    """Counting lower bound for almost all explicit lookup tables.

    The bound assumes a finite gate alphabet and gates acting on at most
    ``max_gate_arity`` ordered wires. It is a typical/worst-case family bound, not
    a lower bound for every structured table.
    """

    entries: int
    word_bits: int
    index_bits: int
    ancilla_qubits: int
    total_circuit_qubits: int
    table_description_bits: int
    gate_type_count: int
    max_gate_arity: int
    gate_instance_choices_upper_bound: int
    exceptional_fraction: float
    minimum_gate_count: int
    short_circuit_fraction_upper_bound: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class LookupMintermResources:
    entries: int
    word_bits: int
    index_bits: int
    table_description_bits: int
    nonzero_entries: int
    output_one_bits: int
    x_gates: int
    cnot_gates: int
    toffoli_gates: int
    clean_ancillas: int
    logical_qubits: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class ExplicitLookupAmortizationReport:
    entries: int
    word_bits: int
    table_description_bits: int
    classical_setup_cost: float
    quantum_setup_cost: float
    classical_variable_cost: float
    quantum_variable_cost: float
    minimum_instances_for_advantage: int | None

    def to_dict(self) -> dict[str, int | float | None]:
        return asdict(self)


def explicit_lookup_description(
    entries: int, word_bits: int, *, ancilla_qubits: int = 0
) -> ExplicitLookupDescription:
    """Dimensions of ``|i>|y> -> |i>|y xor table[i]>`` for an explicit table."""

    count = _positive_integer("entries", entries)
    width = _positive_integer("word_bits", word_bits)
    ancillas = int(ancilla_qubits)
    if ancillas < 0:
        raise ValueError("ancilla_qubits must be non-negative")
    index_bits = _ceil_log2(count)
    return ExplicitLookupDescription(
        entries=count,
        word_bits=width,
        index_bits=index_bits,
        table_description_bits=count * width,
        ancilla_qubits=ancillas,
        total_circuit_qubits=index_bits + width + ancillas,
    )


def explicit_table_compiler_bit_probe_lower_bound(entries: int, word_bits: int) -> int:
    """Worst-case input-bit probes required by an exact compiler.

    If a compiler ignores one explicit table bit, two descriptions differing only
    in that bit produce the same compiled artifact although their lookup unitaries
    differ on the corresponding address/output bit. Therefore an exact compiler
    must inspect every one of the ``entries * word_bits`` description bits in the
    worst case.
    """

    return explicit_lookup_description(entries, word_bits).table_description_bits


def typical_lookup_circuit_lower_bound(
    entries: int,
    word_bits: int,
    *,
    ancilla_qubits: int = 0,
    gate_type_count: int = 16,
    max_gate_arity: int = 2,
    exceptional_fraction: float = 0.01,
) -> TypicalLookupCircuitLowerBound:
    """Conservative circuit-counting lower bound for almost all lookup tables.

    Let ``A`` upper-bound the number of concrete gate instances available at one
    circuit position. Circuits of length strictly below ``L`` are at most
    ``sum_{j=0}^{L-1} A^j <= A^L/(A-1)``. The returned ``L`` is chosen so this
    upper bound covers no more than ``exceptional_fraction`` of the
    ``2**(entries*word_bits)`` possible tables. Consequently at least
    ``1-exceptional_fraction`` of tables require at least ``L`` gates under the
    declared finite gate model.
    """

    description = explicit_lookup_description(
        entries, word_bits, ancilla_qubits=ancilla_qubits
    )
    gate_types = _positive_integer("gate_type_count", gate_type_count)
    arity = _positive_integer("max_gate_arity", max_gate_arity)
    exceptional = float(exceptional_fraction)
    if not math.isfinite(exceptional) or not 0.0 < exceptional < 1.0:
        raise ValueError("exceptional_fraction must lie strictly between zero and one")

    qubits = max(1, description.total_circuit_qubits)
    choices = gate_types * arity * (qubits**arity)
    choices = max(2, choices)
    log_choices = math.log2(choices)
    description_bits = description.table_description_bits

    # A^L/(A-1) <= exceptional * 2^description_bits.
    real_bound = (
        description_bits
        + math.log2(exceptional)
        + math.log2(choices - 1)
    ) / log_choices
    lower_bound = max(0, math.floor(real_bound))
    while lower_bound > 0 and (
        lower_bound * log_choices - math.log2(choices - 1)
        > description_bits + math.log2(exceptional)
    ):
        lower_bound -= 1

    if lower_bound == 0:
        short_fraction = 0.0
    else:
        log_fraction = (
            lower_bound * log_choices
            - math.log2(choices - 1)
            - description_bits
        )
        short_fraction = min(1.0, 2.0**log_fraction)

    return TypicalLookupCircuitLowerBound(
        entries=description.entries,
        word_bits=description.word_bits,
        index_bits=description.index_bits,
        ancilla_qubits=description.ancilla_qubits,
        total_circuit_qubits=description.total_circuit_qubits,
        table_description_bits=description_bits,
        gate_type_count=gate_types,
        max_gate_arity=arity,
        gate_instance_choices_upper_bound=choices,
        exceptional_fraction=exceptional,
        minimum_gate_count=lower_bound,
        short_circuit_fraction_upper_bound=short_fraction,
    )


def lookup_minterm_resources(
    table: Sequence[int], word_bits: int
) -> LookupMintermResources:
    """Auditable compute-XOR upper bound for a literal explicit lookup table.

    Unused binary addresses (when the number of entries is not a power of two)
    map to zero. Negative index controls are implemented by X conjugation once
    per nonzero address. Each output-one bit receives one multi-controlled X.
    For at least two controls, the count uses the clean-ancilla decomposition
    with ``2*m-3`` Toffoli gates for ``m`` controls.
    """

    values = tuple(int(value) for value in table)
    if not values:
        raise ValueError("table must be non-empty")
    width = _positive_integer("word_bits", word_bits)
    maximum = (1 << width) - 1
    if any(value < 0 or value > maximum for value in values):
        raise ValueError(f"table words must lie in [0, {maximum}]")

    index_bits = _ceil_log2(len(values))
    nonzero_entries = sum(value != 0 for value in values)
    one_bits = sum(value.bit_count() for value in values)
    x_gates = 0
    for address, value in enumerate(values):
        if value == 0:
            continue
        zero_controls = index_bits - address.bit_count()
        x_gates += 2 * zero_controls

    if index_bits == 0:
        x_gates += one_bits
        cnot_gates = 0
        toffoli_gates = 0
        clean_ancillas = 0
    elif index_bits == 1:
        cnot_gates = one_bits
        toffoli_gates = 0
        clean_ancillas = 0
    else:
        cnot_gates = 0
        toffoli_gates = one_bits * (2 * index_bits - 3)
        clean_ancillas = max(0, index_bits - 2)

    return LookupMintermResources(
        entries=len(values),
        word_bits=width,
        index_bits=index_bits,
        table_description_bits=len(values) * width,
        nonzero_entries=nonzero_entries,
        output_one_bits=one_bits,
        x_gates=x_gates,
        cnot_gates=cnot_gates,
        toffoli_gates=toffoli_gates,
        clean_ancillas=clean_ancillas,
        logical_qubits=index_bits + width + clean_ancillas,
    )


def explicit_lookup_amortization_report(
    entries: int,
    word_bits: int,
    *,
    classical_setup_cost_per_table_bit: float = 0.0,
    quantum_setup_cost_per_table_bit: float = 1.0,
    classical_setup_extra: float = 0.0,
    quantum_setup_extra: float = 0.0,
    classical_variable_cost: float,
    quantum_variable_cost: float,
) -> ExplicitLookupAmortizationReport:
    """Break-even workload when an explicit empirical table is compiled once.

    Costs are abstract but must use one common unit. The quantum setup includes
    at least the caller-declared cost per explicit table bit; setting that factor
    to zero is an explicit QRAM/preloaded-oracle assumption, not a free default.
    """

    description_bits = explicit_table_compiler_bit_probe_lower_bound(entries, word_bits)
    classical_per_bit = _non_negative_finite(
        "classical_setup_cost_per_table_bit", classical_setup_cost_per_table_bit
    )
    quantum_per_bit = _non_negative_finite(
        "quantum_setup_cost_per_table_bit", quantum_setup_cost_per_table_bit
    )
    classical_extra = _non_negative_finite("classical_setup_extra", classical_setup_extra)
    quantum_extra = _non_negative_finite("quantum_setup_extra", quantum_setup_extra)
    classical_variable = _non_negative_finite(
        "classical_variable_cost", classical_variable_cost
    )
    quantum_variable = _non_negative_finite("quantum_variable_cost", quantum_variable_cost)

    classical_setup = classical_extra + description_bits * classical_per_bit
    quantum_setup = quantum_extra + description_bits * quantum_per_bit
    classical = AlgorithmCost(
        setup_cost=classical_setup,
        fixed_instance_cost=classical_variable,
        queries=0,
        cost_per_query=0.0,
    )
    quantum = AlgorithmCost(
        setup_cost=quantum_setup,
        fixed_instance_cost=quantum_variable,
        queries=0,
        cost_per_query=0.0,
    )
    return ExplicitLookupAmortizationReport(
        entries=int(entries),
        word_bits=int(word_bits),
        table_description_bits=description_bits,
        classical_setup_cost=classical_setup,
        quantum_setup_cost=quantum_setup,
        classical_variable_cost=classical_variable,
        quantum_variable_cost=quantum_variable,
        minimum_instances_for_advantage=minimum_instances_for_quantum_advantage(
            classical, quantum
        ),
    )
