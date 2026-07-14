from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

from qrecon.theory import compare_search_queries, optimal_standard_grover_iterations

from .analysis import analyze_finite_oracle
from .arithmetic import _resource_estimate, append_cdkm_fixed_adder
from .compiler import OracleResourceEstimate, TruthTableOracle
from .gradient_arithmetic import (
    ReversibleSingleRecordGradientEqualityOracle,
    ReversibleSingleRecordGradientValueOracle,
)
from .grover import estimate_grover_resources, simulate_grover
from .reversible import ReversibleCircuit, ReversibleGate


@dataclass(frozen=True)
class BatchGradientRangeReport:
    minimum: tuple[int, ...]
    maximum: tuple[int, ...]
    no_overflow: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BatchGradientLayout:
    input_registers: tuple[tuple[tuple[int, ...], ...], ...]
    private_target_registers: tuple[tuple[int, ...], ...]
    output_registers: tuple[tuple[int, ...], ...]
    aggregate_registers: tuple[tuple[int, ...], ...]
    record_gradient_registers: tuple[tuple[int, ...], ...]
    record_work: tuple[int, ...]
    public_target_register: tuple[int, ...]

    @property
    def work_wires(self) -> tuple[int, ...]:
        return (
            tuple(wire for register in self.aggregate_registers for wire in register)
            + tuple(wire for register in self.record_gradient_registers for wire in register)
            + self.record_work
            + self.public_target_register
        )


def _allocate_layout(
    batch_size: int,
    features: int,
    input_bits: int,
    gradient_bits: int,
    record_work_count: int,
    public_targets: bool,
) -> tuple[BatchGradientLayout, int, int]:
    offset = 0
    records: list[tuple[tuple[int, ...], ...]] = []
    targets: list[tuple[int, ...]] = []
    for _ in range(batch_size):
        feature_registers: list[tuple[int, ...]] = []
        for _ in range(features):
            feature_registers.append(tuple(range(offset, offset + input_bits)))
            offset += input_bits
        records.append(tuple(feature_registers))
        if not public_targets:
            targets.append(tuple(range(offset, offset + input_bits)))
            offset += input_bits
    input_qubits = offset
    outputs: list[tuple[int, ...]] = []
    aggregates: list[tuple[int, ...]] = []
    temporary: list[tuple[int, ...]] = []
    for destination in (outputs, aggregates, temporary):
        for _ in range(features + 1):
            destination.append(tuple(range(offset, offset + gradient_bits)))
            offset += gradient_bits
    record_work = tuple(range(offset, offset + record_work_count))
    offset += record_work_count
    public_target_register = ()
    if public_targets:
        public_target_register = tuple(range(offset, offset + input_bits))
        offset += input_bits
    return (
        BatchGradientLayout(
            tuple(records),
            tuple(targets),
            tuple(outputs),
            tuple(aggregates),
            tuple(temporary),
            record_work,
            public_target_register,
        ),
        input_qubits,
        offset,
    )


def _remap_gates(
    gates: Sequence[ReversibleGate], mapping: dict[int, int]
) -> tuple[ReversibleGate, ...]:
    return tuple(
        ReversibleGate(gate.kind, tuple(mapping[wire] for wire in gate.wires))
        for gate in gates
    )


class ReversibleBatchGradientValueOracle:
    """Clean sum-gradient oracle for an ordered batch of linear-regression records."""

    def __init__(
        self,
        weights: Sequence[int],
        bias: int,
        *,
        batch_size: int,
        input_bits: int,
        gradient_bits: int,
        public_targets: Sequence[int] | None = None,
        require_no_overflow: bool = True,
        max_enumeration_bits: int = 16,
    ) -> None:
        row = tuple(int(weight) for weight in weights)
        if not row:
            raise ValueError("weights must contain at least one feature")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if input_bits <= 1 or gradient_bits <= 1:
            raise ValueError("signed input and gradient widths must exceed one bit")
        targets = None if public_targets is None else tuple(int(t) for t in public_targets)
        if targets is not None and len(targets) != batch_size:
            raise ValueError("one public target is required per record")
        minimum = -(1 << (input_bits - 1))
        maximum = (1 << (input_bits - 1)) - 1
        if targets is not None and any(t < minimum or t > maximum for t in targets):
            raise OverflowError("a public target does not fit the target word")

        self.weights = row
        self.bias = int(bias)
        self.batch_size = int(batch_size)
        self.feature_count = len(row)
        self.input_bits_per_word = int(input_bits)
        self.gradient_bits = int(gradient_bits)
        self.public_targets = targets
        self.max_enumeration_bits = int(max_enumeration_bits)
        self._input_bits = self.batch_size * (
            self.feature_count + (0 if targets is not None else 1)
        ) * self.input_bits_per_word
        if self._input_bits > self.max_enumeration_bits:
            raise ValueError("candidate space exceeds max_enumeration_bits")

        record_oracle = ReversibleSingleRecordGradientValueOracle(
            row,
            self.bias,
            input_bits=input_bits,
            gradient_bits=gradient_bits,
            require_no_overflow=False,
            max_enumeration_bits=max(
                max_enumeration_bits, (self.feature_count + 1) * input_bits
            ),
        )
        self.record_oracle = record_oracle
        self.layout, self._input_bits, num_qubits = _allocate_layout(
            self.batch_size,
            self.feature_count,
            input_bits,
            gradient_bits,
            len(record_oracle.layout.work_wires),
            targets is not None,
        )
        self.circuit = ReversibleCircuit(num_qubits)
        record_sequences: list[tuple[ReversibleGate, ...]] = []
        for record_index in range(self.batch_size):
            start = len(self.circuit.gates)
            target_register = (
                self.layout.public_target_register
                if targets is not None
                else self.layout.private_target_registers[record_index]
            )
            if targets is not None:
                target_word = targets[record_index] & ((1 << input_bits) - 1)
                for bit, wire in enumerate(target_register):
                    if (target_word >> bit) & 1:
                        self.circuit.x(wire)
            mapping: dict[int, int] = {}
            source_candidates = record_oracle.layout.candidate_registers
            destination_candidates = (
                self.layout.input_registers[record_index] + (target_register,)
            )
            for source, destination in zip(source_candidates, destination_candidates):
                mapping.update(zip(source, destination))
            for source, destination in zip(
                record_oracle.layout.output_registers,
                self.layout.record_gradient_registers,
            ):
                mapping.update(zip(source, destination))
            mapping.update(
                zip(record_oracle.layout.work_wires, self.layout.record_work)
            )
            record_gates = _remap_gates(record_oracle.circuit.gates, mapping)
            self.circuit.extend(record_gates)
            helper = self.layout.record_work[-1]
            for addend, accumulator in zip(
                self.layout.record_gradient_registers,
                self.layout.aggregate_registers,
            ):
                append_cdkm_fixed_adder(self.circuit, addend, accumulator, helper)
            self.circuit.append_inverse(record_gates)
            if targets is not None:
                target_word = targets[record_index] & ((1 << input_bits) - 1)
                for bit, wire in reversed(tuple(enumerate(target_register))):
                    if (target_word >> bit) & 1:
                        self.circuit.x(wire)
            record_sequences.append(self.circuit.gates[start:])

        for source_register, output_register in zip(
            self.layout.aggregate_registers, self.layout.output_registers
        ):
            for source, target in zip(source_register, output_register):
                self.circuit.cx(source, target)
        for sequence in reversed(record_sequences):
            self.circuit.append_inverse(sequence)

        self.range_report = self._exact_range_report()
        if require_no_overflow and not self.range_report.no_overflow:
            raise OverflowError("gradient word does not contain every aggregate component")

    @property
    def input_bits(self) -> int:
        return self._input_bits

    @property
    def output_bits(self) -> int:
        return (self.feature_count + 1) * self.gradient_bits

    @property
    def population(self) -> int:
        return 1 << self.input_bits

    def encode_candidate(
        self,
        inputs: Sequence[Sequence[int]],
        targets: Sequence[int] | None = None,
    ) -> int:
        rows = tuple(tuple(int(value) for value in row) for row in inputs)
        if len(rows) != self.batch_size or any(
            len(row) != self.feature_count for row in rows
        ):
            raise ValueError("input batch shape does not match the oracle")
        if self.public_targets is None:
            if targets is None or len(tuple(targets)) != self.batch_size:
                raise ValueError("private targets must be supplied")
            target_values = tuple(int(value) for value in targets)
        else:
            if targets is not None:
                raise ValueError("targets are public and must not be encoded")
            target_values = self.public_targets
        minimum = -(1 << (self.input_bits_per_word - 1))
        maximum = (1 << (self.input_bits_per_word - 1)) - 1
        mask = (1 << self.input_bits_per_word) - 1
        word = 0
        offset = 0
        for record_index, row in enumerate(rows):
            for value in row:
                if value < minimum or value > maximum:
                    raise OverflowError("input does not fit candidate word")
                word |= (value & mask) << offset
                offset += self.input_bits_per_word
            if self.public_targets is None:
                target = target_values[record_index]
                if target < minimum or target > maximum:
                    raise OverflowError("target does not fit candidate word")
                word |= (target & mask) << offset
                offset += self.input_bits_per_word
        return word

    def decode_candidate(
        self, candidate_word: int
    ) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
        raw = int(candidate_word)
        if raw < 0 or raw >= self.population:
            raise ValueError("candidate_word is outside the candidate register")
        mask = (1 << self.input_bits_per_word) - 1
        offset = 0
        rows: list[tuple[int, ...]] = []
        targets: list[int] = []

        def decode(word: int) -> int:
            return (
                word - (1 << self.input_bits_per_word)
                if word >= (1 << (self.input_bits_per_word - 1))
                else word
            )

        for record_index in range(self.batch_size):
            row: list[int] = []
            for _ in range(self.feature_count):
                row.append(decode((raw >> offset) & mask))
                offset += self.input_bits_per_word
            rows.append(tuple(row))
            if self.public_targets is None:
                targets.append(decode((raw >> offset) & mask))
                offset += self.input_bits_per_word
            else:
                targets.append(self.public_targets[record_index])
        return tuple(rows), tuple(targets)

    def gradient_components(self, candidate_word: int) -> tuple[int, ...]:
        inputs, targets = self.decode_candidate(candidate_word)
        weight_gradient = [0] * self.feature_count
        bias_gradient = 0
        for row, target in zip(inputs, targets):
            residual = self.bias + sum(
                weight * value for weight, value in zip(self.weights, row)
            ) - target
            bias_gradient += residual
            for index, value in enumerate(row):
                weight_gradient[index] += residual * value
        return tuple(weight_gradient) + (bias_gradient,)

    def evaluate_input_word(self, candidate_word: int) -> int:
        mask = (1 << self.gradient_bits) - 1
        return sum(
            (value & mask) << (index * self.gradient_bits)
            for index, value in enumerate(self.gradient_components(candidate_word))
        )

    def _exact_range_report(self) -> BatchGradientRangeReport:
        minima = [float("inf")] * (self.feature_count + 1)
        maxima = [float("-inf")] * (self.feature_count + 1)
        lower = -(1 << (self.gradient_bits - 1))
        upper = (1 << (self.gradient_bits - 1)) - 1
        safe = True
        for candidate in range(self.population):
            for index, value in enumerate(self.gradient_components(candidate)):
                minima[index] = min(minima[index], value)
                maxima[index] = max(maxima[index], value)
                safe = safe and lower <= value <= upper
        return BatchGradientRangeReport(
            tuple(int(value) for value in minima),
            tuple(int(value) for value in maxima),
            safe,
        )

    def compile_reference_oracle(self) -> TruthTableOracle:
        return TruthTableOracle.from_function(
            self.input_bits,
            self.output_bits,
            self.evaluate_input_word,
            max_input_bits=self.max_enumeration_bits,
            name="aggregate_linear_gradient_reference",
        )

    def _pack_state(self, input_word: int, output_word: int) -> int:
        if input_word < 0 or input_word >= (1 << self.input_bits):
            raise ValueError("input is outside candidate register")
        if output_word < 0 or output_word >= (1 << self.output_bits):
            raise ValueError("output is outside aggregate-gradient register")
        return int(input_word) | (int(output_word) << self.input_bits)

    def _extract(self, state: int) -> tuple[int, int, int]:
        input_mask = (1 << self.input_bits) - 1
        output_mask = (1 << self.output_bits) - 1
        return (
            state & input_mask,
            (state >> self.input_bits) & output_mask,
            state >> (self.input_bits + self.output_bits),
        )

    def apply(
        self, input_word: int, output_word: int = 0, ancillas: int = 0
    ) -> tuple[int, int, int]:
        if ancillas != 0:
            raise ValueError("clean oracle requires ancillas initialized to zero")
        return self._extract(
            self.circuit.apply_state(self._pack_state(input_word, output_word))
        )

    def inverse_apply(
        self, input_word: int, output_word: int = 0, ancillas: int = 0
    ) -> tuple[int, int, int]:
        if ancillas != 0:
            raise ValueError("clean oracle requires ancillas initialized to zero")
        return self._extract(
            self.circuit.apply_inverse_state(self._pack_state(input_word, output_word))
        )

    def verify_basis_permutation(self) -> bool:
        for candidate in range(self.population):
            expected = self.evaluate_input_word(candidate)
            for output in (0, 1, (1 << self.output_bits) - 1):
                forward = self.apply(candidate, output)
                if forward != (candidate, output ^ expected, 0):
                    return False
                if self.inverse_apply(*forward) != (candidate, output, 0):
                    return False
        return True

    def resource_estimate(self, *, phase_kickback: bool = False) -> OracleResourceEstimate:
        if phase_kickback:
            raise ValueError("phase kickback requires the equality predicate")
        return _resource_estimate(
            self.circuit,
            input_qubits=self.input_bits,
            output_qubits=self.output_bits,
            work_qubits=len(self.layout.work_wires),
            synthesis=(
                "ordered-batch aggregate-gradient oracle built from one reusable "
                "single-record residual/product circuit, modular gradient sums, "
                "output copy, and reverse record cleanup"
            ),
        )


class ReversibleBatchGradientEqualityOracle(
    ReversibleSingleRecordGradientEqualityOracle
):
    def resource_estimate(self, *, phase_kickback: bool = False) -> OracleResourceEstimate:
        return _resource_estimate(
            self.circuit,
            input_qubits=self.input_bits,
            output_qubits=1,
            work_qubits=len(self.layout.work_wires),
            synthesis=(
                "clean aggregate-gradient value oracle, exact full-word equality, "
                "and Bennett reverse uncomputation"
            ),
        )


def run_batch_gradient_reconstruction(
    value_oracle: ReversibleBatchGradientValueOracle,
    inputs: Sequence[Sequence[int]],
    targets: Sequence[int] | None = None,
    *,
    target_success: float = 0.8,
) -> dict[str, object]:
    candidate = value_oracle.encode_candidate(inputs, targets)
    observed = value_oracle.evaluate_input_word(candidate)
    verifier = ReversibleBatchGradientEqualityOracle(value_oracle, observed)
    marked = verifier.marked_inputs()
    comparison = compare_search_queries(
        value_oracle.population, len(marked), target_success=target_success
    )
    iterations = optimal_standard_grover_iterations(
        value_oracle.population, len(marked)
    ) or 0
    simulation = simulate_grover(verifier, iterations)
    return {
        "true_candidate": candidate,
        "observed_word": observed,
        "marked_candidates": list(marked),
        "exact_original_identifiable": len(marked) == 1,
        "finite_identifiability": analyze_finite_oracle(
            value_oracle.compile_reference_oracle()
        ).to_dict(),
        "classical_queries": comparison.classical_queries,
        "grover_queries": comparison.grover_queries,
        "grover_success_probability": simulation.success_probability,
        "grover_resources": estimate_grover_resources(verifier, iterations).to_dict(),
    }
