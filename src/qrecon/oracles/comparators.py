from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .arithmetic import ReversibleIntegerAffineValueOracle, _resource_estimate
from .compiler import OracleResourceEstimate
from .models import QuantizedNetwork
from .reversible import ReversibleCircuit, ReversibleGate


def append_equality_to_constant(
    circuit: ReversibleCircuit,
    source: Sequence[int],
    target: int,
    work: Sequence[int],
    constant: int,
) -> tuple[ReversibleGate, ...]:
    """XOR ``[source == constant]`` into target and return all work bits to zero."""

    register = tuple(int(wire) for wire in source)
    target_wire = int(target)
    ancillas = tuple(int(wire) for wire in work)
    width = len(register)
    if width <= 0:
        raise ValueError("equality source must contain at least one bit")
    value = int(constant)
    if value < 0 or value >= (1 << width):
        raise ValueError(f"constant must fit {width} bits")
    required_work = max(0, width - 2)
    if len(ancillas) < required_work:
        raise ValueError(f"equality comparator requires {required_work} clean work bits")
    used = ancillas[:required_work]
    all_wires = register + (target_wire,) + used
    if len(set(all_wires)) != len(all_wires):
        raise ValueError("source, target, and equality work wires must be disjoint")

    start = len(circuit.gates)
    zero_positions = [index for index in range(width) if not ((value >> index) & 1)]
    for index in zero_positions:
        circuit.x(register[index])

    if width == 1:
        circuit.cx(register[0], target_wire)
    elif width == 2:
        circuit.ccx(register[0], register[1], target_wire)
    else:
        circuit.ccx(register[0], register[1], used[0])
        for index in range(2, width - 1):
            circuit.ccx(used[index - 2], register[index], used[index - 1])
        circuit.ccx(used[-1], register[-1], target_wire)
        for index in reversed(range(2, width - 1)):
            circuit.ccx(used[index - 2], register[index], used[index - 1])
        circuit.ccx(register[0], register[1], used[0])

    for index in reversed(zero_positions):
        circuit.x(register[index])
    return circuit.gates[start:]


@dataclass(frozen=True)
class AffineEqualityLayout:
    input_wires: tuple[int, ...]
    target: int
    value_wires: tuple[int, ...]
    value_work: tuple[int, ...]
    equality_work: tuple[int, ...]

    @property
    def work_wires(self) -> tuple[int, ...]:
        return self.value_wires + self.value_work + self.equality_work


def _remap_gates(
    gates: Sequence[ReversibleGate], mapping: dict[int, int]
) -> tuple[ReversibleGate, ...]:
    return tuple(
        ReversibleGate(gate.kind, tuple(mapping[wire] for wire in gate.wires))
        for gate in gates
    )


class ReversibleIntegerAffineEqualityOracle:
    """Clean exact-observation verifier composed with the arithmetic value oracle."""

    output_bits = 1

    def __init__(
        self,
        weights: Sequence[Sequence[int]],
        biases: Sequence[int],
        target_word: int,
        *,
        input_bits_per_feature: int,
        accumulator_bits: int,
        signed_inputs: bool = True,
        signed_accumulator: bool = True,
        require_no_overflow: bool = True,
        max_enumeration_bits: int = 16,
    ) -> None:
        value_oracle = ReversibleIntegerAffineValueOracle(
            weights,
            biases,
            input_bits_per_feature=input_bits_per_feature,
            accumulator_bits=accumulator_bits,
            signed_inputs=signed_inputs,
            signed_accumulator=signed_accumulator,
            require_no_overflow=require_no_overflow,
        )
        target_value = int(target_word)
        if target_value < 0 or target_value >= (1 << value_oracle.output_bits):
            raise ValueError("target_word does not fit the affine output register")
        limit = int(max_enumeration_bits)
        if limit <= 0:
            raise ValueError("max_enumeration_bits must be positive")

        self.value_oracle = value_oracle
        self.target_word = target_value
        self.max_enumeration_bits = limit
        self.feature_count = value_oracle.feature_count
        self.input_bits_per_feature = value_oracle.input_bits_per_feature
        self.accumulator_bits = value_oracle.accumulator_bits
        self.signed_inputs = value_oracle.signed_inputs
        self.range_report = value_oracle.range_report

        input_wires = tuple(range(value_oracle.input_bits))
        target = value_oracle.input_bits
        offset = target + 1
        value_wires = tuple(range(offset, offset + value_oracle.output_bits))
        offset += value_oracle.output_bits
        value_work = tuple(range(offset, offset + len(value_oracle.layout.work_wires)))
        offset += len(value_work)
        equality_work = tuple(
            range(offset, offset + max(0, value_oracle.output_bits - 2))
        )
        offset += len(equality_work)
        self.layout = AffineEqualityLayout(
            input_wires,
            target,
            value_wires,
            value_work,
            equality_work,
        )
        self.circuit = ReversibleCircuit(offset)

        mapping: dict[int, int] = {}
        for source in range(value_oracle.circuit.num_qubits):
            if source < value_oracle.input_bits:
                mapping[source] = source
            else:
                mapping[source] = source + 1
        value_gates = _remap_gates(value_oracle.circuit.gates, mapping)
        self.circuit.extend(value_gates)
        append_equality_to_constant(
            self.circuit,
            self.layout.value_wires,
            self.layout.target,
            self.layout.equality_work,
            self.target_word,
        )
        self.circuit.append_inverse(value_gates)

    @property
    def input_bits(self) -> int:
        return self.value_oracle.input_bits

    def encode_inputs(self, values: Sequence[int]) -> int:
        return self.value_oracle.encode_inputs(values)

    def decode_input_word(self, input_word: int) -> tuple[int, ...]:
        return self.value_oracle.decode_input_word(input_word)

    def evaluate_predicate(self, input_word: int) -> int:
        return int(self.value_oracle.evaluate_input_word(input_word) == self.target_word)

    def _pack_state(self, input_word: int, target: int) -> int:
        input_value = int(input_word)
        target_value = int(target)
        if input_value < 0 or input_value >= (1 << self.input_bits):
            raise ValueError("input_word is outside the candidate register")
        if target_value not in (0, 1):
            raise ValueError("target must be a single bit")
        return input_value | (target_value << self.input_bits)

    def _extract(self, state: int) -> tuple[int, int, int]:
        input_mask = (1 << self.input_bits) - 1
        return (
            state & input_mask,
            (state >> self.input_bits) & 1,
            state >> (self.input_bits + 1),
        )

    def apply(
        self, input_word: int, output_word: int = 0, ancillas: int = 0
    ) -> tuple[int, int, int]:
        if ancillas != 0:
            raise ValueError("clean oracle requires ancillas initialized to zero")
        return self._extract(self.circuit.apply_state(self._pack_state(input_word, output_word)))

    def inverse_apply(
        self, input_word: int, output_word: int = 0, ancillas: int = 0
    ) -> tuple[int, int, int]:
        if ancillas != 0:
            raise ValueError("clean oracle requires ancillas initialized to zero")
        return self._extract(
            self.circuit.apply_inverse_state(self._pack_state(input_word, output_word))
        )

    def marked_inputs(self) -> tuple[int, ...]:
        if self.input_bits > self.max_enumeration_bits:
            raise ValueError("marked-input enumeration exceeds max_enumeration_bits")
        return tuple(
            word
            for word in range(1 << self.input_bits)
            if self.evaluate_predicate(word)
        )

    def phase_sign(self, input_word: int) -> int:
        return -1 if self.apply(input_word, 0)[1] else 1

    def verify_basis_permutation(self) -> bool:
        if self.input_bits > self.max_enumeration_bits:
            raise ValueError("exhaustive verification exceeds max_enumeration_bits")
        for input_word in range(1 << self.input_bits):
            predicate = self.evaluate_predicate(input_word)
            for target in (0, 1):
                forward = self.apply(input_word, target)
                if forward != (input_word, target ^ predicate, 0):
                    return False
                if self.inverse_apply(*forward) != (input_word, target, 0):
                    return False
        return True

    def resource_estimate(self, *, phase_kickback: bool = False) -> OracleResourceEstimate:
        return _resource_estimate(
            self.circuit,
            input_qubits=self.input_bits,
            output_qubits=1,
            work_qubits=len(self.layout.work_wires),
            synthesis=(
                "structure-preserving affine exact-observation verifier: clean "
                "integer affine value oracle, constant equality tree, and Bennett "
                "uncomputation of value and comparator work"
            ),
        )


def compile_structure_preserving_affine_equality_oracle(
    model: QuantizedNetwork,
    target_word: int,
    *,
    require_no_overflow: bool = True,
    max_enumeration_bits: int = 16,
) -> ReversibleIntegerAffineEqualityOracle:
    if len(model.layers) != 1:
        raise ValueError("affine equality lowering requires one affine layer")
    layer = model.layers[0]
    formats = (
        layer.input_format,
        layer.weight_format,
        layer.bias_format,
        layer.output_format,
    )
    if any(fmt.fractional_bits != 0 for fmt in formats):
        raise ValueError("fractional fixed-point equality lowering is not yet supported")
    if layer.activation != "identity" or model.output_mode != "raw":
        raise ValueError("affine equality lowering requires raw identity output")
    return ReversibleIntegerAffineEqualityOracle(
        layer.weights,
        layer.biases,
        target_word,
        input_bits_per_feature=layer.input_format.bits,
        accumulator_bits=layer.output_format.bits,
        signed_inputs=layer.input_format.signed,
        signed_accumulator=layer.output_format.signed,
        require_no_overflow=require_no_overflow,
        max_enumeration_bits=max_enumeration_bits,
    )
