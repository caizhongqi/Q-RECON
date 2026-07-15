from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .arithmetic import _resource_estimate
from .comparators import append_equality_to_constant
from .compiler import OracleResourceEstimate
from .fixed_point_mlp import ReversibleFixedPointMLPValueOracle, _remap_gates
from .models import QuantizedAffineLayer
from .reversible import ReversibleCircuit


@dataclass(frozen=True)
class FixedPointMLPEqualityLayout:
    input_wires: tuple[int, ...]
    target: int
    value_wires: tuple[int, ...]
    value_work: tuple[int, ...]
    equality_work: tuple[int, ...]

    @property
    def work_wires(self) -> tuple[int, ...]:
        return self.value_wires + self.value_work + self.equality_work


class ReversibleFixedPointMLPEqualityOracle:
    """Clean exact-output verifier for a two-layer fixed-point MLP.

    The circuit computes the complete bit-exact MLP output, toggles one target
    qubit iff the packed output equals ``target_codes``, and reverses the value
    computation.  Consequently every value and comparator work qubit returns to
    zero and the oracle can be used directly through phase kickback.
    """

    output_bits = 1

    def __init__(
        self,
        hidden_layer: QuantizedAffineLayer,
        output_layer: QuantizedAffineLayer,
        target_codes: Sequence[int],
        *,
        require_no_overflow: bool = True,
        max_enumeration_bits: int = 16,
    ) -> None:
        target = tuple(int(value) for value in target_codes)
        if len(target) != output_layer.output_dimension:
            raise ValueError("one target code is required per final output")
        for value in target:
            output_layer.output_format.require_code(value)

        self.value = ReversibleFixedPointMLPValueOracle(
            hidden_layer,
            output_layer,
            require_no_overflow=require_no_overflow,
            max_enumeration_bits=max_enumeration_bits,
        )
        self.hidden_layer = hidden_layer
        self.output_layer = output_layer
        self.target_codes = target
        self.max_enumeration_bits = int(max_enumeration_bits)
        if self.max_enumeration_bits <= 0:
            raise ValueError("max_enumeration_bits must be positive")

        target_word = 0
        width = output_layer.output_format.bits
        for index, code in enumerate(target):
            target_word |= output_layer.output_format.code_to_word(code) << (
                index * width
            )
        self.target_word = target_word

        offset = self.value.input_bits
        target_wire = offset
        offset += 1
        value_wires = tuple(range(offset, offset + self.value.output_bits))
        offset += len(value_wires)
        value_work = tuple(
            range(offset, offset + len(self.value.layout.work_wires))
        )
        offset += len(value_work)
        equality_work = tuple(
            range(offset, offset + max(0, self.value.output_bits - 2))
        )
        offset += len(equality_work)
        self.layout = FixedPointMLPEqualityLayout(
            tuple(range(self.value.input_bits)),
            target_wire,
            value_wires,
            value_work,
            equality_work,
        )
        self.circuit = ReversibleCircuit(offset)

        mapping: dict[int, int] = {}
        mapping.update(zip(self.value.layout.input_wires, self.layout.input_wires))
        mapping.update(zip(self.value.layout.output_wires, self.layout.value_wires))
        mapping.update(zip(self.value.layout.work_wires, self.layout.value_work))
        value_gates = _remap_gates(self.value.circuit.gates, mapping)

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
        return self.value.input_bits

    def encode_inputs(self, values: Sequence[int]) -> int:
        return self.value.encode_inputs(values)

    def evaluate_predicate(self, input_word: int) -> int:
        return int(self.value.evaluate_input_word(input_word) == self.target_word)

    def _pack_state(self, input_word: int, output_word: int) -> int:
        candidate = int(input_word)
        target = int(output_word)
        if candidate < 0 or candidate >= (1 << self.input_bits):
            raise ValueError("input_word is outside the candidate register")
        if target not in (0, 1):
            raise ValueError("output_word must be one bit")
        return candidate | (target << self.input_bits)

    def _extract(self, state: int) -> tuple[int, int, int]:
        return (
            state & ((1 << self.input_bits) - 1),
            (state >> self.input_bits) & 1,
            state >> (self.input_bits + 1),
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
            self.circuit.apply_inverse_state(
                self._pack_state(input_word, output_word)
            )
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
            expected = self.evaluate_predicate(input_word)
            for output_word in (0, 1):
                forward = self.apply(input_word, output_word)
                if forward != (input_word, output_word ^ expected, 0):
                    return False
                if self.inverse_apply(*forward) != (input_word, output_word, 0):
                    return False
        return True

    def resource_estimate(
        self, *, phase_kickback: bool = False
    ) -> OracleResourceEstimate:
        return _resource_estimate(
            self.circuit,
            input_qubits=self.input_bits,
            output_qubits=1,
            work_qubits=len(self.layout.work_wires),
            synthesis=(
                "two-layer fixed-point MLP exact-observation verifier: clean "
                "fixed-point MLP value oracle, packed constant equality tree, "
                "and complete Bennett uncomputation"
            ),
        )


def compile_structure_preserving_fixed_point_mlp_equality_oracle(
    hidden_layer: QuantizedAffineLayer,
    output_layer: QuantizedAffineLayer,
    target_codes: Sequence[int],
    *,
    require_no_overflow: bool = True,
    max_enumeration_bits: int = 16,
) -> ReversibleFixedPointMLPEqualityOracle:
    return ReversibleFixedPointMLPEqualityOracle(
        hidden_layer,
        output_layer,
        target_codes,
        require_no_overflow=require_no_overflow,
        max_enumeration_bits=max_enumeration_bits,
    )
