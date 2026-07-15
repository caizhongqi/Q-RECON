from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .arithmetic import ReversibleIntegerAffinePredicateOracle, _resource_estimate
from .compiler import OracleResourceEstimate
from .fixed_point_affine import ReversibleFixedPointAffineValueOracle
from .models import QuantizedAffineLayer
from .reversible import ReversibleCircuit, ReversibleGate


def _remap_gates(
    gates: Sequence[ReversibleGate], mapping: dict[int, int]
) -> tuple[ReversibleGate, ...]:
    return tuple(
        ReversibleGate(gate.kind, tuple(mapping[wire] for wire in gate.wires))
        for gate in gates
    )


def _identity_copy(layer: QuantizedAffineLayer) -> QuantizedAffineLayer:
    return QuantizedAffineLayer(
        weights=layer.weights,
        biases=layer.biases,
        input_format=layer.input_format,
        weight_format=layer.weight_format,
        bias_format=layer.bias_format,
        output_format=layer.output_format,
        activation="identity",
    )


def _pack_output_codes(layer: QuantizedAffineLayer, values: Sequence[int]) -> int:
    word = 0
    for index, code in enumerate(values):
        word |= layer.output_format.code_to_word(code) << (
            index * layer.output_format.bits
        )
    return word


@dataclass(frozen=True)
class FixedPointAffineReLULayout:
    input_wires: tuple[int, ...]
    output_wires: tuple[int, ...]
    preactivation_wires: tuple[int, ...]
    affine_work: tuple[int, ...]

    @property
    def work_wires(self) -> tuple[int, ...]:
        return self.preactivation_wires + self.affine_work


class ReversibleFixedPointAffineReLUValueOracle:
    """Clean fixed-point affine/ReLU value oracle with exact reference semantics."""

    def __init__(
        self,
        layer: QuantizedAffineLayer,
        *,
        require_no_overflow: bool = True,
        max_enumeration_bits: int = 16,
    ) -> None:
        if layer.activation != "relu":
            raise ValueError("fixed-point ReLU lowering requires activation='relu'")
        self.layer = layer
        self.max_enumeration_bits = int(max_enumeration_bits)
        self.affine = ReversibleFixedPointAffineValueOracle(
            _identity_copy(layer),
            require_no_overflow=require_no_overflow,
            max_enumeration_bits=max_enumeration_bits,
        )

        input_bits = self.affine.input_bits
        output_bits = self.affine.output_bits
        offset = input_bits
        output_wires = tuple(range(offset, offset + output_bits)); offset += output_bits
        preactivation = tuple(range(offset, offset + output_bits)); offset += output_bits
        affine_work = tuple(
            range(offset, offset + len(self.affine.layout.work_wires))
        ); offset += len(affine_work)
        self.layout = FixedPointAffineReLULayout(
            tuple(range(input_bits)), output_wires, preactivation, affine_work
        )
        self.circuit = ReversibleCircuit(offset)

        mapping: dict[int, int] = {}
        mapping.update(zip(self.affine.layout.input_wires, self.layout.input_wires))
        mapping.update(zip(self.affine.layout.output_wires, preactivation))
        mapping.update(zip(self.affine.layout.work_wires, affine_work))
        affine_gates = _remap_gates(self.affine.circuit.gates, mapping)
        self.circuit.extend(affine_gates)

        bits = layer.output_format.bits
        for row in range(layer.output_dimension):
            start = row * bits
            source = preactivation[start : start + bits]
            target = output_wires[start : start + bits]
            if layer.output_format.signed:
                sign = source[-1]
                self.circuit.x(sign)
                for source_wire, target_wire in zip(source[:-1], target[:-1]):
                    self.circuit.ccx(sign, source_wire, target_wire)
                self.circuit.x(sign)
            else:
                for source_wire, target_wire in zip(source, target):
                    self.circuit.cx(source_wire, target_wire)

        self.circuit.append_inverse(affine_gates)

    @property
    def input_bits(self) -> int:
        return self.affine.input_bits

    @property
    def output_bits(self) -> int:
        return self.affine.output_bits

    def encode_inputs(self, values: Sequence[int]) -> int:
        return self.affine.encode_inputs(values)

    def evaluate_input_word(self, input_word: int) -> int:
        inputs = self.affine.raw_affine.decode_input_word(input_word)
        return _pack_output_codes(self.layer, self.layer.evaluate_codes(inputs))

    def _pack_state(self, input_word: int, output_word: int) -> int:
        if input_word < 0 or input_word >= (1 << self.input_bits):
            raise ValueError("input_word is outside the input register")
        if output_word < 0 or output_word >= (1 << self.output_bits):
            raise ValueError("output_word is outside the output register")
        return int(input_word) | (int(output_word) << self.input_bits)

    def _extract(self, state: int) -> tuple[int, int, int]:
        input_mask = (1 << self.input_bits) - 1
        output_mask = (1 << self.output_bits) - 1
        return (
            state & input_mask,
            (state >> self.input_bits) & output_mask,
            state >> (self.input_bits + self.output_bits),
        )

    def apply(self, input_word: int, output_word: int = 0, ancillas: int = 0) -> tuple[int, int, int]:
        if ancillas != 0:
            raise ValueError("clean oracle requires ancillas initialized to zero")
        return self._extract(self.circuit.apply_state(self._pack_state(input_word, output_word)))

    def inverse_apply(self, input_word: int, output_word: int = 0, ancillas: int = 0) -> tuple[int, int, int]:
        if ancillas != 0:
            raise ValueError("clean oracle requires ancillas initialized to zero")
        return self._extract(self.circuit.apply_inverse_state(self._pack_state(input_word, output_word)))

    def verify_basis_permutation(self) -> bool:
        if self.input_bits > self.max_enumeration_bits:
            raise ValueError("exhaustive verification exceeds max_enumeration_bits")
        for input_word in range(1 << self.input_bits):
            expected = self.evaluate_input_word(input_word)
            for output_word in (0, 1, (1 << self.output_bits) - 1):
                forward = self.apply(input_word, output_word)
                if forward != (input_word, output_word ^ expected, 0):
                    return False
                if self.inverse_apply(*forward) != (input_word, output_word, 0):
                    return False
        return True

    def resource_estimate(self, *, phase_kickback: bool = False) -> OracleResourceEstimate:
        if phase_kickback:
            raise ValueError("phase kickback requires a one-bit predicate")
        return _resource_estimate(
            self.circuit,
            input_qubits=self.input_bits,
            output_qubits=self.output_bits,
            work_qubits=len(self.layout.work_wires),
            synthesis=(
                "fixed-point affine/ReLU value oracle: clean fixed-point affine "
                "preactivation, sign-controlled nonnegative copy, and reverse affine cleanup"
            ),
        )


@dataclass(frozen=True)
class FixedPointMLPLayout:
    input_wires: tuple[int, ...]
    output_wires: tuple[int, ...]
    hidden_wires: tuple[int, ...]
    hidden_work: tuple[int, ...]
    output_work: tuple[int, ...]

    @property
    def work_wires(self) -> tuple[int, ...]:
        return self.hidden_wires + self.hidden_work + self.output_work


class ReversibleFixedPointMLPValueOracle:
    """Clean two-layer fixed-point ``Affine-ReLU-Affine`` value oracle."""

    def __init__(
        self,
        hidden_layer: QuantizedAffineLayer,
        output_layer: QuantizedAffineLayer,
        *,
        require_no_overflow: bool = True,
        max_enumeration_bits: int = 16,
    ) -> None:
        if hidden_layer.activation != "relu":
            raise ValueError("hidden layer must use ReLU")
        if output_layer.activation != "identity":
            raise ValueError("output layer must use identity activation")
        if hidden_layer.output_dimension != output_layer.input_dimension:
            raise ValueError("hidden and output layer dimensions do not match")
        if hidden_layer.output_format != output_layer.input_format:
            raise ValueError("hidden output and final input formats must match exactly")
        self.hidden_layer = hidden_layer
        self.output_layer = output_layer
        self.max_enumeration_bits = int(max_enumeration_bits)
        self.hidden = ReversibleFixedPointAffineReLUValueOracle(
            hidden_layer,
            require_no_overflow=require_no_overflow,
            max_enumeration_bits=max_enumeration_bits,
        )
        self.output = ReversibleFixedPointAffineValueOracle(
            output_layer,
            require_no_overflow=require_no_overflow,
            max_enumeration_bits=max_enumeration_bits,
        )

        offset = self.hidden.input_bits
        output_wires = tuple(range(offset, offset + self.output.output_bits)); offset += self.output.output_bits
        hidden_wires = tuple(range(offset, offset + self.hidden.output_bits)); offset += self.hidden.output_bits
        hidden_work = tuple(range(offset, offset + len(self.hidden.layout.work_wires))); offset += len(hidden_work)
        output_work = tuple(range(offset, offset + len(self.output.layout.work_wires))); offset += len(output_work)
        self.layout = FixedPointMLPLayout(
            tuple(range(self.hidden.input_bits)),
            output_wires,
            hidden_wires,
            hidden_work,
            output_work,
        )
        self.circuit = ReversibleCircuit(offset)

        hidden_mapping: dict[int, int] = {}
        hidden_mapping.update(zip(self.hidden.layout.input_wires, self.layout.input_wires))
        hidden_mapping.update(zip(self.hidden.layout.output_wires, hidden_wires))
        hidden_mapping.update(zip(self.hidden.layout.work_wires, hidden_work))
        hidden_gates = _remap_gates(self.hidden.circuit.gates, hidden_mapping)

        output_mapping: dict[int, int] = {}
        output_mapping.update(zip(self.output.layout.input_wires, hidden_wires))
        output_mapping.update(zip(self.output.layout.output_wires, output_wires))
        output_mapping.update(zip(self.output.layout.work_wires, output_work))
        output_gates = _remap_gates(self.output.circuit.gates, output_mapping)

        self.circuit.extend(hidden_gates)
        self.circuit.extend(output_gates)
        self.circuit.append_inverse(hidden_gates)

    @property
    def input_bits(self) -> int:
        return self.hidden.input_bits

    @property
    def output_bits(self) -> int:
        return self.output.output_bits

    def encode_inputs(self, values: Sequence[int]) -> int:
        return self.hidden.encode_inputs(values)

    def evaluate_input_word(self, input_word: int) -> int:
        inputs = self.hidden.affine.raw_affine.decode_input_word(input_word)
        hidden = self.hidden_layer.evaluate_codes(inputs)
        outputs = self.output_layer.evaluate_codes(hidden)
        return _pack_output_codes(self.output_layer, outputs)

    def _pack_state(self, input_word: int, output_word: int) -> int:
        if input_word < 0 or input_word >= (1 << self.input_bits):
            raise ValueError("input_word is outside the input register")
        if output_word < 0 or output_word >= (1 << self.output_bits):
            raise ValueError("output_word is outside the output register")
        return int(input_word) | (int(output_word) << self.input_bits)

    def _extract(self, state: int) -> tuple[int, int, int]:
        input_mask = (1 << self.input_bits) - 1
        output_mask = (1 << self.output_bits) - 1
        return (
            state & input_mask,
            (state >> self.input_bits) & output_mask,
            state >> (self.input_bits + self.output_bits),
        )

    def apply(self, input_word: int, output_word: int = 0, ancillas: int = 0) -> tuple[int, int, int]:
        if ancillas != 0:
            raise ValueError("clean oracle requires ancillas initialized to zero")
        return self._extract(self.circuit.apply_state(self._pack_state(input_word, output_word)))

    def inverse_apply(self, input_word: int, output_word: int = 0, ancillas: int = 0) -> tuple[int, int, int]:
        if ancillas != 0:
            raise ValueError("clean oracle requires ancillas initialized to zero")
        return self._extract(self.circuit.apply_inverse_state(self._pack_state(input_word, output_word)))

    def verify_basis_permutation(self) -> bool:
        if self.input_bits > self.max_enumeration_bits:
            raise ValueError("exhaustive verification exceeds max_enumeration_bits")
        for input_word in range(1 << self.input_bits):
            expected = self.evaluate_input_word(input_word)
            for output_word in (0, 1, (1 << self.output_bits) - 1):
                forward = self.apply(input_word, output_word)
                if forward != (input_word, output_word ^ expected, 0):
                    return False
                if self.inverse_apply(*forward) != (input_word, output_word, 0):
                    return False
        return True

    def resource_estimate(self, *, phase_kickback: bool = False) -> OracleResourceEstimate:
        if phase_kickback:
            raise ValueError("phase kickback requires a one-bit predicate")
        return _resource_estimate(
            self.circuit,
            input_qubits=self.input_bits,
            output_qubits=self.output_bits,
            work_qubits=len(self.layout.work_wires),
            synthesis=(
                "two-layer fixed-point MLP value oracle: clean affine/ReLU hidden "
                "activation, clean fixed-point output affine, and reverse hidden cleanup"
            ),
        )


@dataclass(frozen=True)
class FixedPointMLPPredicateLayout:
    input_wires: tuple[int, ...]
    target: int
    logit_wires: tuple[int, ...]
    value_work: tuple[int, ...]
    predicate_work: tuple[int, ...]

    @property
    def work_wires(self) -> tuple[int, ...]:
        return self.logit_wires + self.value_work + self.predicate_work


class ReversibleFixedPointMLPPredicateOracle:
    """Clean threshold predicate for a one-output fixed-point two-layer MLP."""

    def __init__(
        self,
        hidden_layer: QuantizedAffineLayer,
        output_layer: QuantizedAffineLayer,
        *,
        threshold_code: int = 0,
        require_no_overflow: bool = True,
        max_enumeration_bits: int = 16,
    ) -> None:
        if output_layer.output_dimension != 1:
            raise ValueError("predicate lowering requires one final output")
        output_layer.output_format.require_code(threshold_code)
        self.threshold_code = int(threshold_code)
        self.value = ReversibleFixedPointMLPValueOracle(
            hidden_layer,
            output_layer,
            require_no_overflow=require_no_overflow,
            max_enumeration_bits=max_enumeration_bits,
        )
        self.output_format = output_layer.output_format
        self.max_enumeration_bits = int(max_enumeration_bits)
        predicate_bits = self.output_format.bits + 1
        self.predicate = ReversibleIntegerAffinePredicateOracle(
            (1,),
            0,
            input_bits_per_feature=self.output_format.bits,
            accumulator_bits=predicate_bits,
            threshold=self.threshold_code,
            signed_inputs=self.output_format.signed,
            require_no_overflow=True,
        )

        offset = self.value.input_bits
        target = offset; offset += 1
        logit = tuple(range(offset, offset + self.value.output_bits)); offset += len(logit)
        value_work = tuple(range(offset, offset + len(self.value.layout.work_wires))); offset += len(value_work)
        predicate_work = tuple(range(offset, offset + len(self.predicate.layout.work_wires))); offset += len(predicate_work)
        self.layout = FixedPointMLPPredicateLayout(
            tuple(range(self.value.input_bits)), target, logit, value_work, predicate_work
        )
        self.circuit = ReversibleCircuit(offset)

        value_mapping: dict[int, int] = {}
        value_mapping.update(zip(self.value.layout.input_wires, self.layout.input_wires))
        value_mapping.update(zip(self.value.layout.output_wires, logit))
        value_mapping.update(zip(self.value.layout.work_wires, value_work))
        value_gates = _remap_gates(self.value.circuit.gates, value_mapping)

        predicate_mapping: dict[int, int] = {}
        predicate_mapping.update(zip(self.predicate.layout.input_registers[0], logit))
        predicate_mapping[self.predicate.layout.target] = target
        predicate_mapping.update(zip(self.predicate.layout.work_wires, predicate_work))
        predicate_gates = _remap_gates(self.predicate.circuit.gates, predicate_mapping)

        self.circuit.extend(value_gates)
        self.circuit.extend(predicate_gates)
        self.circuit.append_inverse(value_gates)

    @property
    def input_bits(self) -> int:
        return self.value.input_bits

    @property
    def output_bits(self) -> int:
        return 1

    def encode_inputs(self, values: Sequence[int]) -> int:
        return self.value.encode_inputs(values)

    def evaluate_label(self, input_word: int) -> int:
        logit_word = self.value.evaluate_input_word(input_word)
        logit = self.output_format.word_to_code(logit_word)
        return int(logit >= self.threshold_code)

    def _pack_state(self, input_word: int, output_word: int) -> int:
        if input_word < 0 or input_word >= (1 << self.input_bits):
            raise ValueError("input_word is outside the input register")
        if output_word not in (0, 1):
            raise ValueError("output_word must be one bit")
        return int(input_word) | (int(output_word) << self.input_bits)

    def _extract(self, state: int) -> tuple[int, int, int]:
        return (
            state & ((1 << self.input_bits) - 1),
            (state >> self.input_bits) & 1,
            state >> (self.input_bits + 1),
        )

    def apply(self, input_word: int, output_word: int = 0, ancillas: int = 0) -> tuple[int, int, int]:
        if ancillas != 0:
            raise ValueError("clean oracle requires ancillas initialized to zero")
        return self._extract(self.circuit.apply_state(self._pack_state(input_word, output_word)))

    def inverse_apply(self, input_word: int, output_word: int = 0, ancillas: int = 0) -> tuple[int, int, int]:
        if ancillas != 0:
            raise ValueError("clean oracle requires ancillas initialized to zero")
        return self._extract(self.circuit.apply_inverse_state(self._pack_state(input_word, output_word)))

    def marked_inputs(self) -> tuple[int, ...]:
        if self.input_bits > self.max_enumeration_bits:
            raise ValueError("marked-input enumeration exceeds max_enumeration_bits")
        return tuple(
            word for word in range(1 << self.input_bits) if self.evaluate_label(word)
        )

    def phase_sign(self, input_word: int) -> int:
        return -1 if self.evaluate_label(input_word) else 1

    def verify_basis_permutation(self) -> bool:
        if self.input_bits > self.max_enumeration_bits:
            raise ValueError("exhaustive verification exceeds max_enumeration_bits")
        for input_word in range(1 << self.input_bits):
            expected = self.evaluate_label(input_word)
            for output_word in (0, 1):
                forward = self.apply(input_word, output_word)
                if forward != (input_word, output_word ^ expected, 0):
                    return False
                if self.inverse_apply(*forward) != (input_word, output_word, 0):
                    return False
        return True

    def resource_estimate(self, *, phase_kickback: bool = False) -> OracleResourceEstimate:
        return _resource_estimate(
            self.circuit,
            input_qubits=self.input_bits,
            output_qubits=1,
            work_qubits=len(self.layout.work_wires),
            synthesis=(
                "two-layer fixed-point MLP threshold oracle: clean fixed-point value "
                "oracle, signed/unsigned integer threshold predicate, and full value cleanup"
            ),
        )


def compile_structure_preserving_fixed_point_mlp_value_oracle(
    hidden_layer: QuantizedAffineLayer,
    output_layer: QuantizedAffineLayer,
    *,
    require_no_overflow: bool = True,
    max_enumeration_bits: int = 16,
) -> ReversibleFixedPointMLPValueOracle:
    return ReversibleFixedPointMLPValueOracle(
        hidden_layer,
        output_layer,
        require_no_overflow=require_no_overflow,
        max_enumeration_bits=max_enumeration_bits,
    )


def compile_structure_preserving_fixed_point_mlp_threshold_oracle(
    hidden_layer: QuantizedAffineLayer,
    output_layer: QuantizedAffineLayer,
    *,
    threshold_code: int = 0,
    require_no_overflow: bool = True,
    max_enumeration_bits: int = 16,
) -> ReversibleFixedPointMLPPredicateOracle:
    return ReversibleFixedPointMLPPredicateOracle(
        hidden_layer,
        output_layer,
        threshold_code=threshold_code,
        require_no_overflow=require_no_overflow,
        max_enumeration_bits=max_enumeration_bits,
    )
