from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

from .arithmetic import ReversibleIntegerAffinePredicateOracle, _resource_estimate
from .comparators import append_equality_to_constant
from .compiler import OracleResourceEstimate
from .fixed_point_affine import ReversibleFixedPointAffineValueOracle
from .fixed_point_mlp import ReversibleFixedPointAffineReLUValueOracle, _remap_gates
from .models import NetworkRangeReport, QuantizedAffineLayer, QuantizedNetwork
from .reversible import ReversibleCircuit, ReversibleGate


@dataclass(frozen=True)
class FixedPointDeepMLPReachabilityCertificate:
    """Layerwise range proof for an arbitrary-depth fixed-point network."""

    layer_raw_bounds: tuple[tuple[tuple[int, int], ...], ...]
    layer_encoded_bounds: tuple[tuple[tuple[int, int], ...], ...]
    no_overflow: bool

    @classmethod
    def from_network_report(
        cls, report: NetworkRangeReport
    ) -> "FixedPointDeepMLPReachabilityCertificate":
        if not report.layer_reports:
            raise ValueError("a reachability certificate requires at least one layer")
        return cls(
            layer_raw_bounds=tuple(item.raw_output_bounds for item in report.layer_reports),
            layer_encoded_bounds=tuple(
                item.encoded_output_bounds for item in report.layer_reports
            ),
            no_overflow=report.no_overflow,
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FixedPointDeepMLPLayout:
    input_wires: tuple[int, ...]
    output_wires: tuple[int, ...]
    hidden_wires: tuple[tuple[int, ...], ...]
    shared_work: tuple[int, ...]

    @property
    def work_wires(self) -> tuple[int, ...]:
        hidden = tuple(wire for register in self.hidden_wires for wire in register)
        return hidden + self.shared_work


@dataclass(frozen=True)
class FixedPointDeepMLPResourceBreakdown:
    """Exact layer multiplicities plus the synthesized total resource record."""

    layer_once: tuple[OracleResourceEstimate, ...]
    layer_multiplicity: tuple[int, ...]
    hidden_register_qubits: int
    shared_work_qubits: int
    total: OracleResourceEstimate

    def to_dict(self) -> dict[str, object]:
        return {
            "layer_once": [item.to_dict() for item in self.layer_once],
            "layer_multiplicity": list(self.layer_multiplicity),
            "hidden_register_qubits": self.hidden_register_qubits,
            "shared_work_qubits": self.shared_work_qubits,
            "total": self.total.to_dict(),
        }


class ReversibleFixedPointDeepMLPValueOracle:
    """Clean arbitrary-depth fixed-point affine/ReLU network value oracle.

    Every non-final layer is computed into a retained activation register. The
    final layer XORs its bit-exact value into the public output register, after
    which the hidden stages are reversed. Component arithmetic work is reused
    because every layer oracle is clean on entry and exit.
    """

    def __init__(
        self,
        layers: Sequence[QuantizedAffineLayer],
        *,
        require_no_overflow: bool = True,
        max_enumeration_bits: int = 16,
    ) -> None:
        declared = tuple(layers)
        if not declared:
            raise ValueError("at least one fixed-point layer is required")
        if declared[-1].activation != "identity":
            raise ValueError("the final fixed-point layer must use identity activation")
        limit = int(max_enumeration_bits)
        if limit <= 0:
            raise ValueError("max_enumeration_bits must be positive")

        self.layers = declared
        self.max_enumeration_bits = limit
        self.network = QuantizedNetwork(declared, output_mode="raw")
        self.network_range_report = self.network.range_report()
        self.reachability_certificate = (
            FixedPointDeepMLPReachabilityCertificate.from_network_report(
                self.network_range_report
            )
        )
        if require_no_overflow and not self.reachability_certificate.no_overflow:
            raise OverflowError(
                "reachable fixed-point network values exceed a declared layer format; "
                f"certificate={self.reachability_certificate.to_dict()}"
            )

        compiled: list[
            ReversibleFixedPointAffineValueOracle
            | ReversibleFixedPointAffineReLUValueOracle
        ] = []
        for layer in declared:
            if layer.activation == "relu":
                oracle = ReversibleFixedPointAffineReLUValueOracle(
                    layer,
                    require_no_overflow=False,
                    max_enumeration_bits=limit,
                )
            else:
                oracle = ReversibleFixedPointAffineValueOracle(
                    layer,
                    require_no_overflow=False,
                    max_enumeration_bits=limit,
                )
            compiled.append(oracle)
        self.layer_oracles = tuple(compiled)

        input_bits = self.layer_oracles[0].input_bits
        output_bits = self.layer_oracles[-1].output_bits
        offset = input_bits
        output_wires = tuple(range(offset, offset + output_bits))
        offset += output_bits

        hidden_wires: list[tuple[int, ...]] = []
        for oracle in self.layer_oracles[:-1]:
            register = tuple(range(offset, offset + oracle.output_bits))
            offset += len(register)
            hidden_wires.append(register)

        shared_work_size = max(
            len(oracle.layout.work_wires) for oracle in self.layer_oracles
        )
        shared_work = tuple(range(offset, offset + shared_work_size))
        offset += len(shared_work)
        self.layout = FixedPointDeepMLPLayout(
            input_wires=tuple(range(input_bits)),
            output_wires=output_wires,
            hidden_wires=tuple(hidden_wires),
            shared_work=shared_work,
        )
        self.circuit = ReversibleCircuit(offset)

        source_wires = self.layout.input_wires
        hidden_stage_gates: list[tuple[ReversibleGate, ...]] = []
        for index, oracle in enumerate(self.layer_oracles):
            target_wires = (
                self.layout.output_wires
                if index == len(self.layer_oracles) - 1
                else self.layout.hidden_wires[index]
            )
            mapping: dict[int, int] = {}
            mapping.update(zip(oracle.layout.input_wires, source_wires))
            mapping.update(zip(oracle.layout.output_wires, target_wires))
            mapping.update(zip(oracle.layout.work_wires, self.layout.shared_work))
            gates = _remap_gates(oracle.circuit.gates, mapping)
            self.circuit.extend(gates)
            if index < len(self.layer_oracles) - 1:
                hidden_stage_gates.append(gates)
                source_wires = target_wires

        for gates in reversed(hidden_stage_gates):
            self.circuit.append_inverse(gates)

    @property
    def input_bits(self) -> int:
        return self.network.input_bits

    @property
    def output_bits(self) -> int:
        return self.network.output_bits

    def encode_inputs(self, values: Sequence[int]) -> int:
        return self.network.encode_input_codes(values)

    def decode_input_word(self, input_word: int) -> tuple[int, ...]:
        return self.network.decode_input_word(input_word)

    def evaluate_input_word(self, input_word: int) -> int:
        return self.network.evaluate_input_word(input_word)

    def _pack_state(self, input_word: int, output_word: int) -> int:
        candidate = int(input_word)
        target = int(output_word)
        if candidate < 0 or candidate >= (1 << self.input_bits):
            raise ValueError("input_word is outside the input register")
        if target < 0 or target >= (1 << self.output_bits):
            raise ValueError("output_word is outside the output register")
        return candidate | (target << self.input_bits)

    def _extract(self, state: int) -> tuple[int, int, int]:
        return (
            state & ((1 << self.input_bits) - 1),
            (state >> self.input_bits) & ((1 << self.output_bits) - 1),
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
            self.circuit.apply_inverse_state(
                self._pack_state(input_word, output_word)
            )
        )

    def verify_basis_permutation(self) -> bool:
        if self.input_bits > self.max_enumeration_bits:
            raise ValueError("exhaustive verification exceeds max_enumeration_bits")
        target_words = {0, (1 << self.output_bits) - 1}
        if self.output_bits > 1:
            target_words.add(1)
        for input_word in range(1 << self.input_bits):
            expected = self.evaluate_input_word(input_word)
            for output_word in target_words:
                forward = self.apply(input_word, output_word)
                if forward != (input_word, output_word ^ expected, 0):
                    return False
                if self.inverse_apply(*forward) != (input_word, output_word, 0):
                    return False
        return True

    def resource_estimate(
        self, *, phase_kickback: bool = False
    ) -> OracleResourceEstimate:
        if phase_kickback:
            raise ValueError("phase kickback requires a one-bit predicate")
        return _resource_estimate(
            self.circuit,
            input_qubits=self.input_bits,
            output_qubits=self.output_bits,
            work_qubits=len(self.layout.work_wires),
            synthesis=(
                f"{len(self.layers)}-layer fixed-point affine/ReLU value oracle: "
                "layerwise clean arithmetic, shared arithmetic work, retained hidden "
                "activations, and reverse hidden-stage cleanup"
            ),
        )

    def resource_breakdown(self) -> FixedPointDeepMLPResourceBreakdown:
        once = tuple(oracle.resource_estimate() for oracle in self.layer_oracles)
        multiplicity = tuple(
            1 if index == len(self.layer_oracles) - 1 else 2
            for index in range(len(self.layer_oracles))
        )
        return FixedPointDeepMLPResourceBreakdown(
            layer_once=once,
            layer_multiplicity=multiplicity,
            hidden_register_qubits=sum(map(len, self.layout.hidden_wires)),
            shared_work_qubits=len(self.layout.shared_work),
            total=self.resource_estimate(),
        )


@dataclass(frozen=True)
class FixedPointDeepMLPPredicateLayout:
    input_wires: tuple[int, ...]
    target: int
    value_wires: tuple[int, ...]
    value_work: tuple[int, ...]
    predicate_work: tuple[int, ...]

    @property
    def work_wires(self) -> tuple[int, ...]:
        return self.value_wires + self.value_work + self.predicate_work


class ReversibleFixedPointDeepMLPPredicateOracle:
    """Clean threshold predicate for a one-output arbitrary-depth network."""

    output_bits = 1

    def __init__(
        self,
        layers: Sequence[QuantizedAffineLayer],
        *,
        threshold_code: int = 0,
        require_no_overflow: bool = True,
        max_enumeration_bits: int = 16,
    ) -> None:
        declared = tuple(layers)
        if not declared or declared[-1].output_dimension != 1:
            raise ValueError("threshold lowering requires one final output")
        declared[-1].output_format.require_code(threshold_code)
        self.threshold_code = int(threshold_code)
        self.value = ReversibleFixedPointDeepMLPValueOracle(
            declared,
            require_no_overflow=require_no_overflow,
            max_enumeration_bits=max_enumeration_bits,
        )
        self.output_format = declared[-1].output_format
        self.max_enumeration_bits = int(max_enumeration_bits)
        self.predicate = ReversibleIntegerAffinePredicateOracle(
            (1,),
            0,
            input_bits_per_feature=self.output_format.bits,
            accumulator_bits=self.output_format.bits + 1,
            threshold=self.threshold_code,
            signed_inputs=self.output_format.signed,
            require_no_overflow=True,
        )

        offset = self.value.input_bits
        target = offset
        offset += 1
        value_wires = tuple(range(offset, offset + self.value.output_bits))
        offset += len(value_wires)
        value_work = tuple(
            range(offset, offset + len(self.value.layout.work_wires))
        )
        offset += len(value_work)
        predicate_work = tuple(
            range(offset, offset + len(self.predicate.layout.work_wires))
        )
        offset += len(predicate_work)
        self.layout = FixedPointDeepMLPPredicateLayout(
            tuple(range(self.value.input_bits)),
            target,
            value_wires,
            value_work,
            predicate_work,
        )
        self.circuit = ReversibleCircuit(offset)

        value_mapping: dict[int, int] = {}
        value_mapping.update(zip(self.value.layout.input_wires, self.layout.input_wires))
        value_mapping.update(zip(self.value.layout.output_wires, self.layout.value_wires))
        value_mapping.update(zip(self.value.layout.work_wires, self.layout.value_work))
        value_gates = _remap_gates(self.value.circuit.gates, value_mapping)

        predicate_mapping: dict[int, int] = {}
        predicate_mapping.update(
            zip(self.predicate.layout.input_registers[0], self.layout.value_wires)
        )
        predicate_mapping[self.predicate.layout.target] = self.layout.target
        predicate_mapping.update(
            zip(self.predicate.layout.work_wires, self.layout.predicate_work)
        )
        predicate_gates = _remap_gates(
            self.predicate.circuit.gates, predicate_mapping
        )

        self.circuit.extend(value_gates)
        self.circuit.extend(predicate_gates)
        self.circuit.append_inverse(value_gates)

    @property
    def input_bits(self) -> int:
        return self.value.input_bits

    def encode_inputs(self, values: Sequence[int]) -> int:
        return self.value.encode_inputs(values)

    def evaluate_label(self, input_word: int) -> int:
        logit_word = self.value.evaluate_input_word(input_word)
        logit = self.output_format.word_to_code(logit_word)
        return int(logit >= self.threshold_code)

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

    def resource_estimate(
        self, *, phase_kickback: bool = False
    ) -> OracleResourceEstimate:
        return _resource_estimate(
            self.circuit,
            input_qubits=self.input_bits,
            output_qubits=1,
            work_qubits=len(self.layout.work_wires),
            synthesis=(
                f"{len(self.value.layers)}-layer fixed-point threshold oracle: "
                "clean deep value computation, signed threshold predicate, and "
                "complete value uncomputation"
            ),
        )


@dataclass(frozen=True)
class FixedPointDeepMLPEqualityLayout:
    input_wires: tuple[int, ...]
    target: int
    value_wires: tuple[int, ...]
    value_work: tuple[int, ...]
    equality_work: tuple[int, ...]

    @property
    def work_wires(self) -> tuple[int, ...]:
        return self.value_wires + self.value_work + self.equality_work


class ReversibleFixedPointDeepMLPEqualityOracle:
    """Clean exact-output verifier for an arbitrary-depth fixed-point network."""

    output_bits = 1

    def __init__(
        self,
        layers: Sequence[QuantizedAffineLayer],
        target_codes: Sequence[int],
        *,
        require_no_overflow: bool = True,
        max_enumeration_bits: int = 16,
    ) -> None:
        declared = tuple(layers)
        if not declared:
            raise ValueError("at least one fixed-point layer is required")
        target = tuple(int(value) for value in target_codes)
        final = declared[-1]
        if len(target) != final.output_dimension:
            raise ValueError("one target code is required per final output")
        for value in target:
            final.output_format.require_code(value)

        self.value = ReversibleFixedPointDeepMLPValueOracle(
            declared,
            require_no_overflow=require_no_overflow,
            max_enumeration_bits=max_enumeration_bits,
        )
        self.layers = declared
        self.target_codes = target
        self.max_enumeration_bits = int(max_enumeration_bits)
        target_word = 0
        for index, code in enumerate(target):
            target_word |= final.output_format.code_to_word(code) << (
                index * final.output_format.bits
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
        self.layout = FixedPointDeepMLPEqualityLayout(
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
        return -1 if self.evaluate_predicate(input_word) else 1

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
                f"{len(self.layers)}-layer fixed-point exact-output oracle: "
                "clean deep value computation, packed constant equality, and "
                "complete value uncomputation"
            ),
        )


def compile_structure_preserving_fixed_point_deep_mlp_value_oracle(
    layers: Sequence[QuantizedAffineLayer],
    *,
    require_no_overflow: bool = True,
    max_enumeration_bits: int = 16,
) -> ReversibleFixedPointDeepMLPValueOracle:
    return ReversibleFixedPointDeepMLPValueOracle(
        layers,
        require_no_overflow=require_no_overflow,
        max_enumeration_bits=max_enumeration_bits,
    )


def compile_structure_preserving_fixed_point_deep_mlp_threshold_oracle(
    layers: Sequence[QuantizedAffineLayer],
    *,
    threshold_code: int = 0,
    require_no_overflow: bool = True,
    max_enumeration_bits: int = 16,
) -> ReversibleFixedPointDeepMLPPredicateOracle:
    return ReversibleFixedPointDeepMLPPredicateOracle(
        layers,
        threshold_code=threshold_code,
        require_no_overflow=require_no_overflow,
        max_enumeration_bits=max_enumeration_bits,
    )


def compile_structure_preserving_fixed_point_deep_mlp_equality_oracle(
    layers: Sequence[QuantizedAffineLayer],
    target_codes: Sequence[int],
    *,
    require_no_overflow: bool = True,
    max_enumeration_bits: int = 16,
) -> ReversibleFixedPointDeepMLPEqualityOracle:
    return ReversibleFixedPointDeepMLPEqualityOracle(
        layers,
        target_codes,
        require_no_overflow=require_no_overflow,
        max_enumeration_bits=max_enumeration_bits,
    )
