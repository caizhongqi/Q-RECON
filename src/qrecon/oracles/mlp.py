from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

from .arithmetic import (
    AffineRangeReport,
    ReversibleIntegerAffinePredicateOracle,
    ReversibleIntegerAffineValueOracle,
)
from .compiler import OracleResourceEstimate
from .models import QuantizedNetwork
from .reversible import ReversibleCircuit, ReversibleGate


@dataclass(frozen=True)
class ReversibleMLPLayout:
    input_registers: tuple[tuple[int, ...], ...]
    target: int
    hidden_linear_registers: tuple[tuple[int, ...], ...]
    hidden_relu_registers: tuple[tuple[int, ...], ...]
    first_stage_work: tuple[int, ...]
    second_stage_work: tuple[int, ...]

    @property
    def work_wires(self) -> tuple[int, ...]:
        return (
            tuple(wire for register in self.hidden_linear_registers for wire in register)
            + tuple(wire for register in self.hidden_relu_registers for wire in register)
            + self.first_stage_work
            + self.second_stage_work
        )


@dataclass(frozen=True)
class MLPRangeReport:
    first_affine: AffineRangeReport
    second_affine: AffineRangeReport
    no_overflow: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "first_affine": self.first_affine.to_dict(),
            "second_affine": self.second_affine.to_dict(),
            "no_overflow": self.no_overflow,
        }


@dataclass(frozen=True)
class MLPResourceBreakdown:
    first_affine_once: OracleResourceEstimate
    second_predicate_once: OracleResourceEstimate
    hidden_neurons: int
    hidden_bits: int
    relu_x_gates_compute_uncompute: int
    relu_toffoli_gates_compute_uncompute: int
    total: OracleResourceEstimate

    def to_dict(self) -> dict[str, object]:
        return {
            "first_affine_once": self.first_affine_once.to_dict(),
            "second_predicate_once": self.second_predicate_once.to_dict(),
            "hidden_neurons": self.hidden_neurons,
            "hidden_bits": self.hidden_bits,
            "relu_x_gates_compute_uncompute": self.relu_x_gates_compute_uncompute,
            "relu_toffoli_gates_compute_uncompute": self.relu_toffoli_gates_compute_uncompute,
            "total": self.total.to_dict(),
        }


def append_signed_relu_copy(
    circuit: ReversibleCircuit,
    signed_input: Sequence[int],
    unsigned_output: Sequence[int],
) -> tuple[ReversibleGate, ...]:
    """XOR ``max(0, x)`` into a clean same-width output word.

    ``signed_input`` is a little-endian two's-complement register. The output sign
    bit is never toggled. The input sign bit is temporarily inverted to turn the
    non-negative condition into a positive control and is restored before return.
    """

    source = tuple(int(wire) for wire in signed_input)
    target = tuple(int(wire) for wire in unsigned_output)
    if not source or len(source) != len(target):
        raise ValueError("ReLU input and output registers must have equal positive width")
    if len(set(source + target)) != len(source) + len(target):
        raise ValueError("ReLU input and output registers must be disjoint")
    start = len(circuit.gates)
    if len(source) == 1:
        return ()
    sign = source[-1]
    circuit.x(sign)
    for bit in range(len(source) - 1):
        circuit.ccx(sign, source[bit], target[bit])
    circuit.x(sign)
    return circuit.gates[start:]


def _remap_gates(
    gates: Sequence[ReversibleGate], mapping: dict[int, int]
) -> tuple[ReversibleGate, ...]:
    remapped: list[ReversibleGate] = []
    for gate in gates:
        try:
            wires = tuple(mapping[wire] for wire in gate.wires)
        except KeyError as exc:
            raise ValueError(f"missing wire mapping for source wire {exc.args[0]}") from exc
        remapped.append(ReversibleGate(gate.kind, wires))
    return tuple(remapped)


def _allocate_layout(
    feature_count: int,
    input_bits_per_feature: int,
    hidden_neurons: int,
    hidden_bits: int,
    first_work_count: int,
    second_work_count: int,
) -> tuple[ReversibleMLPLayout, int]:
    offset = 0
    inputs: list[tuple[int, ...]] = []
    for _ in range(feature_count):
        inputs.append(tuple(range(offset, offset + input_bits_per_feature)))
        offset += input_bits_per_feature
    target = offset
    offset += 1
    linear: list[tuple[int, ...]] = []
    for _ in range(hidden_neurons):
        linear.append(tuple(range(offset, offset + hidden_bits)))
        offset += hidden_bits
    relu: list[tuple[int, ...]] = []
    for _ in range(hidden_neurons):
        relu.append(tuple(range(offset, offset + hidden_bits)))
        offset += hidden_bits
    first_work = tuple(range(offset, offset + first_work_count))
    offset += first_work_count
    second_work = tuple(range(offset, offset + second_work_count))
    offset += second_work_count
    return (
        ReversibleMLPLayout(
            tuple(inputs),
            target,
            tuple(linear),
            tuple(relu),
            first_work,
            second_work,
        ),
        offset,
    )


def _register_mapping(
    source_registers: Sequence[Sequence[int]],
    target_registers: Sequence[Sequence[int]],
    mapping: dict[int, int],
) -> None:
    if len(source_registers) != len(target_registers):
        raise ValueError("source and target register counts differ")
    for source, target in zip(source_registers, target_registers):
        if len(source) != len(target):
            raise ValueError("source and target register widths differ")
        mapping.update(zip(source, target))


class ReversibleIntegerMLPPredicateOracle:
    """Clean two-layer integer ``Affine -> ReLU -> Affine/Threshold`` oracle.

    The compiler composes two already-clean structure-preserving affine oracles.
    First-layer logits are materialized, ReLU outputs are computed with exact
    sign-controlled Toffolis, the final threshold bit is toggled, and every hidden
    and arithmetic work register is uncomputed by reversing the preceding stages.
    """

    output_bits = 1

    def __init__(
        self,
        first_weights: Sequence[Sequence[int]],
        first_biases: Sequence[int],
        second_weights: Sequence[int],
        second_bias: int,
        *,
        threshold: int = 0,
        input_bits_per_feature: int,
        hidden_bits: int,
        output_accumulator_bits: int,
        signed_inputs: bool = True,
        require_no_overflow: bool = True,
        max_enumeration_bits: int = 16,
    ) -> None:
        first = ReversibleIntegerAffineValueOracle(
            first_weights,
            first_biases,
            input_bits_per_feature=input_bits_per_feature,
            accumulator_bits=hidden_bits,
            signed_inputs=signed_inputs,
            signed_accumulator=True,
            require_no_overflow=require_no_overflow,
        )
        second = ReversibleIntegerAffinePredicateOracle(
            second_weights,
            second_bias,
            threshold=threshold,
            input_bits_per_feature=hidden_bits,
            accumulator_bits=output_accumulator_bits,
            signed_inputs=True,
            require_no_overflow=require_no_overflow,
            max_enumeration_bits=max_enumeration_bits,
        )
        self._initialize(
            first,
            second,
            hidden_bits=hidden_bits,
            max_enumeration_bits=max_enumeration_bits,
        )

    @classmethod
    def from_components(
        cls,
        first: ReversibleIntegerAffineValueOracle,
        second: ReversibleIntegerAffinePredicateOracle,
        *,
        hidden_bits: int,
        max_enumeration_bits: int = 16,
    ) -> "ReversibleIntegerMLPPredicateOracle":
        instance = cls.__new__(cls)
        instance._initialize(
            first,
            second,
            hidden_bits=hidden_bits,
            max_enumeration_bits=max_enumeration_bits,
        )
        return instance

    def _initialize(
        self,
        first: ReversibleIntegerAffineValueOracle,
        second: ReversibleIntegerAffinePredicateOracle,
        *,
        hidden_bits: int,
        max_enumeration_bits: int,
    ) -> None:
        width = int(hidden_bits)
        if width <= 0:
            raise ValueError("hidden_bits must be positive")
        if first.output_bits % width:
            raise ValueError("first affine output width is not divisible by hidden_bits")
        hidden_neurons = first.output_bits // width
        if hidden_neurons <= 0:
            raise ValueError("the hidden layer must contain at least one neuron")
        if second.input_bits != hidden_neurons * width:
            raise ValueError("second predicate input shape does not match hidden activations")
        if second.output_bits != 1:
            raise ValueError("second stage must be a one-bit predicate oracle")
        limit = int(max_enumeration_bits)
        if limit <= 0:
            raise ValueError("max_enumeration_bits must be positive")

        self.first_oracle = first
        self.second_oracle = second
        self.hidden_bits = width
        self.hidden_neurons = hidden_neurons
        self.max_enumeration_bits = limit
        self.feature_count = first.feature_count
        self.input_bits_per_feature = first.input_bits_per_feature
        self.signed_inputs = first.signed_inputs
        self.range_report = MLPRangeReport(
            first_affine=first.range_report,
            second_affine=second.range_report,
            no_overflow=bool(
                first.range_report.no_overflow and second.range_report.no_overflow
            ),
        )

        self.layout, num_qubits = _allocate_layout(
            first.feature_count,
            first.input_bits_per_feature,
            hidden_neurons,
            width,
            len(first.layout.work_wires),
            len(second.layout.work_wires),
        )
        self.circuit = ReversibleCircuit(num_qubits)

        first_mapping: dict[int, int] = {}
        _register_mapping(
            first.layout.input_registers,
            self.layout.input_registers,
            first_mapping,
        )
        _register_mapping(
            first.layout.output_registers,
            self.layout.hidden_linear_registers,
            first_mapping,
        )
        first_mapping.update(zip(first.layout.work_wires, self.layout.first_stage_work))
        first_gates = _remap_gates(first.circuit.gates, first_mapping)
        self.circuit.extend(first_gates)

        relu_start = len(self.circuit.gates)
        for source, target in zip(
            self.layout.hidden_linear_registers,
            self.layout.hidden_relu_registers,
        ):
            append_signed_relu_copy(self.circuit, source, target)
        relu_gates = self.circuit.gates[relu_start:]

        second_mapping: dict[int, int] = {second.layout.target: self.layout.target}
        _register_mapping(
            second.layout.input_registers,
            self.layout.hidden_relu_registers,
            second_mapping,
        )
        second_mapping.update(zip(second.layout.work_wires, self.layout.second_stage_work))
        second_gates = _remap_gates(second.circuit.gates, second_mapping)
        self.circuit.extend(second_gates)
        self.circuit.append_inverse(relu_gates)
        self.circuit.append_inverse(first_gates)

    @property
    def input_bits(self) -> int:
        return self.first_oracle.input_bits

    def encode_inputs(self, values: Sequence[int]) -> int:
        return self.first_oracle.encode_inputs(values)

    def decode_input_word(self, input_word: int) -> tuple[int, ...]:
        return self.first_oracle.decode_input_word(input_word)

    def hidden_activations(self, input_word: int) -> tuple[int, ...]:
        packed = self.first_oracle.evaluate_input_word(input_word)
        mask = (1 << self.hidden_bits) - 1
        values: list[int] = []
        for index in range(self.hidden_neurons):
            word = (packed >> (index * self.hidden_bits)) & mask
            signed = word - (1 << self.hidden_bits) if word >> (self.hidden_bits - 1) else word
            values.append(max(0, signed))
        return tuple(values)

    def evaluate_predicate(self, input_word: int) -> int:
        activation_word = 0
        for index, value in enumerate(self.hidden_activations(input_word)):
            activation_word |= value << (index * self.hidden_bits)
        return self.second_oracle.evaluate_predicate(activation_word)

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
            raise ValueError(
                "marked-input enumeration is disabled above max_enumeration_bits"
            )
        return tuple(
            word
            for word in range(1 << self.input_bits)
            if self.evaluate_predicate(word)
        )

    def phase_sign(self, input_word: int) -> int:
        predicate = self.apply(input_word, 0)[1]
        return -1 if predicate else 1

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
        counts = self.circuit.gate_counts()
        toffoli = counts["ccx"]
        return OracleResourceEstimate(
            input_qubits=self.input_bits,
            output_qubits=1,
            peak_clean_ancillas=len(self.layout.work_wires),
            logical_qubits=self.circuit.num_qubits,
            controlled_x_terms=counts["cx"] + counts["ccx"],
            negative_control_x_gates=0,
            x_gates=counts["x"],
            cnot_gates=counts["cx"],
            toffoli_gates=toffoli,
            h_gates=0,
            z_gates=0,
            t_count_upper_bound=7 * toffoli,
            t_depth_upper_bound=3 * toffoli,
            logical_depth_upper_bound=self.circuit.logical_depth(),
            synthesis=(
                "two-layer structure-preserving integer MLP: clean affine value "
                "oracle, exact sign-controlled ReLU copy, clean affine-threshold "
                "predicate, and Bennett reverse uncomputation"
            ),
        )

    def resource_breakdown(self) -> MLPResourceBreakdown:
        first = self.first_oracle.resource_estimate()
        second = self.second_oracle.resource_estimate(phase_kickback=True)
        total = self.resource_estimate(phase_kickback=True)
        return MLPResourceBreakdown(
            first_affine_once=first,
            second_predicate_once=second,
            hidden_neurons=self.hidden_neurons,
            hidden_bits=self.hidden_bits,
            relu_x_gates_compute_uncompute=(
                4 * self.hidden_neurons if self.hidden_bits > 1 else 0
            ),
            relu_toffoli_gates_compute_uncompute=(
                2 * self.hidden_neurons * max(0, self.hidden_bits - 1)
            ),
            total=total,
        )


def compile_structure_preserving_mlp_threshold_oracle(
    model: QuantizedNetwork,
    *,
    require_no_overflow: bool = True,
    max_enumeration_bits: int = 16,
) -> ReversibleIntegerMLPPredicateOracle:
    """Lower a two-layer integer ``Affine-ReLU-Affine`` binary network."""

    if len(model.layers) != 2:
        raise ValueError("MLP lowering requires exactly two affine layers")
    first, second = model.layers
    if first.activation != "relu" or second.activation != "identity":
        raise ValueError("MLP lowering requires ReLU then identity activations")
    if model.output_mode != "binary_threshold" or second.output_dimension != 1:
        raise ValueError("MLP lowering requires one binary-threshold output")
    formats = (
        first.input_format,
        first.weight_format,
        first.bias_format,
        first.output_format,
        second.input_format,
        second.weight_format,
        second.bias_format,
        second.output_format,
    )
    if any(fmt.fractional_bits != 0 for fmt in formats):
        raise ValueError("fractional fixed-point MLP lowering is not yet supported")
    if not first.output_format.signed or not second.output_format.signed:
        raise ValueError("ReLU and final sign threshold require signed layer outputs")
    return ReversibleIntegerMLPPredicateOracle(
        first.weights,
        first.biases,
        second.weights[0],
        second.biases[0],
        threshold=model.binary_threshold,
        input_bits_per_feature=first.input_format.bits,
        hidden_bits=first.output_format.bits,
        output_accumulator_bits=second.output_format.bits,
        signed_inputs=first.input_format.signed,
        require_no_overflow=require_no_overflow,
        max_enumeration_bits=max_enumeration_bits,
    )
