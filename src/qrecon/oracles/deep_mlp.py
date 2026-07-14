from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .arithmetic import (
    AffineRangeReport,
    ReversibleIntegerAffinePredicateOracle,
    ReversibleIntegerAffineValueOracle,
)
from .compiler import OracleResourceEstimate
from .mlp import append_signed_relu_copy
from .models import QuantizedNetwork
from .reversible import ReversibleCircuit, ReversibleGate


@dataclass(frozen=True)
class DeepMLPHiddenLayout:
    preactivation_registers: tuple[tuple[int, ...], ...]
    activation_registers: tuple[tuple[int, ...], ...]

    @property
    def wires(self) -> tuple[int, ...]:
        return tuple(
            wire
            for register in self.preactivation_registers + self.activation_registers
            for wire in register
        )


@dataclass(frozen=True)
class DeepMLPLayout:
    input_registers: tuple[tuple[int, ...], ...]
    target: int
    hidden_layers: tuple[DeepMLPHiddenLayout, ...]
    shared_arithmetic_work: tuple[int, ...]

    @property
    def work_wires(self) -> tuple[int, ...]:
        hidden = tuple(
            wire for layer in self.hidden_layers for wire in layer.wires
        )
        return hidden + self.shared_arithmetic_work


@dataclass(frozen=True)
class DeepMLPRangeReport:
    hidden_affine: tuple[AffineRangeReport, ...]
    final_affine: AffineRangeReport
    no_overflow: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "hidden_affine": [report.to_dict() for report in self.hidden_affine],
            "final_affine": self.final_affine.to_dict(),
            "no_overflow": self.no_overflow,
        }


@dataclass(frozen=True)
class DeepMLPResourceBreakdown:
    hidden_affine_once: tuple[OracleResourceEstimate, ...]
    final_predicate_once: OracleResourceEstimate
    relu_x_gates_compute_uncompute: int
    relu_toffoli_gates_compute_uncompute: int
    shared_arithmetic_work_qubits: int
    total: OracleResourceEstimate

    def to_dict(self) -> dict[str, object]:
        return {
            "hidden_affine_once": [item.to_dict() for item in self.hidden_affine_once],
            "final_predicate_once": self.final_predicate_once.to_dict(),
            "relu_x_gates_compute_uncompute": self.relu_x_gates_compute_uncompute,
            "relu_toffoli_gates_compute_uncompute": self.relu_toffoli_gates_compute_uncompute,
            "shared_arithmetic_work_qubits": self.shared_arithmetic_work_qubits,
            "total": self.total.to_dict(),
        }


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


def _map_registers(
    source: Sequence[Sequence[int]],
    target: Sequence[Sequence[int]],
    mapping: dict[int, int],
) -> None:
    if len(source) != len(target):
        raise ValueError("source and target register counts differ")
    for left, right in zip(source, target):
        if len(left) != len(right):
            raise ValueError("source and target register widths differ")
        mapping.update(zip(left, right))


def _allocate_layout(
    input_features: int,
    input_bits: int,
    hidden_shapes: Sequence[tuple[int, int]],
    shared_work: int,
) -> tuple[DeepMLPLayout, int]:
    offset = 0
    inputs: list[tuple[int, ...]] = []
    for _ in range(input_features):
        inputs.append(tuple(range(offset, offset + input_bits)))
        offset += input_bits
    target = offset
    offset += 1
    layers: list[DeepMLPHiddenLayout] = []
    for neurons, width in hidden_shapes:
        preactivation: list[tuple[int, ...]] = []
        activation: list[tuple[int, ...]] = []
        for _ in range(neurons):
            preactivation.append(tuple(range(offset, offset + width)))
            offset += width
        for _ in range(neurons):
            activation.append(tuple(range(offset, offset + width)))
            offset += width
        layers.append(DeepMLPHiddenLayout(tuple(preactivation), tuple(activation)))
    work = tuple(range(offset, offset + shared_work))
    offset += shared_work
    return DeepMLPLayout(tuple(inputs), target, tuple(layers), work), offset


class ReversibleIntegerDeepMLPPredicateOracle:
    """Clean arbitrary-depth integer ReLU MLP predicate with shared work reuse."""

    output_bits = 1

    def __init__(
        self,
        hidden_weights: Sequence[Sequence[Sequence[int]]],
        hidden_biases: Sequence[Sequence[int]],
        final_weights: Sequence[int],
        final_bias: int,
        *,
        threshold: int = 0,
        input_bits_per_feature: int,
        hidden_bits: Sequence[int],
        output_accumulator_bits: int,
        signed_inputs: bool = True,
        require_no_overflow: bool = True,
        max_enumeration_bits: int = 16,
    ) -> None:
        weight_layers = tuple(
            tuple(tuple(int(weight) for weight in row) for row in layer)
            for layer in hidden_weights
        )
        bias_layers = tuple(
            tuple(int(bias) for bias in layer) for layer in hidden_biases
        )
        widths = tuple(int(width) for width in hidden_bits)
        if not weight_layers or len(weight_layers) != len(bias_layers):
            raise ValueError("one hidden bias vector is required per hidden weight matrix")
        if len(widths) != len(weight_layers) or any(width <= 0 for width in widths):
            raise ValueError("one positive word width is required per hidden layer")
        if input_bits_per_feature <= 0 or output_accumulator_bits <= 1:
            raise ValueError("input and final accumulator widths must be positive")
        limit = int(max_enumeration_bits)
        if limit <= 0:
            raise ValueError("max_enumeration_bits must be positive")

        affine_oracles: list[ReversibleIntegerAffineValueOracle] = []
        current_bits = int(input_bits_per_feature)
        current_signed = bool(signed_inputs)
        previous_outputs: int | None = None
        for layer_index, (weights, biases, width) in enumerate(
            zip(weight_layers, bias_layers, widths)
        ):
            if not weights or not weights[0]:
                raise ValueError("every hidden weight matrix must be non-empty")
            if previous_outputs is not None and len(weights[0]) != previous_outputs:
                raise ValueError(
                    f"hidden layer {layer_index} input dimension does not match "
                    "the preceding activation count"
                )
            oracle = ReversibleIntegerAffineValueOracle(
                weights,
                biases,
                input_bits_per_feature=current_bits,
                accumulator_bits=width,
                signed_inputs=current_signed,
                signed_accumulator=True,
                require_no_overflow=require_no_overflow,
            )
            affine_oracles.append(oracle)
            previous_outputs = oracle.output_count
            current_bits = width
            current_signed = True

        final = ReversibleIntegerAffinePredicateOracle(
            final_weights,
            final_bias,
            threshold=threshold,
            input_bits_per_feature=widths[-1],
            accumulator_bits=output_accumulator_bits,
            signed_inputs=True,
            require_no_overflow=require_no_overflow,
            max_enumeration_bits=max_enumeration_bits,
        )
        if final.feature_count != affine_oracles[-1].output_count:
            raise ValueError("final predicate input dimension does not match last hidden layer")

        self.hidden_oracles = tuple(affine_oracles)
        self.final_oracle = final
        self.hidden_bits = widths
        self.hidden_neurons = tuple(oracle.output_count for oracle in affine_oracles)
        self.feature_count = affine_oracles[0].feature_count
        self.input_bits_per_feature = affine_oracles[0].input_bits_per_feature
        self.signed_inputs = affine_oracles[0].signed_inputs
        self.max_enumeration_bits = limit
        reports = tuple(oracle.range_report for oracle in affine_oracles)
        self.range_report = DeepMLPRangeReport(
            hidden_affine=reports,
            final_affine=final.range_report,
            no_overflow=all(report.no_overflow for report in reports)
            and final.range_report.no_overflow,
        )

        shared_work = max(
            [len(oracle.layout.work_wires) for oracle in affine_oracles]
            + [len(final.layout.work_wires)]
        )
        shapes = tuple(zip(self.hidden_neurons, self.hidden_bits))
        self.layout, num_qubits = _allocate_layout(
            self.feature_count,
            self.input_bits_per_feature,
            shapes,
            shared_work,
        )
        self.circuit = ReversibleCircuit(num_qubits)
        source_registers = self.layout.input_registers
        stage_gates: list[tuple[tuple[ReversibleGate, ...], tuple[ReversibleGate, ...]]] = []

        for oracle, layer_layout in zip(self.hidden_oracles, self.layout.hidden_layers):
            mapping: dict[int, int] = {}
            _map_registers(oracle.layout.input_registers, source_registers, mapping)
            _map_registers(
                oracle.layout.output_registers,
                layer_layout.preactivation_registers,
                mapping,
            )
            mapping.update(
                zip(oracle.layout.work_wires, self.layout.shared_arithmetic_work)
            )
            affine_gates = _remap_gates(oracle.circuit.gates, mapping)
            self.circuit.extend(affine_gates)
            relu_start = len(self.circuit.gates)
            for source, target in zip(
                layer_layout.preactivation_registers,
                layer_layout.activation_registers,
            ):
                append_signed_relu_copy(self.circuit, source, target)
            relu_gates = self.circuit.gates[relu_start:]
            stage_gates.append((affine_gates, relu_gates))
            source_registers = layer_layout.activation_registers

        final_mapping: dict[int, int] = {final.layout.target: self.layout.target}
        _map_registers(final.layout.input_registers, source_registers, final_mapping)
        final_mapping.update(
            zip(final.layout.work_wires, self.layout.shared_arithmetic_work)
        )
        self.circuit.extend(_remap_gates(final.circuit.gates, final_mapping))

        for affine_gates, relu_gates in reversed(stage_gates):
            self.circuit.append_inverse(relu_gates)
            self.circuit.append_inverse(affine_gates)

    @property
    def input_bits(self) -> int:
        return self.hidden_oracles[0].input_bits

    def encode_inputs(self, values: Sequence[int]) -> int:
        return self.hidden_oracles[0].encode_inputs(values)

    def decode_input_word(self, input_word: int) -> tuple[int, ...]:
        return self.hidden_oracles[0].decode_input_word(input_word)

    @staticmethod
    def _decode_relu_outputs(packed: int, neurons: int, width: int) -> tuple[int, ...]:
        mask = (1 << width) - 1
        outputs: list[int] = []
        for index in range(neurons):
            word = (packed >> (index * width)) & mask
            signed = word - (1 << width) if word >> (width - 1) else word
            outputs.append(max(0, signed))
        return tuple(outputs)

    def hidden_activations(self, input_word: int) -> tuple[tuple[int, ...], ...]:
        current_word = int(input_word)
        layers: list[tuple[int, ...]] = []
        for oracle, neurons, width in zip(
            self.hidden_oracles, self.hidden_neurons, self.hidden_bits
        ):
            packed = oracle.evaluate_input_word(current_word)
            values = self._decode_relu_outputs(packed, neurons, width)
            layers.append(values)
            current_word = sum(
                value << (index * width) for index, value in enumerate(values)
            )
        return tuple(layers)

    def evaluate_predicate(self, input_word: int) -> int:
        activations = self.hidden_activations(input_word)[-1]
        width = self.hidden_bits[-1]
        packed = sum(
            value << (index * width) for index, value in enumerate(activations)
        )
        return self.final_oracle.evaluate_predicate(packed)

    def _pack_state(self, input_word: int, target: int) -> int:
        value = int(input_word)
        bit = int(target)
        if value < 0 or value >= (1 << self.input_bits):
            raise ValueError("input_word is outside the candidate register")
        if bit not in (0, 1):
            raise ValueError("target must be a single bit")
        return value | (bit << self.input_bits)

    def _extract(self, state: int) -> tuple[int, int, int]:
        mask = (1 << self.input_bits) - 1
        return state & mask, (state >> self.input_bits) & 1, state >> (self.input_bits + 1)

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
            word for word in range(1 << self.input_bits) if self.evaluate_predicate(word)
        )

    def phase_sign(self, input_word: int) -> int:
        return -1 if self.apply(input_word, 0)[1] else 1

    def verify_basis_permutation(self) -> bool:
        if self.input_bits > self.max_enumeration_bits:
            raise ValueError("exhaustive verification exceeds max_enumeration_bits")
        for input_word in range(1 << self.input_bits):
            expected = self.evaluate_predicate(input_word)
            for target in (0, 1):
                forward = self.apply(input_word, target)
                if forward != (input_word, target ^ expected, 0):
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
                "arbitrary-depth structure-preserving integer ReLU MLP with shared "
                "clean arithmetic work, exact layerwise gate remapping, final "
                "threshold phase predicate, and reverse liveness cleanup"
            ),
        )

    def resource_breakdown(self) -> DeepMLPResourceBreakdown:
        hidden = tuple(oracle.resource_estimate() for oracle in self.hidden_oracles)
        final = self.final_oracle.resource_estimate(phase_kickback=True)
        relu_x = sum(
            4 * neurons if width > 1 else 0
            for neurons, width in zip(self.hidden_neurons, self.hidden_bits)
        )
        relu_toffoli = sum(
            2 * neurons * max(0, width - 1)
            for neurons, width in zip(self.hidden_neurons, self.hidden_bits)
        )
        return DeepMLPResourceBreakdown(
            hidden_affine_once=hidden,
            final_predicate_once=final,
            relu_x_gates_compute_uncompute=relu_x,
            relu_toffoli_gates_compute_uncompute=relu_toffoli,
            shared_arithmetic_work_qubits=len(self.layout.shared_arithmetic_work),
            total=self.resource_estimate(phase_kickback=True),
        )


def compile_structure_preserving_deep_mlp_threshold_oracle(
    model: QuantizedNetwork,
    *,
    require_no_overflow: bool = True,
    max_enumeration_bits: int = 16,
) -> ReversibleIntegerDeepMLPPredicateOracle:
    """Lower an integer ReLU MLP with one or more hidden layers."""

    if len(model.layers) < 2:
        raise ValueError("deep MLP lowering requires at least one hidden layer")
    hidden = model.layers[:-1]
    final = model.layers[-1]
    if any(layer.activation != "relu" for layer in hidden):
        raise ValueError("every hidden layer must use ReLU")
    if final.activation != "identity":
        raise ValueError("the final layer must use identity activation")
    if model.output_mode != "binary_threshold" or final.output_dimension != 1:
        raise ValueError("deep MLP lowering requires one binary-threshold output")
    formats = tuple(
        fmt
        for layer in model.layers
        for fmt in (
            layer.input_format,
            layer.weight_format,
            layer.bias_format,
            layer.output_format,
        )
    )
    if any(fmt.fractional_bits != 0 for fmt in formats):
        raise ValueError("fractional fixed-point deep MLP lowering is not yet supported")
    if any(not layer.output_format.signed for layer in model.layers):
        raise ValueError("all hidden and final outputs must use signed words")
    return ReversibleIntegerDeepMLPPredicateOracle(
        tuple(layer.weights for layer in hidden),
        tuple(layer.biases for layer in hidden),
        final.weights[0],
        final.biases[0],
        threshold=model.binary_threshold,
        input_bits_per_feature=hidden[0].input_format.bits,
        hidden_bits=tuple(layer.output_format.bits for layer in hidden),
        output_accumulator_bits=final.output_format.bits,
        signed_inputs=hidden[0].input_format.signed,
        require_no_overflow=require_no_overflow,
        max_enumeration_bits=max_enumeration_bits,
    )
