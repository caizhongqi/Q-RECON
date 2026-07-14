from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .arithmetic import ReversibleIntegerAffineValueOracle, _resource_estimate
from .compiler import OracleResourceEstimate
from .fixed_point import FixedPointFormat, rescale_code
from .models import QuantizedAffineLayer
from .requantization import ReversibleFixedPointRequantizationOracle
from .reversible import ReversibleCircuit, ReversibleGate


@dataclass(frozen=True)
class FixedPointAffineLayout:
    input_wires: tuple[int, ...]
    output_wires: tuple[int, ...]
    raw_output_wires: tuple[int, ...]
    affine_work: tuple[int, ...]
    requantization_work: tuple[int, ...]

    @property
    def work_wires(self) -> tuple[int, ...]:
        return self.raw_output_wires + self.affine_work + self.requantization_work


def _bits_for_signed_range(minimum: int, maximum: int) -> int:
    bits = 2
    while minimum < -(1 << (bits - 1)) or maximum > (1 << (bits - 1)) - 1:
        bits += 1
    return bits


def _remap_gates(
    gates: Sequence[ReversibleGate], mapping: dict[int, int]
) -> tuple[ReversibleGate, ...]:
    return tuple(
        ReversibleGate(gate.kind, tuple(mapping[wire] for wire in gate.wires))
        for gate in gates
    )


class ReversibleFixedPointAffineValueOracle:
    """Clean fixed-point affine value oracle with exact deterministic requantization."""

    def __init__(
        self,
        layer: QuantizedAffineLayer,
        *,
        require_no_overflow: bool = True,
        max_enumeration_bits: int = 16,
    ) -> None:
        if layer.activation != "identity":
            raise ValueError("fixed-point affine lowering currently requires identity activation")
        self.layer = layer
        self.max_enumeration_bits = int(max_enumeration_bits)
        product_fractional_bits = (
            layer.input_format.fractional_bits + layer.weight_format.fractional_bits
        )
        aligned_biases = tuple(
            rescale_code(
                bias,
                layer.bias_format.fractional_bits,
                product_fractional_bits,
            )
            for bias in layer.biases
        )
        raw_bounds: list[tuple[int, int]] = []
        for row, bias in zip(layer.weights, aligned_biases):
            lower = bias
            upper = bias
            for weight in row:
                if weight >= 0:
                    lower += layer.input_format.min_code * weight
                    upper += layer.input_format.max_code * weight
                else:
                    lower += layer.input_format.max_code * weight
                    upper += layer.input_format.min_code * weight
            raw_bounds.append((lower, upper))
        accumulator_bits = max(
            layer.input_format.bits,
            max(_bits_for_signed_range(lower, upper) for lower, upper in raw_bounds),
        )
        self.accumulator_format = FixedPointFormat(
            accumulator_bits,
            fractional_bits=product_fractional_bits,
            signed=True,
            overflow="raise",
        )
        self.raw_affine = ReversibleIntegerAffineValueOracle(
            layer.weights,
            aligned_biases,
            input_bits_per_feature=layer.input_format.bits,
            accumulator_bits=accumulator_bits,
            signed_inputs=layer.input_format.signed,
            signed_accumulator=True,
            require_no_overflow=require_no_overflow,
        )
        self.requantizer = ReversibleFixedPointRequantizationOracle(
            self.accumulator_format,
            layer.output_format,
            require_no_overflow=require_no_overflow,
        )
        self.raw_bounds = tuple(raw_bounds)

        input_bits = self.raw_affine.input_bits
        output_bits = layer.output_dimension * layer.output_format.bits
        offset = input_bits
        output_wires = tuple(range(offset, offset + output_bits)); offset += output_bits
        raw_output_wires = tuple(
            range(offset, offset + layer.output_dimension * accumulator_bits)
        ); offset += len(raw_output_wires)
        affine_work = tuple(
            range(offset, offset + len(self.raw_affine.layout.work_wires))
        ); offset += len(affine_work)
        requant_work = tuple(
            range(offset, offset + len(self.requantizer.layout.work_wires))
        ); offset += len(requant_work)
        self.layout = FixedPointAffineLayout(
            tuple(range(input_bits)), output_wires, raw_output_wires,
            affine_work, requant_work
        )
        self.circuit = ReversibleCircuit(offset)

        affine_mapping: dict[int, int] = {}
        for source in range(self.raw_affine.input_bits):
            affine_mapping[source] = source
        source_output_start = self.raw_affine.input_bits
        for source, destination in zip(
            range(source_output_start, source_output_start + self.raw_affine.output_bits),
            raw_output_wires,
        ):
            affine_mapping[source] = destination
        source_work_start = source_output_start + self.raw_affine.output_bits
        for source, destination in zip(
            range(source_work_start, self.raw_affine.circuit.num_qubits),
            affine_work,
        ):
            affine_mapping[source] = destination
        affine_gates = _remap_gates(self.raw_affine.circuit.gates, affine_mapping)
        self.circuit.extend(affine_gates)

        for row_index in range(layer.output_dimension):
            raw_start = row_index * accumulator_bits
            raw_register = raw_output_wires[raw_start : raw_start + accumulator_bits]
            out_start = row_index * layer.output_format.bits
            output_register = output_wires[
                out_start : out_start + layer.output_format.bits
            ]
            mapping: dict[int, int] = {}
            mapping.update(zip(self.requantizer.layout.input_register, raw_register))
            mapping.update(zip(self.requantizer.layout.output_register, output_register))
            mapping.update(zip(self.requantizer.layout.work_wires, requant_work))
            self.circuit.extend(_remap_gates(self.requantizer.circuit.gates, mapping))
        self.circuit.append_inverse(affine_gates)

    @property
    def input_bits(self) -> int:
        return self.raw_affine.input_bits

    @property
    def output_bits(self) -> int:
        return self.layer.output_dimension * self.layer.output_format.bits

    def encode_inputs(self, values: Sequence[int]) -> int:
        return self.raw_affine.encode_inputs(values)

    def evaluate_input_word(self, input_word: int) -> int:
        inputs = self.raw_affine.decode_input_word(input_word)
        outputs = self.layer.evaluate_codes(inputs)
        word = 0
        for index, code in enumerate(outputs):
            word |= self.layer.output_format.code_to_word(code) << (
                index * self.layer.output_format.bits
            )
        return word

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
                "structure-preserving fixed-point affine oracle: integer product-scale "
                "affine accumulation, clean half-away-from-zero requantization per output, "
                "output copy and reverse affine uncomputation"
            ),
        )


def compile_structure_preserving_fixed_point_affine_oracle(
    layer: QuantizedAffineLayer,
    *,
    require_no_overflow: bool = True,
    max_enumeration_bits: int = 16,
) -> ReversibleFixedPointAffineValueOracle:
    return ReversibleFixedPointAffineValueOracle(
        layer,
        require_no_overflow=require_no_overflow,
        max_enumeration_bits=max_enumeration_bits,
    )
