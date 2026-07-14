from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

from .compiler import OracleResourceEstimate
from .models import QuantizedNetwork
from .reversible import ReversibleCircuit, ReversibleGate, pack_register, unpack_register


@dataclass(frozen=True)
class AffineRowRange:
    minimum: int
    maximum: int
    accumulator_minimum: int
    accumulator_maximum: int
    no_overflow: bool


@dataclass(frozen=True)
class AffineRangeReport:
    rows: tuple[AffineRowRange, ...]
    no_overflow: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "rows": [asdict(row) for row in self.rows],
            "no_overflow": self.no_overflow,
        }


@dataclass(frozen=True)
class AffineValueLayout:
    input_registers: tuple[tuple[int, ...], ...]
    output_registers: tuple[tuple[int, ...], ...]
    accumulator: tuple[int, ...]
    scratch: tuple[int, ...]
    helper: int

    @property
    def work_wires(self) -> tuple[int, ...]:
        return self.accumulator + self.scratch + (self.helper,)


@dataclass(frozen=True)
class AffinePredicateLayout:
    input_registers: tuple[tuple[int, ...], ...]
    target: int
    accumulator: tuple[int, ...]
    scratch: tuple[int, ...]
    helper: int

    @property
    def work_wires(self) -> tuple[int, ...]:
        return self.accumulator + self.scratch + (self.helper,)


def _limits(bits: int, signed: bool) -> tuple[int, int]:
    if bits <= 0:
        raise ValueError("bit width must be positive")
    if signed:
        return -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    return 0, (1 << bits) - 1


def _decode_word(word: int, bits: int, signed: bool) -> int:
    raw = int(word)
    if raw < 0 or raw >= (1 << bits):
        raise ValueError(f"word must fit {bits} bits")
    if signed and raw >= (1 << (bits - 1)):
        return raw - (1 << bits)
    return raw


def _encode_code(code: int, bits: int, signed: bool) -> int:
    minimum, maximum = _limits(bits, signed)
    value = int(code)
    if value < minimum or value > maximum:
        raise OverflowError(f"code {value} does not fit the declared input format")
    return value & ((1 << bits) - 1)


def _validate_affine(
    weights: Sequence[Sequence[int]], biases: Sequence[int]
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
    rows = tuple(tuple(int(weight) for weight in row) for row in weights)
    offsets = tuple(int(bias) for bias in biases)
    if not rows or not rows[0]:
        raise ValueError("weights must contain at least one non-empty row")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("all affine rows must have the same feature count")
    if len(offsets) != len(rows):
        raise ValueError("one bias is required per affine row")
    return rows, offsets


def affine_range_report(
    weights: Sequence[Sequence[int]],
    biases: Sequence[int],
    *,
    input_bits: int,
    accumulator_bits: int,
    signed_inputs: bool = True,
    signed_accumulator: bool = True,
) -> AffineRangeReport:
    rows, offsets = _validate_affine(weights, biases)
    input_minimum, input_maximum = _limits(input_bits, signed_inputs)
    accumulator_minimum, accumulator_maximum = _limits(
        accumulator_bits, signed_accumulator
    )
    reports: list[AffineRowRange] = []
    for row, bias in zip(rows, offsets):
        minimum = int(bias)
        maximum = int(bias)
        for weight in row:
            if weight >= 0:
                minimum += weight * input_minimum
                maximum += weight * input_maximum
            else:
                minimum += weight * input_maximum
                maximum += weight * input_minimum
        safe = accumulator_minimum <= minimum and maximum <= accumulator_maximum
        reports.append(
            AffineRowRange(
                minimum=minimum,
                maximum=maximum,
                accumulator_minimum=accumulator_minimum,
                accumulator_maximum=accumulator_maximum,
                no_overflow=safe,
            )
        )
    return AffineRangeReport(tuple(reports), all(report.no_overflow for report in reports))


def append_cdkm_fixed_adder(
    circuit: ReversibleCircuit,
    addend: Sequence[int],
    accumulator: Sequence[int],
    helper: int,
) -> None:
    """Append a Cuccaro-style fixed-width in-place ripple-carry addition.

    On computational-basis states and a clean helper, this maps
    ``|a>|b>|0>`` to ``|a>|a+b mod 2**n>|0>``. The implementation uses one
    MAJ and one UMA block per bit, for exactly ``2n`` Toffoli and ``4n`` CNOT
    gates before any surrounding copy/uncompute operations.
    """

    left = tuple(int(wire) for wire in addend)
    right = tuple(int(wire) for wire in accumulator)
    carry = int(helper)
    if not left or len(left) != len(right):
        raise ValueError("adder registers must have the same positive width")
    all_wires = left + right + (carry,)
    if len(set(all_wires)) != len(all_wires):
        raise ValueError("adder registers and helper must be disjoint")

    def majority(a: int, b: int, c: int) -> None:
        circuit.cx(a, b)
        circuit.cx(a, c)
        circuit.ccx(c, b, a)

    def unmajority_add(a: int, b: int, c: int) -> None:
        circuit.ccx(c, b, a)
        circuit.cx(a, c)
        circuit.cx(c, b)

    majority(left[0], right[0], carry)
    for index in range(len(left) - 1):
        majority(left[index + 1], right[index + 1], left[index])
    for index in reversed(range(len(left) - 1)):
        unmajority_add(left[index + 1], right[index + 1], left[index])
    unmajority_add(left[0], right[0], carry)


def _append_shifted_input_copy(
    circuit: ReversibleCircuit,
    input_register: tuple[int, ...],
    scratch: tuple[int, ...],
    shift: int,
    signed_input: bool,
) -> tuple[ReversibleGate, ...]:
    if shift < 0 or shift >= len(scratch):
        raise ValueError("shift is outside the accumulator width")
    start = len(circuit.gates)
    for target_index in range(shift, len(scratch)):
        source_index = target_index - shift
        if source_index < len(input_register):
            source = input_register[source_index]
        elif signed_input:
            source = input_register[-1]
        else:
            continue
        circuit.cx(source, scratch[target_index])
    return circuit.gates[start:]


def _append_affine_compute(
    circuit: ReversibleCircuit,
    input_registers: tuple[tuple[int, ...], ...],
    accumulator: tuple[int, ...],
    scratch: tuple[int, ...],
    helper: int,
    weights: Sequence[int],
    bias: int,
    signed_inputs: bool,
) -> None:
    if len(weights) != len(input_registers):
        raise ValueError("one weight is required per input register")
    if len(accumulator) != len(scratch):
        raise ValueError("accumulator and scratch widths must match")
    width = len(accumulator)
    mask = (1 << width) - 1

    for input_register, weight in zip(input_registers, weights):
        coefficient = int(weight) & mask
        for shift in range(width):
            if not ((coefficient >> shift) & 1):
                continue
            copy_gates = _append_shifted_input_copy(
                circuit, input_register, scratch, shift, signed_inputs
            )
            append_cdkm_fixed_adder(circuit, scratch, accumulator, helper)
            circuit.append_inverse(copy_gates)

    bias_word = int(bias) & mask
    if bias_word:
        loaded: list[int] = []
        for bit, wire in enumerate(scratch):
            if (bias_word >> bit) & 1:
                circuit.x(wire)
                loaded.append(wire)
        append_cdkm_fixed_adder(circuit, scratch, accumulator, helper)
        for wire in reversed(loaded):
            circuit.x(wire)


def _value_layout(
    features: int, input_bits: int, outputs: int, accumulator_bits: int
) -> tuple[AffineValueLayout, int]:
    offset = 0
    input_registers: list[tuple[int, ...]] = []
    for _ in range(features):
        register = tuple(range(offset, offset + input_bits))
        input_registers.append(register)
        offset += input_bits
    output_registers: list[tuple[int, ...]] = []
    for _ in range(outputs):
        register = tuple(range(offset, offset + accumulator_bits))
        output_registers.append(register)
        offset += accumulator_bits
    accumulator = tuple(range(offset, offset + accumulator_bits))
    offset += accumulator_bits
    scratch = tuple(range(offset, offset + accumulator_bits))
    offset += accumulator_bits
    helper = offset
    offset += 1
    return (
        AffineValueLayout(
            tuple(input_registers),
            tuple(output_registers),
            accumulator,
            scratch,
            helper,
        ),
        offset,
    )


def _predicate_layout(
    features: int, input_bits: int, accumulator_bits: int
) -> tuple[AffinePredicateLayout, int]:
    offset = 0
    input_registers: list[tuple[int, ...]] = []
    for _ in range(features):
        register = tuple(range(offset, offset + input_bits))
        input_registers.append(register)
        offset += input_bits
    target = offset
    offset += 1
    accumulator = tuple(range(offset, offset + accumulator_bits))
    offset += accumulator_bits
    scratch = tuple(range(offset, offset + accumulator_bits))
    offset += accumulator_bits
    helper = offset
    offset += 1
    return (
        AffinePredicateLayout(
            tuple(input_registers), target, accumulator, scratch, helper
        ),
        offset,
    )


def _resource_estimate(
    circuit: ReversibleCircuit,
    *,
    input_qubits: int,
    output_qubits: int,
    work_qubits: int,
    synthesis: str,
) -> OracleResourceEstimate:
    counts = circuit.gate_counts()
    toffoli = counts["ccx"]
    return OracleResourceEstimate(
        input_qubits=input_qubits,
        output_qubits=output_qubits,
        peak_clean_ancillas=work_qubits,
        logical_qubits=circuit.num_qubits,
        controlled_x_terms=counts["cx"] + counts["ccx"],
        negative_control_x_gates=0,
        x_gates=counts["x"],
        cnot_gates=counts["cx"],
        toffoli_gates=toffoli,
        h_gates=0,
        z_gates=0,
        t_count_upper_bound=7 * toffoli,
        t_depth_upper_bound=3 * toffoli,
        logical_depth_upper_bound=circuit.logical_depth(),
        synthesis=synthesis,
    )


class ReversibleIntegerAffineValueOracle:
    """Polynomial-size clean oracle for integer affine maps.

    Inputs may be unsigned or two's-complement signed words. Every row is computed
    modulo ``2**accumulator_bits`` by constant shift-add multiplication, copied to
    its output register, and then completely uncomputed. When ``require_no_overflow``
    is true, interval analysis certifies that the modular representation equals the
    intended mathematical integer result over the full input word domain.
    """

    def __init__(
        self,
        weights: Sequence[Sequence[int]],
        biases: Sequence[int],
        *,
        input_bits_per_feature: int,
        accumulator_bits: int,
        signed_inputs: bool = True,
        signed_accumulator: bool = True,
        require_no_overflow: bool = True,
    ) -> None:
        rows, offsets = _validate_affine(weights, biases)
        if input_bits_per_feature <= 0 or accumulator_bits <= 0:
            raise ValueError("input and accumulator widths must be positive")
        if input_bits_per_feature > accumulator_bits:
            raise ValueError("input words cannot be wider than the accumulator")
        self.weights = rows
        self.biases = offsets
        self.feature_count = len(rows[0])
        self.output_count = len(rows)
        self.input_bits_per_feature = int(input_bits_per_feature)
        self.accumulator_bits = int(accumulator_bits)
        self.signed_inputs = bool(signed_inputs)
        self.signed_accumulator = bool(signed_accumulator)
        self.range_report = affine_range_report(
            rows,
            offsets,
            input_bits=self.input_bits_per_feature,
            accumulator_bits=self.accumulator_bits,
            signed_inputs=self.signed_inputs,
            signed_accumulator=self.signed_accumulator,
        )
        if require_no_overflow and not self.range_report.no_overflow:
            raise OverflowError(
                "affine range is not representable in the declared accumulator"
            )

        self.layout, num_qubits = _value_layout(
            self.feature_count,
            self.input_bits_per_feature,
            self.output_count,
            self.accumulator_bits,
        )
        self.circuit = ReversibleCircuit(num_qubits)
        for row, bias, output_register in zip(
            self.weights, self.biases, self.layout.output_registers
        ):
            start = len(self.circuit.gates)
            _append_affine_compute(
                self.circuit,
                self.layout.input_registers,
                self.layout.accumulator,
                self.layout.scratch,
                self.layout.helper,
                row,
                bias,
                self.signed_inputs,
            )
            compute_gates = self.circuit.gates[start:]
            for source, target in zip(self.layout.accumulator, output_register):
                self.circuit.cx(source, target)
            self.circuit.append_inverse(compute_gates)

    @property
    def input_bits(self) -> int:
        return self.feature_count * self.input_bits_per_feature

    @property
    def output_bits(self) -> int:
        return self.output_count * self.accumulator_bits

    def encode_inputs(self, values: Sequence[int]) -> int:
        codes = tuple(int(value) for value in values)
        if len(codes) != self.feature_count:
            raise ValueError(f"expected {self.feature_count} input values")
        word = 0
        for index, code in enumerate(codes):
            encoded = _encode_code(
                code, self.input_bits_per_feature, self.signed_inputs
            )
            word |= encoded << (index * self.input_bits_per_feature)
        return word

    def decode_input_word(self, input_word: int) -> tuple[int, ...]:
        raw = int(input_word)
        if raw < 0 or raw >= (1 << self.input_bits):
            raise ValueError("input_word is outside the candidate register")
        mask = (1 << self.input_bits_per_feature) - 1
        return tuple(
            _decode_word(
                (raw >> (index * self.input_bits_per_feature)) & mask,
                self.input_bits_per_feature,
                self.signed_inputs,
            )
            for index in range(self.feature_count)
        )

    def evaluate_input_word(self, input_word: int) -> int:
        values = self.decode_input_word(input_word)
        mask = (1 << self.accumulator_bits) - 1
        output = 0
        for row_index, (row, bias) in enumerate(zip(self.weights, self.biases)):
            result = int(bias) + sum(weight * value for weight, value in zip(row, values))
            output |= (result & mask) << (row_index * self.accumulator_bits)
        return output

    def _pack_state(self, input_word: int, output_word: int) -> int:
        input_value = int(input_word)
        output_value = int(output_word)
        if input_value < 0 or input_value >= (1 << self.input_bits):
            raise ValueError("input_word is outside the candidate register")
        if output_value < 0 or output_value >= (1 << self.output_bits):
            raise ValueError("output_word is outside the output register")
        return input_value | (output_value << self.input_bits)

    def _extract(self, state: int) -> tuple[int, int, int]:
        input_mask = (1 << self.input_bits) - 1
        output_mask = (1 << self.output_bits) - 1
        input_word = state & input_mask
        output_word = (state >> self.input_bits) & output_mask
        work_word = state >> (self.input_bits + self.output_bits)
        return input_word, output_word, work_word

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

    def verify_basis_permutation(self, *, exhaustive_output_words: bool = False) -> bool:
        output_words = range(1 << self.output_bits) if exhaustive_output_words else (0, 1)
        for input_word in range(1 << self.input_bits):
            for output_word in output_words:
                expected_output = output_word ^ self.evaluate_input_word(input_word)
                forward = self.apply(input_word, output_word)
                if forward != (input_word, expected_output, 0):
                    return False
                if self.inverse_apply(*forward) != (input_word, output_word, 0):
                    return False
        return True

    def resource_estimate(self, *, phase_kickback: bool = False) -> OracleResourceEstimate:
        if phase_kickback:
            raise ValueError("phase_kickback is defined for a one-bit predicate oracle")
        return _resource_estimate(
            self.circuit,
            input_qubits=self.input_bits,
            output_qubits=self.output_bits,
            work_qubits=len(self.layout.work_wires),
            synthesis=(
                "structure-preserving integer affine compiler: sign/zero extension, "
                "constant shift-add multiplication, Cuccaro MAJ/UMA modular addition, "
                "copy, and exact reverse uncomputation"
            ),
        )


class ReversibleIntegerAffinePredicateOracle:
    """Clean threshold predicate for one signed integer affine row."""

    output_bits = 1

    def __init__(
        self,
        weights: Sequence[int],
        bias: int,
        *,
        threshold: int = 0,
        input_bits_per_feature: int,
        accumulator_bits: int,
        signed_inputs: bool = True,
        require_no_overflow: bool = True,
        max_enumeration_bits: int = 16,
    ) -> None:
        row = tuple(int(weight) for weight in weights)
        if not row:
            raise ValueError("weights must contain at least one feature")
        if input_bits_per_feature <= 0 or accumulator_bits <= 1:
            raise ValueError("predicate widths must be positive and signed")
        if input_bits_per_feature > accumulator_bits:
            raise ValueError("input words cannot be wider than the accumulator")
        self.weights = row
        self.bias = int(bias)
        self.threshold = int(threshold)
        self.feature_count = len(row)
        self.input_bits_per_feature = int(input_bits_per_feature)
        self.accumulator_bits = int(accumulator_bits)
        self.signed_inputs = bool(signed_inputs)
        self.max_enumeration_bits = int(max_enumeration_bits)
        if self.max_enumeration_bits <= 0:
            raise ValueError("max_enumeration_bits must be positive")
        effective_bias = self.bias - self.threshold
        self.range_report = affine_range_report(
            (self.weights,),
            (effective_bias,),
            input_bits=self.input_bits_per_feature,
            accumulator_bits=self.accumulator_bits,
            signed_inputs=self.signed_inputs,
            signed_accumulator=True,
        )
        if require_no_overflow and not self.range_report.no_overflow:
            raise OverflowError(
                "affine-threshold difference is not representable in the signed accumulator"
            )

        self.layout, num_qubits = _predicate_layout(
            self.feature_count, self.input_bits_per_feature, self.accumulator_bits
        )
        self.circuit = ReversibleCircuit(num_qubits)
        start = len(self.circuit.gates)
        _append_affine_compute(
            self.circuit,
            self.layout.input_registers,
            self.layout.accumulator,
            self.layout.scratch,
            self.layout.helper,
            self.weights,
            effective_bias,
            self.signed_inputs,
        )
        compute_gates = self.circuit.gates[start:]
        self.circuit.x(self.layout.target)
        self.circuit.cx(self.layout.accumulator[-1], self.layout.target)
        self.circuit.append_inverse(compute_gates)

    @property
    def input_bits(self) -> int:
        return self.feature_count * self.input_bits_per_feature

    def encode_inputs(self, values: Sequence[int]) -> int:
        helper = ReversibleIntegerAffineValueOracle(
            (self.weights,),
            (self.bias,),
            input_bits_per_feature=self.input_bits_per_feature,
            accumulator_bits=self.accumulator_bits,
            signed_inputs=self.signed_inputs,
            signed_accumulator=True,
            require_no_overflow=False,
        )
        return helper.encode_inputs(values)

    def decode_input_word(self, input_word: int) -> tuple[int, ...]:
        raw = int(input_word)
        if raw < 0 or raw >= (1 << self.input_bits):
            raise ValueError("input_word is outside the candidate register")
        mask = (1 << self.input_bits_per_feature) - 1
        return tuple(
            _decode_word(
                (raw >> (index * self.input_bits_per_feature)) & mask,
                self.input_bits_per_feature,
                self.signed_inputs,
            )
            for index in range(self.feature_count)
        )

    def evaluate_predicate(self, input_word: int) -> int:
        values = self.decode_input_word(input_word)
        score = self.bias + sum(weight * value for weight, value in zip(self.weights, values))
        return int(score >= self.threshold)

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
        input_word = state & input_mask
        target = (state >> self.input_bits) & 1
        work_word = state >> (self.input_bits + 1)
        return input_word, target, work_word

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
                "structure-preserving signed affine-threshold compiler: modular "
                "constant shift-add, Cuccaro MAJ/UMA addition, sign-bit predicate, "
                "and exact reverse uncomputation"
            ),
        )


def compile_structure_preserving_affine_oracle(
    model: QuantizedNetwork,
    *,
    require_no_overflow: bool = True,
) -> ReversibleIntegerAffineValueOracle:
    """Lower a one-layer integer quantized network to the arithmetic oracle."""

    if len(model.layers) != 1:
        raise ValueError("the first arithmetic milestone supports one affine layer")
    layer = model.layers[0]
    formats = (
        layer.input_format,
        layer.weight_format,
        layer.bias_format,
        layer.output_format,
    )
    if any(fmt.fractional_bits != 0 for fmt in formats):
        raise ValueError("fractional fixed-point requantization is not yet supported")
    if layer.activation != "identity":
        raise ValueError("the first arithmetic milestone does not lower activations")
    if model.output_mode != "raw":
        raise ValueError("value-oracle adapter requires raw network output")
    return ReversibleIntegerAffineValueOracle(
        layer.weights,
        layer.biases,
        input_bits_per_feature=layer.input_format.bits,
        accumulator_bits=layer.output_format.bits,
        signed_inputs=layer.input_format.signed,
        signed_accumulator=layer.output_format.signed,
        require_no_overflow=require_no_overflow,
    )


def compile_structure_preserving_threshold_oracle(
    model: QuantizedNetwork,
    *,
    require_no_overflow: bool = True,
    max_enumeration_bits: int = 16,
) -> ReversibleIntegerAffinePredicateOracle:
    """Lower one integer affine logit and its signed threshold to a clean predicate."""

    if len(model.layers) != 1:
        raise ValueError("the first arithmetic milestone supports one affine layer")
    layer = model.layers[0]
    formats = (
        layer.input_format,
        layer.weight_format,
        layer.bias_format,
        layer.output_format,
    )
    if any(fmt.fractional_bits != 0 for fmt in formats):
        raise ValueError("fractional fixed-point requantization is not yet supported")
    if layer.activation != "identity":
        raise ValueError("the first arithmetic milestone does not lower activations")
    if model.output_mode != "binary_threshold" or layer.output_dimension != 1:
        raise ValueError("threshold adapter requires one binary-threshold logit")
    if not layer.output_format.signed:
        raise ValueError("sign-bit threshold compilation requires a signed accumulator")
    return ReversibleIntegerAffinePredicateOracle(
        layer.weights[0],
        layer.biases[0],
        threshold=model.binary_threshold,
        input_bits_per_feature=layer.input_format.bits,
        accumulator_bits=layer.output_format.bits,
        signed_inputs=layer.input_format.signed,
        require_no_overflow=require_no_overflow,
        max_enumeration_bits=max_enumeration_bits,
    )
