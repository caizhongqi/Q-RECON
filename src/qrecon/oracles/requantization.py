from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

from .arithmetic import _resource_estimate, append_cdkm_fixed_adder
from .compiler import OracleResourceEstimate
from .fixed_point import FixedPointFormat, rescale_code
from .reversible import ReversibleCircuit, ReversibleGate


@dataclass(frozen=True)
class RequantizationRangeReport:
    source_minimum: int
    source_maximum: int
    result_minimum: int
    result_maximum: int
    target_minimum: int
    target_maximum: int
    no_overflow: bool

    def to_dict(self) -> dict[str, int | bool]:
        return asdict(self)


@dataclass(frozen=True)
class RequantizationLayout:
    input_register: tuple[int, ...]
    output_register: tuple[int, ...]
    magnitude: tuple[int, ...]
    quotient: tuple[int, ...]
    addend: tuple[int, ...]
    helper: int
    control_work: tuple[int, ...]

    @property
    def work_wires(self) -> tuple[int, ...]:
        return (
            self.magnitude
            + self.quotient
            + self.addend
            + (self.helper,)
            + self.control_work
        )


def _append_multi_controlled_x(
    circuit: ReversibleCircuit,
    controls: Sequence[int],
    target: int,
    work: Sequence[int],
) -> tuple[ReversibleGate, ...]:
    control_wires = tuple(int(wire) for wire in controls)
    ancillas = tuple(int(wire) for wire in work)
    if target in control_wires or len(set(control_wires)) != len(control_wires):
        raise ValueError("controls and target must be distinct")
    needed = max(0, len(control_wires) - 2)
    if len(ancillas) < needed:
        raise ValueError(f"multi-controlled X requires {needed} clean work bits")
    if set(ancillas[:needed]).intersection(control_wires + (target,)):
        raise ValueError("multi-controlled X work must be disjoint")

    start = len(circuit.gates)
    if not control_wires:
        circuit.x(target)
    elif len(control_wires) == 1:
        circuit.cx(control_wires[0], target)
    elif len(control_wires) == 2:
        circuit.ccx(control_wires[0], control_wires[1], target)
    else:
        used = ancillas[:needed]
        circuit.ccx(control_wires[0], control_wires[1], used[0])
        for index in range(2, len(control_wires) - 1):
            circuit.ccx(used[index - 2], control_wires[index], used[index - 1])
        circuit.ccx(used[-1], control_wires[-1], target)
        for index in reversed(range(2, len(control_wires) - 1)):
            circuit.ccx(used[index - 2], control_wires[index], used[index - 1])
        circuit.ccx(control_wires[0], control_wires[1], used[0])
    return circuit.gates[start:]


def append_controlled_increment(
    circuit: ReversibleCircuit,
    control: int,
    register: Sequence[int],
    work: Sequence[int],
) -> tuple[ReversibleGate, ...]:
    """Add one modulo ``2**n`` when ``control`` is one, preserving clean work."""

    word = tuple(int(wire) for wire in register)
    if not word:
        raise ValueError("register must be non-empty")
    if control in word:
        raise ValueError("control must be disjoint from the incremented register")
    start = len(circuit.gates)
    for index in reversed(range(1, len(word))):
        _append_multi_controlled_x(
            circuit,
            (int(control),) + word[:index],
            word[index],
            work,
        )
    circuit.cx(int(control), word[0])
    return circuit.gates[start:]


def append_controlled_twos_complement(
    circuit: ReversibleCircuit,
    control: int,
    register: Sequence[int],
    work: Sequence[int],
) -> tuple[ReversibleGate, ...]:
    """Conditionally replace a word by its two's complement, with clean work."""

    word = tuple(int(wire) for wire in register)
    start = len(circuit.gates)
    for wire in word:
        circuit.cx(int(control), wire)
    append_controlled_increment(circuit, int(control), word, work)
    return circuit.gates[start:]


def _allocate_requantization_layout(
    source_bits: int,
    target_bits: int,
    magnitude_bits: int,
    control_work_bits: int,
) -> tuple[RequantizationLayout, int]:
    offset = 0
    source = tuple(range(offset, offset + source_bits)); offset += source_bits
    output = tuple(range(offset, offset + target_bits)); offset += target_bits
    magnitude = tuple(range(offset, offset + magnitude_bits)); offset += magnitude_bits
    quotient = tuple(range(offset, offset + target_bits)); offset += target_bits
    addend = tuple(range(offset, offset + magnitude_bits)); offset += magnitude_bits
    helper = offset; offset += 1
    control_work = tuple(range(offset, offset + control_work_bits)); offset += control_work_bits
    return RequantizationLayout(
        source, output, magnitude, quotient, addend, helper, control_work
    ), offset


class ReversibleFixedPointRequantizationOracle:
    """Clean deterministic fixed-point downscaling with half-away-from-zero ties.

    With ``require_no_overflow=True`` the full source register must map into the
    target format.  A composed compiler may set it to ``False`` only after it has
    separately certified that every *reachable* source word is representable.
    The circuit remains a total reversible permutation, but the reference
    fixed-point semantics are then certified only on that reachable subset.
    """

    def __init__(
        self,
        source_format: FixedPointFormat,
        target_format: FixedPointFormat,
        *,
        require_no_overflow: bool = True,
    ) -> None:
        if source_format.fractional_bits < target_format.fractional_bits:
            raise ValueError("this compiler currently supports downscaling only")
        if source_format.signed and not target_format.signed:
            raise OverflowError("a signed full-domain source cannot fit an unsigned target")
        self.source_format = source_format
        self.target_format = target_format
        self.require_no_overflow = bool(require_no_overflow)
        self.shift = source_format.fractional_bits - target_format.fractional_bits
        result_minimum = rescale_code(
            source_format.min_code,
            source_format.fractional_bits,
            target_format.fractional_bits,
        )
        result_maximum = rescale_code(
            source_format.max_code,
            source_format.fractional_bits,
            target_format.fractional_bits,
        )
        safe = target_format.contains(result_minimum) and target_format.contains(result_maximum)
        self.range_report = RequantizationRangeReport(
            source_format.min_code,
            source_format.max_code,
            result_minimum,
            result_maximum,
            target_format.min_code,
            target_format.max_code,
            safe,
        )
        if not safe and self.require_no_overflow:
            raise OverflowError(
                "requantized source range does not fit target format; set "
                "require_no_overflow=False only inside a composition that separately "
                "certifies all reachable source words"
            )

        magnitude_bits = max(source_format.bits + 1, self.shift + 1)
        control_work_bits = (
            max(0, max(magnitude_bits, target_format.bits) - 2)
            if source_format.signed
            else 0
        )
        self.layout, num_qubits = _allocate_requantization_layout(
            source_format.bits,
            target_format.bits,
            magnitude_bits,
            control_work_bits,
        )
        self.circuit = ReversibleCircuit(num_qubits)
        compute_start = len(self.circuit.gates)

        for source, target in zip(self.layout.input_register, self.layout.magnitude):
            self.circuit.cx(source, target)
        sign_wire = self.layout.input_register[-1]
        if source_format.signed:
            for target in self.layout.magnitude[source_format.bits :]:
                self.circuit.cx(sign_wire, target)
            append_controlled_twos_complement(
                self.circuit,
                sign_wire,
                self.layout.magnitude,
                self.layout.control_work,
            )

        if self.shift > 0:
            half = 1 << (self.shift - 1)
            loaded: list[int] = []
            for bit, wire in enumerate(self.layout.addend):
                if (half >> bit) & 1:
                    self.circuit.x(wire)
                    loaded.append(wire)
            append_cdkm_fixed_adder(
                self.circuit,
                self.layout.addend,
                self.layout.magnitude,
                self.layout.helper,
            )
            for wire in reversed(loaded):
                self.circuit.x(wire)

        for target_index, target in enumerate(self.layout.quotient):
            source_index = target_index + self.shift
            if source_index < len(self.layout.magnitude):
                self.circuit.cx(self.layout.magnitude[source_index], target)
        if source_format.signed:
            append_controlled_twos_complement(
                self.circuit,
                sign_wire,
                self.layout.quotient,
                self.layout.control_work,
            )

        compute_gates = self.circuit.gates[compute_start:]
        for source, target in zip(self.layout.quotient, self.layout.output_register):
            self.circuit.cx(source, target)
        self.circuit.append_inverse(compute_gates)

    @property
    def input_bits(self) -> int:
        return self.source_format.bits

    @property
    def output_bits(self) -> int:
        return self.target_format.bits

    def source_word_is_representable(self, input_word: int) -> bool:
        source_code = self.source_format.word_to_code(input_word)
        result = rescale_code(
            source_code,
            self.source_format.fractional_bits,
            self.target_format.fractional_bits,
        )
        return self.target_format.contains(result)

    def evaluate_input_word(self, input_word: int) -> int:
        source_code = self.source_format.word_to_code(input_word)
        target_code = self.target_format.requantize(
            source_code, self.source_format.fractional_bits
        )
        return self.target_format.code_to_word(target_code)

    def decode_output_word(self, output_word: int) -> int:
        return self.target_format.word_to_code(output_word)

    def _pack_state(self, input_word: int, output_word: int) -> int:
        if input_word < 0 or input_word >= (1 << self.input_bits):
            raise ValueError("input_word is outside the source register")
        if output_word < 0 or output_word >= (1 << self.output_bits):
            raise ValueError("output_word is outside the target register")
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

    def verify_basis_permutation(
        self,
        *,
        exhaustive_output_words: bool = True,
        certified_inputs_only: bool = True,
    ) -> bool:
        outputs = range(1 << self.output_bits) if exhaustive_output_words else (0, 1)
        for input_word in range(1 << self.input_bits):
            if certified_inputs_only and not self.source_word_is_representable(input_word):
                continue
            expected = self.evaluate_input_word(input_word)
            for output_word in outputs:
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
                "structure-preserving fixed-point downscaler: sign extension, "
                "controlled absolute value, half-unit addition, logical right shift, "
                "conditional two's-complement restoration, copy and uncomputation"
            ),
        )
