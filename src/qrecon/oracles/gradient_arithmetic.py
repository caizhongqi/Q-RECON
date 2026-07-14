from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

from qrecon.theory import compare_search_queries, optimal_standard_grover_iterations

from .analysis import analyze_finite_oracle
from .arithmetic import (
    ReversibleIntegerAffineValueOracle,
    _resource_estimate,
    append_cdkm_fixed_adder,
)
from .comparators import append_equality_to_constant
from .compiler import OracleResourceEstimate
from .grover import estimate_grover_resources, simulate_grover
from .reversible import ReversibleCircuit, ReversibleGate


@dataclass(frozen=True)
class GradientArithmeticRangeReport:
    residual_minimum: int
    residual_maximum: int
    product_bounds: tuple[tuple[int, int], ...]
    gradient_minimum: int
    gradient_maximum: int
    no_overflow: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class GradientValueLayout:
    candidate_registers: tuple[tuple[int, ...], ...]
    output_registers: tuple[tuple[int, ...], ...]
    residual: tuple[int, ...]
    accumulator: tuple[int, ...]
    scratch: tuple[int, ...]
    helper: int

    @property
    def work_wires(self) -> tuple[int, ...]:
        return self.residual + self.accumulator + self.scratch + (self.helper,)


@dataclass(frozen=True)
class GradientEqualityLayout:
    input_wires: tuple[int, ...]
    target: int
    value_wires: tuple[int, ...]
    value_work: tuple[int, ...]
    equality_work: tuple[int, ...]

    @property
    def work_wires(self) -> tuple[int, ...]:
        return self.value_wires + self.value_work + self.equality_work


def append_signed_modular_product(
    circuit: ReversibleCircuit,
    multiplicand: Sequence[int],
    multiplier: Sequence[int],
    accumulator: Sequence[int],
    scratch: Sequence[int],
    helper: int,
) -> tuple[ReversibleGate, ...]:
    """Add a two's-complement product modulo ``2**w`` into ``accumulator``.

    The shorter multiplicand is sign-extended. Every multiplier bit controls a
    shifted copy into clean scratch, which is added by the fixed ripple-carry
    adder and then erased. Multiplication in ``Z/(2**w)`` agrees with signed
    two's-complement multiplication modulo the word size.
    """

    left = tuple(int(wire) for wire in multiplicand)
    right = tuple(int(wire) for wire in multiplier)
    output = tuple(int(wire) for wire in accumulator)
    temporary = tuple(int(wire) for wire in scratch)
    carry = int(helper)
    width = len(output)
    if not left or not right or width <= 0:
        raise ValueError("multiplier registers must be non-empty")
    if len(right) != width or len(temporary) != width:
        raise ValueError("multiplier, accumulator, and scratch widths must match")
    if len(left) > width:
        raise ValueError("multiplicand cannot be wider than the product word")
    all_wires = left + right + output + temporary + (carry,)
    if len(set(all_wires)) != len(all_wires):
        raise ValueError("product registers and helper must be disjoint")

    start = len(circuit.gates)
    for shift, multiplier_bit in enumerate(right):
        copy_start = len(circuit.gates)
        for target_index in range(shift, width):
            source_index = target_index - shift
            source = left[source_index] if source_index < len(left) else left[-1]
            circuit.ccx(multiplier_bit, source, temporary[target_index])
        copy_gates = circuit.gates[copy_start:]
        append_cdkm_fixed_adder(circuit, temporary, output, carry)
        circuit.append_inverse(copy_gates)
    return circuit.gates[start:]


def gradient_arithmetic_range_report(
    weights: Sequence[int],
    bias: int,
    *,
    input_bits: int,
    gradient_bits: int,
) -> GradientArithmeticRangeReport:
    """Conservative interval proof for residual and gradient products."""

    if input_bits <= 1 or gradient_bits <= 1:
        raise ValueError("signed input and gradient widths must exceed one bit")
    row = tuple(int(weight) for weight in weights)
    if not row:
        raise ValueError("weights must contain at least one feature")
    input_minimum = -(1 << (input_bits - 1))
    input_maximum = (1 << (input_bits - 1)) - 1
    residual_minimum = int(bias) - input_maximum
    residual_maximum = int(bias) - input_minimum
    for weight in row:
        if weight >= 0:
            residual_minimum += weight * input_minimum
            residual_maximum += weight * input_maximum
        else:
            residual_minimum += weight * input_maximum
            residual_maximum += weight * input_minimum
    product_bounds = tuple(
        (
            min(
                residual_minimum * input_minimum,
                residual_minimum * input_maximum,
                residual_maximum * input_minimum,
                residual_maximum * input_maximum,
            ),
            max(
                residual_minimum * input_minimum,
                residual_minimum * input_maximum,
                residual_maximum * input_minimum,
                residual_maximum * input_maximum,
            ),
        )
        for _ in row
    )
    gradient_minimum = -(1 << (gradient_bits - 1))
    gradient_maximum = (1 << (gradient_bits - 1)) - 1
    safe = gradient_minimum <= residual_minimum <= residual_maximum <= gradient_maximum
    safe = safe and all(
        gradient_minimum <= lower <= upper <= gradient_maximum
        for lower, upper in product_bounds
    )
    return GradientArithmeticRangeReport(
        residual_minimum,
        residual_maximum,
        product_bounds,
        gradient_minimum,
        gradient_maximum,
        safe,
    )


def _allocate_value_layout(
    features: int, input_bits: int, gradient_bits: int
) -> tuple[GradientValueLayout, int]:
    offset = 0
    candidates: list[tuple[int, ...]] = []
    for _ in range(features + 1):
        candidates.append(tuple(range(offset, offset + input_bits)))
        offset += input_bits
    outputs: list[tuple[int, ...]] = []
    for _ in range(features + 1):
        outputs.append(tuple(range(offset, offset + gradient_bits)))
        offset += gradient_bits
    residual = tuple(range(offset, offset + gradient_bits))
    offset += gradient_bits
    accumulator = tuple(range(offset, offset + gradient_bits))
    offset += gradient_bits
    scratch = tuple(range(offset, offset + gradient_bits))
    offset += gradient_bits
    helper = offset
    offset += 1
    return (
        GradientValueLayout(
            tuple(candidates), tuple(outputs), residual, accumulator, scratch, helper
        ),
        offset,
    )


def _remap_gates(
    gates: Sequence[ReversibleGate], mapping: dict[int, int]
) -> tuple[ReversibleGate, ...]:
    return tuple(
        ReversibleGate(gate.kind, tuple(mapping[wire] for wire in gate.wires))
        for gate in gates
    )


class ReversibleSingleRecordGradientValueOracle:
    """Clean polynomial-size exact-gradient oracle for one linear-regression record."""

    def __init__(
        self,
        weights: Sequence[int],
        bias: int,
        *,
        input_bits: int,
        gradient_bits: int,
        require_no_overflow: bool = True,
        max_enumeration_bits: int = 16,
    ) -> None:
        row = tuple(int(weight) for weight in weights)
        if not row:
            raise ValueError("weights must contain at least one feature")
        if input_bits <= 1 or gradient_bits <= 1 or input_bits > gradient_bits:
            raise ValueError("signed input and gradient widths are invalid")
        self.weights = row
        self.bias = int(bias)
        self.feature_count = len(row)
        self.input_bits_per_word = int(input_bits)
        self.gradient_bits = int(gradient_bits)
        self.max_enumeration_bits = int(max_enumeration_bits)
        if self.max_enumeration_bits <= 0:
            raise ValueError("max_enumeration_bits must be positive")
        self.range_report = gradient_arithmetic_range_report(
            row, self.bias, input_bits=input_bits, gradient_bits=gradient_bits
        )
        if require_no_overflow and not self.range_report.no_overflow:
            raise OverflowError("gradient word does not contain the certified range")
        if self.input_bits > self.max_enumeration_bits:
            raise ValueError("candidate space exceeds max_enumeration_bits")

        residual_oracle = ReversibleIntegerAffineValueOracle(
            (row + (-1,),),
            (self.bias,),
            input_bits_per_feature=input_bits,
            accumulator_bits=gradient_bits,
            signed_inputs=True,
            signed_accumulator=True,
            require_no_overflow=require_no_overflow,
        )
        self.residual_oracle = residual_oracle
        self.layout, num_qubits = _allocate_value_layout(
            self.feature_count, input_bits, gradient_bits
        )
        self.circuit = ReversibleCircuit(num_qubits)
        mapping: dict[int, int] = {}
        for source, target in zip(
            residual_oracle.layout.input_registers, self.layout.candidate_registers
        ):
            mapping.update(zip(source, target))
        mapping.update(
            zip(residual_oracle.layout.output_registers[0], self.layout.residual)
        )
        mapping.update(
            zip(
                residual_oracle.layout.work_wires,
                self.layout.accumulator + self.layout.scratch + (self.layout.helper,),
            )
        )
        residual_gates = _remap_gates(residual_oracle.circuit.gates, mapping)
        self.circuit.extend(residual_gates)
        for source, target in zip(self.layout.residual, self.layout.output_registers[-1]):
            self.circuit.cx(source, target)
        for feature_register, output_register in zip(
            self.layout.candidate_registers[:-1], self.layout.output_registers[:-1]
        ):
            product_start = len(self.circuit.gates)
            append_signed_modular_product(
                self.circuit,
                feature_register,
                self.layout.residual,
                self.layout.accumulator,
                self.layout.scratch,
                self.layout.helper,
            )
            product_gates = self.circuit.gates[product_start:]
            for source, target in zip(self.layout.accumulator, output_register):
                self.circuit.cx(source, target)
            self.circuit.append_inverse(product_gates)
        self.circuit.append_inverse(residual_gates)

    @property
    def input_bits(self) -> int:
        return (self.feature_count + 1) * self.input_bits_per_word

    @property
    def output_bits(self) -> int:
        return (self.feature_count + 1) * self.gradient_bits

    def encode_candidate(self, inputs: Sequence[int], target: int) -> int:
        values = tuple(int(value) for value in inputs) + (int(target),)
        if len(values) != self.feature_count + 1:
            raise ValueError("candidate dimension mismatch")
        minimum = -(1 << (self.input_bits_per_word - 1))
        maximum = (1 << (self.input_bits_per_word - 1)) - 1
        mask = (1 << self.input_bits_per_word) - 1
        word = 0
        for index, value in enumerate(values):
            if value < minimum or value > maximum:
                raise OverflowError("candidate value does not fit input word")
            word |= (value & mask) << (index * self.input_bits_per_word)
        return word

    def decode_candidate(self, candidate_word: int) -> tuple[tuple[int, ...], int]:
        raw = int(candidate_word)
        if raw < 0 or raw >= (1 << self.input_bits):
            raise ValueError("candidate_word is outside the input register")
        mask = (1 << self.input_bits_per_word) - 1
        values: list[int] = []
        for index in range(self.feature_count + 1):
            word = (raw >> (index * self.input_bits_per_word)) & mask
            if word >= (1 << (self.input_bits_per_word - 1)):
                word -= 1 << self.input_bits_per_word
            values.append(word)
        return tuple(values[:-1]), values[-1]

    def gradient_components(self, candidate_word: int) -> tuple[int, ...]:
        inputs, target = self.decode_candidate(candidate_word)
        residual = self.bias + sum(
            weight * value for weight, value in zip(self.weights, inputs)
        ) - target
        return tuple(residual * value for value in inputs) + (residual,)

    def evaluate_input_word(self, candidate_word: int) -> int:
        mask = (1 << self.gradient_bits) - 1
        return sum(
            (value & mask) << (index * self.gradient_bits)
            for index, value in enumerate(self.gradient_components(candidate_word))
        )

    def _pack_state(self, input_word: int, output_word: int) -> int:
        if input_word < 0 or input_word >= (1 << self.input_bits):
            raise ValueError("input_word is outside the candidate register")
        if output_word < 0 or output_word >= (1 << self.output_bits):
            raise ValueError("output_word is outside the gradient register")
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
            raise ValueError("phase kickback requires the equality predicate")
        return _resource_estimate(
            self.circuit,
            input_qubits=self.input_bits,
            output_qubits=self.output_bits,
            work_qubits=len(self.layout.work_wires),
            synthesis=(
                "structure-preserving single-record linear-gradient oracle: clean "
                "affine residual, signed modular variable products, gradient copy, "
                "and exact reverse uncomputation"
            ),
        )


class ReversibleSingleRecordGradientEqualityOracle:
    """Clean predicate that marks candidates with one exact released gradient."""

    output_bits = 1

    def __init__(
        self,
        value_oracle: ReversibleSingleRecordGradientValueOracle,
        observed_word: int,
    ) -> None:
        target_value = int(observed_word)
        if target_value < 0 or target_value >= (1 << value_oracle.output_bits):
            raise ValueError("observed_word does not fit the gradient output register")
        self.value_oracle = value_oracle
        self.observed_word = target_value
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
        self.layout = GradientEqualityLayout(
            input_wires, target, value_wires, value_work, equality_work
        )
        self.circuit = ReversibleCircuit(offset)
        mapping = {
            source: (source if source < value_oracle.input_bits else source + 1)
            for source in range(value_oracle.circuit.num_qubits)
        }
        value_gates = _remap_gates(value_oracle.circuit.gates, mapping)
        self.circuit.extend(value_gates)
        append_equality_to_constant(
            self.circuit,
            self.layout.value_wires,
            self.layout.target,
            self.layout.equality_work,
            self.observed_word,
        )
        self.circuit.append_inverse(value_gates)

    @property
    def input_bits(self) -> int:
        return self.value_oracle.input_bits

    def evaluate_predicate(self, input_word: int) -> int:
        return int(self.value_oracle.evaluate_input_word(input_word) == self.observed_word)

    def _pack_state(self, input_word: int, target: int) -> int:
        if target not in (0, 1):
            raise ValueError("target must be one bit")
        if input_word < 0 or input_word >= (1 << self.input_bits):
            raise ValueError("input_word is outside the candidate register")
        return int(input_word) | (int(target) << self.input_bits)

    def _extract(self, state: int) -> tuple[int, int, int]:
        mask = (1 << self.input_bits) - 1
        return (
            state & mask,
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
            self.circuit.apply_inverse_state(self._pack_state(input_word, output_word))
        )

    def marked_inputs(self) -> tuple[int, ...]:
        if self.input_bits > self.value_oracle.max_enumeration_bits:
            raise ValueError("marked-input enumeration exceeds max_enumeration_bits")
        return tuple(
            word
            for word in range(1 << self.input_bits)
            if self.evaluate_predicate(word)
        )

    def phase_sign(self, input_word: int) -> int:
        return -1 if self.apply(input_word, 0)[1] else 1

    def verify_basis_permutation(self) -> bool:
        if self.input_bits > self.value_oracle.max_enumeration_bits:
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
        return _resource_estimate(
            self.circuit,
            input_qubits=self.input_bits,
            output_qubits=1,
            work_qubits=len(self.layout.work_wires),
            synthesis=(
                "structure-preserving exact-gradient verifier: clean residual/product "
                "value oracle, full-word equality tree, and Bennett uncomputation"
            ),
        )


def compile_structure_preserving_gradient_value_oracle(
    spec: object,
    *,
    require_no_overflow: bool = True,
) -> ReversibleSingleRecordGradientValueOracle:
    """Lower a compatible ``SingleRecordGradientLeakageSpec`` to arithmetic gates."""

    input_format = getattr(spec, "input_format")
    target_format = getattr(spec, "target_format")
    gradient_format = getattr(spec, "gradient_format")
    if target_format != input_format:
        raise ValueError("the arithmetic gradient compiler requires equal input/target formats")
    if any(
        fmt.fractional_bits != 0
        for fmt in (input_format, target_format, gradient_format)
    ):
        raise ValueError("fractional gradient lowering is not yet supported")
    if not input_format.signed or not gradient_format.signed:
        raise ValueError("the arithmetic gradient compiler requires signed words")
    return ReversibleSingleRecordGradientValueOracle(
        getattr(spec, "weights"),
        getattr(spec, "bias"),
        input_bits=input_format.bits,
        gradient_bits=gradient_format.bits,
        require_no_overflow=require_no_overflow,
        max_enumeration_bits=getattr(spec, "max_enumeration_bits"),
    )


def run_structure_preserving_gradient_reconstruction(
    spec: object,
    inputs: Sequence[int],
    target: int,
    *,
    target_success: float = 0.8,
) -> object:
    """Run exact gradient search using the polynomial-size compiled phase oracle."""

    from .gradient_reconstruction import GradientReconstructionReport

    value_oracle = compile_structure_preserving_gradient_value_oracle(spec)
    true_candidate = getattr(spec, "encode_candidate")(inputs, target)
    observed = value_oracle.evaluate_input_word(true_candidate)
    verifier = ReversibleSingleRecordGradientEqualityOracle(value_oracle, observed)
    marked = verifier.marked_inputs()
    population = 1 << value_oracle.input_bits
    comparison = compare_search_queries(
        population, len(marked), target_success=target_success
    )
    iterations = optimal_standard_grover_iterations(population, len(marked)) or 0
    simulation = simulate_grover(verifier, iterations)
    reference_oracle = getattr(spec, "compile_value_oracle")()
    return GradientReconstructionReport(
        true_candidate=true_candidate,
        observed_word=observed,
        marked_candidates=marked,
        exact_original_identifiable=len(marked) == 1,
        finite_identifiability=analyze_finite_oracle(reference_oracle),
        target_fibre_size=len(marked),
        classical_queries=comparison.classical_queries,
        grover_queries=comparison.grover_queries,
        grover_success_probability=simulation.success_probability,
        grover_resources=estimate_grover_resources(verifier, iterations),
    )
