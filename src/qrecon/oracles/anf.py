from __future__ import annotations

from dataclasses import dataclass

from .compiler import OracleResourceEstimate, TruthTableOracle


@dataclass(frozen=True)
class MonomialGate:
    """Positive-control multi-controlled X implementing one ANF monomial."""

    control_mask: int
    target_output_bit: int

    @property
    def controls(self) -> int:
        return int(self.control_mask).bit_count()

    def matches(self, input_word: int) -> bool:
        return (int(input_word) & self.control_mask) == self.control_mask

    def apply(self, input_word: int, output_word: int) -> int:
        if self.matches(input_word):
            return int(output_word) ^ (1 << self.target_output_bit)
        return int(output_word)


def _mobius_coefficients(values: list[int], input_bits: int) -> tuple[int, ...]:
    coefficients = [int(value) & 1 for value in values]
    for variable in range(input_bits):
        bit = 1 << variable
        for mask in range(1 << input_bits):
            if mask & bit:
                coefficients[mask] ^= coefficients[mask ^ bit]
    return tuple(coefficients)


@dataclass(frozen=True)
class ANFOracle:
    """Exact clean oracle synthesized from algebraic normal forms over GF(2)."""

    input_bits: int
    output_bits: int
    table: tuple[int, ...]
    monomials: tuple[tuple[int, ...], ...]
    name: str = "anf_oracle"

    def __post_init__(self) -> None:
        table = tuple(int(value) for value in self.table)
        monomials = tuple(tuple(int(mask) for mask in output) for output in self.monomials)
        object.__setattr__(self, "table", table)
        object.__setattr__(self, "monomials", monomials)
        if self.input_bits <= 0 or self.output_bits <= 0:
            raise ValueError("input_bits and output_bits must be positive")
        if len(table) != (1 << self.input_bits):
            raise ValueError("table length does not match input_bits")
        if len(monomials) != self.output_bits:
            raise ValueError("one monomial list is required per output bit")
        output_limit = 1 << self.output_bits
        input_limit = 1 << self.input_bits
        if any(value < 0 or value >= output_limit for value in table):
            raise ValueError("table output does not fit output_bits")
        if any(mask < 0 or mask >= input_limit for output in monomials for mask in output):
            raise ValueError("monomial control mask does not fit input_bits")

    @classmethod
    def from_truth_table(cls, oracle: TruthTableOracle) -> "ANFOracle":
        outputs: list[tuple[int, ...]] = []
        for output_bit in range(oracle.output_bits):
            values = [(word >> output_bit) & 1 for word in oracle.table]
            coefficients = _mobius_coefficients(values, oracle.input_bits)
            outputs.append(tuple(mask for mask, coefficient in enumerate(coefficients) if coefficient))
        return cls(
            input_bits=oracle.input_bits,
            output_bits=oracle.output_bits,
            table=oracle.table,
            monomials=tuple(outputs),
            name=f"{oracle.name}_anf",
        )

    @property
    def gates(self) -> tuple[MonomialGate, ...]:
        return tuple(
            MonomialGate(mask, output_bit)
            for output_bit, masks in enumerate(self.monomials)
            for mask in masks
        )

    def evaluate_polynomial(self, input_word: int) -> int:
        input_value = int(input_word)
        if input_value < 0 or input_value >= (1 << self.input_bits):
            raise ValueError("input_word is outside the input register")
        output = 0
        for gate in self.gates:
            output = gate.apply(input_value, output)
        return output

    def apply(
        self, input_word: int, output_word: int = 0, ancillas: int = 0
    ) -> tuple[int, int, int]:
        input_value = int(input_word)
        output_value = int(output_word)
        if input_value < 0 or input_value >= (1 << self.input_bits):
            raise ValueError("input_word is outside the input register")
        if output_value < 0 or output_value >= (1 << self.output_bits):
            raise ValueError("output_word is outside the output register")
        if ancillas != 0:
            raise ValueError("clean oracle requires ancillas initialized to zero")
        for gate in self.gates:
            output_value = gate.apply(input_value, output_value)
        return input_value, output_value, 0

    def inverse_apply(
        self, input_word: int, output_word: int = 0, ancillas: int = 0
    ) -> tuple[int, int, int]:
        input_value = int(input_word)
        output_value = int(output_word)
        if input_value < 0 or input_value >= (1 << self.input_bits):
            raise ValueError("input_word is outside the input register")
        if output_value < 0 or output_value >= (1 << self.output_bits):
            raise ValueError("output_word is outside the output register")
        if ancillas != 0:
            raise ValueError("clean oracle requires ancillas initialized to zero")
        for gate in reversed(self.gates):
            output_value = gate.apply(input_value, output_value)
        return input_value, output_value, 0

    def verify_reference_equivalence(self) -> bool:
        return all(
            self.evaluate_polynomial(input_word) == expected
            for input_word, expected in enumerate(self.table)
        )

    def verify_basis_permutation(self, *, exhaustive_output_words: bool = True) -> bool:
        output_words = range(1 << self.output_bits) if exhaustive_output_words else (0,)
        seen: set[tuple[int, int]] = set()
        for input_word in range(1 << self.input_bits):
            for output_word in output_words:
                output = self.apply(input_word, output_word)
                restored = self.inverse_apply(*output)
                if restored != (input_word, output_word, 0):
                    return False
                key = (output[0], output[1])
                if key in seen:
                    return False
                seen.add(key)
        return True

    def marked_inputs(self) -> tuple[int, ...]:
        if self.output_bits != 1:
            raise ValueError("marked_inputs requires a one-bit predicate oracle")
        return tuple(index for index, value in enumerate(self.table) if value == 1)

    def phase_sign(self, input_word: int) -> int:
        if self.output_bits != 1:
            raise ValueError("phase_sign requires a one-bit predicate oracle")
        return -1 if self.evaluate_polynomial(input_word) else 1

    def resource_estimate(self, *, phase_kickback: bool = False) -> OracleResourceEstimate:
        x_gates = 0
        cnot_gates = 0
        toffoli_gates = 0
        peak_ancillas = 0
        logical_depth = 0
        for gate in self.gates:
            controls = gate.controls
            if controls == 0:
                x_gates += 1
                logical_depth += 1
            elif controls == 1:
                cnot_gates += 1
                logical_depth += 1
            else:
                count = 2 * controls - 3
                toffoli_gates += count
                peak_ancillas = max(peak_ancillas, controls - 2)
                logical_depth += count
        output_qubits = 1 if phase_kickback else self.output_bits
        return OracleResourceEstimate(
            input_qubits=self.input_bits,
            output_qubits=output_qubits,
            peak_clean_ancillas=peak_ancillas,
            logical_qubits=self.input_bits + output_qubits + peak_ancillas,
            controlled_x_terms=len(self.gates),
            negative_control_x_gates=0,
            x_gates=x_gates,
            cnot_gates=cnot_gates,
            toffoli_gates=toffoli_gates,
            h_gates=0,
            z_gates=0,
            t_count_upper_bound=7 * toffoli_gates,
            t_depth_upper_bound=3 * toffoli_gates,
            logical_depth_upper_bound=logical_depth,
            synthesis=(
                "algebraic normal form over GF(2); one positive-control monomial "
                "gate per nonzero coefficient; k>=2 uses k-2 clean ancillas and "
                "2k-3 Toffolis"
            ),
        )


@dataclass(frozen=True)
class SynthesisComparison:
    minterm: OracleResourceEstimate
    anf: OracleResourceEstimate
    selected: str


def compare_exact_syntheses(oracle: TruthTableOracle) -> SynthesisComparison:
    minterm = oracle.resource_estimate()
    anf_oracle = ANFOracle.from_truth_table(oracle)
    anf = anf_oracle.resource_estimate()
    minterm_key = (
        minterm.t_count_upper_bound,
        minterm.toffoli_gates,
        minterm.controlled_x_terms,
        minterm.logical_depth_upper_bound,
    )
    anf_key = (
        anf.t_count_upper_bound,
        anf.toffoli_gates,
        anf.controlled_x_terms,
        anf.logical_depth_upper_bound,
    )
    return SynthesisComparison(
        minterm=minterm,
        anf=anf,
        selected="anf" if anf_key < minterm_key else "minterm",
    )
