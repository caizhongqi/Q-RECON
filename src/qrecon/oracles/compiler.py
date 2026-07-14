from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Callable, Literal

from .models import QuantizedNetwork

VerifierMetric = Literal["exact", "hamming", "absolute"]


@dataclass(frozen=True)
class MintermGate:
    """An abstract mixed-polarity multi-controlled X gate."""

    required_input: int
    input_bits: int
    target_output_bit: int

    @property
    def negative_controls(self) -> int:
        return self.input_bits - int(self.required_input).bit_count()

    def matches(self, input_word: int) -> bool:
        return int(input_word) == self.required_input

    def apply(self, input_word: int, output_word: int) -> int:
        """Apply this self-inverse logical gate to a basis output word."""

        if self.matches(input_word):
            return int(output_word) ^ (1 << self.target_output_bit)
        return int(output_word)


@dataclass(frozen=True)
class OracleResourceEstimate:
    input_qubits: int
    output_qubits: int
    peak_clean_ancillas: int
    logical_qubits: int
    controlled_x_terms: int
    negative_control_x_gates: int
    x_gates: int
    cnot_gates: int
    toffoli_gates: int
    h_gates: int
    z_gates: int
    t_count_upper_bound: int
    t_depth_upper_bound: int
    logical_depth_upper_bound: int
    synthesis: str

    @property
    def minterm_gates(self) -> int:
        """Backward-compatible name for the number of emitted controlled-X terms."""

        return self.controlled_x_terms

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


@dataclass(frozen=True)
class TruthTableOracle:
    """Exact clean value oracle synthesized from a complete finite truth table.

    The logical netlist has one mixed-polarity minterm-controlled X for every set
    output bit. The input register is never modified, the output is XORed with
    ``f(x)``, and decomposition ancillas are assumed to be cleanly uncomputed.
    This baseline is exponential in input width by design; it is a correctness
    scaffold and resource upper bound, not an efficient arithmetic compiler.
    """

    input_bits: int
    output_bits: int
    table: tuple[int, ...]
    name: str = "truth_table_oracle"

    def __post_init__(self) -> None:
        table = tuple(int(value) for value in self.table)
        object.__setattr__(self, "table", table)
        if self.input_bits <= 0:
            raise ValueError("input_bits must be positive")
        if self.output_bits <= 0:
            raise ValueError("output_bits must be positive")
        expected = 1 << self.input_bits
        if len(table) != expected:
            raise ValueError(f"truth table must contain {expected} outputs")
        limit = 1 << self.output_bits
        if any(value < 0 or value >= limit for value in table):
            raise ValueError(f"all outputs must fit {self.output_bits} bits")

    @classmethod
    def from_function(
        cls,
        input_bits: int,
        output_bits: int,
        function: Callable[[int], int],
        *,
        max_input_bits: int = 16,
        name: str = "truth_table_oracle",
    ) -> "TruthTableOracle":
        if input_bits > max_input_bits:
            raise ValueError(
                f"refusing to enumerate {input_bits} input bits; "
                f"max_input_bits={max_input_bits}"
            )
        return cls(
            input_bits=input_bits,
            output_bits=output_bits,
            table=tuple(int(function(word)) for word in range(1 << input_bits)),
            name=name,
        )

    @property
    def gates(self) -> tuple[MintermGate, ...]:
        result: list[MintermGate] = []
        for input_word, output_word in enumerate(self.table):
            for output_bit in range(self.output_bits):
                if (output_word >> output_bit) & 1:
                    result.append(MintermGate(input_word, self.input_bits, output_bit))
        return tuple(result)

    @property
    def truth_table_sha256(self) -> str:
        width = max(1, (self.output_bits + 7) // 8)
        payload = b"".join(value.to_bytes(width, "little") for value in self.table)
        header = f"{self.input_bits}:{self.output_bits}:".encode("ascii")
        return hashlib.sha256(header + payload).hexdigest()

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

    def verify_basis_permutation(self, *, exhaustive_output_words: bool = True) -> bool:
        output_words = range(1 << self.output_bits) if exhaustive_output_words else (0,)
        seen: set[tuple[int, int]] = set()
        for input_word in range(1 << self.input_bits):
            for output_word in output_words:
                output = self.apply(input_word, output_word)
                restored = self.inverse_apply(output[0], output[1], output[2])
                if restored != (input_word, output_word, 0):
                    return False
                key = (output[0], output[1])
                if key in seen:
                    return False
                seen.add(key)
        return True

    def marked_inputs(self) -> tuple[int, ...]:
        if self.output_bits != 1 or any(value not in (0, 1) for value in self.table):
            raise ValueError("marked_inputs requires a one-bit predicate oracle")
        return tuple(index for index, value in enumerate(self.table) if value == 1)

    def phase_sign(self, input_word: int) -> int:
        if self.output_bits != 1:
            raise ValueError("phase_sign requires a one-bit predicate oracle")
        value = self.table[int(input_word)]
        return -1 if value else 1

    def resource_estimate(self, *, phase_kickback: bool = False) -> OracleResourceEstimate:
        gates = self.gates
        negative_x = 2 * sum(gate.negative_controls for gate in gates)
        x_gates = negative_x
        cnot_gates = 0
        toffoli_gates = 0
        logical_depth = 0
        peak_ancillas = 0

        for gate in gates:
            controls = gate.input_bits
            if controls == 0:
                x_gates += 1
                gate_depth = 1
            elif controls == 1:
                cnot_gates += 1
                gate_depth = 1
            else:
                count = 2 * controls - 3
                toffoli_gates += count
                peak_ancillas = max(peak_ancillas, controls - 2)
                gate_depth = count
            logical_depth += 2 * gate.negative_controls + gate_depth

        output_qubits = 1 if phase_kickback else self.output_bits
        logical_qubits = self.input_bits + output_qubits + peak_ancillas
        return OracleResourceEstimate(
            input_qubits=self.input_bits,
            output_qubits=output_qubits,
            peak_clean_ancillas=peak_ancillas,
            logical_qubits=logical_qubits,
            controlled_x_terms=len(gates),
            negative_control_x_gates=negative_x,
            x_gates=x_gates,
            cnot_gates=cnot_gates,
            toffoli_gates=toffoli_gates,
            h_gates=0,
            z_gates=0,
            t_count_upper_bound=7 * toffoli_gates,
            t_depth_upper_bound=3 * toffoli_gates,
            logical_depth_upper_bound=logical_depth,
            synthesis=(
                "naive mixed-polarity minterms; each k-controlled X with k>=2 "
                "uses k-2 clean ancillas and 2k-3 Toffolis; 7 T and T-depth 3 "
                "per exact Toffoli"
            ),
        )


def compile_model_value_oracle(
    model: QuantizedNetwork,
    *,
    max_input_bits: int = 16,
    require_no_overflow: bool = True,
) -> TruthTableOracle:
    report = model.range_report()
    if require_no_overflow and not report.no_overflow:
        raise OverflowError(
            "model range proof is not overflow-free; choose wider formats or "
            "compile with an explicit saturating contract"
        )
    return TruthTableOracle.from_function(
        model.input_bits,
        model.output_bits,
        model.evaluate_input_word,
        max_input_bits=max_input_bits,
        name="quantized_model_value_oracle",
    )


def compile_verifier_oracle(
    value_oracle: TruthTableOracle,
    target_word: int,
    *,
    metric: VerifierMetric = "exact",
    threshold: int = 0,
) -> TruthTableOracle:
    target = int(target_word)
    if target < 0 or target >= (1 << value_oracle.output_bits):
        raise ValueError("target_word does not fit the value-oracle output register")
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    if metric not in ("exact", "hamming", "absolute"):
        raise ValueError("unsupported verifier metric")

    def predicate(input_word: int) -> int:
        value = value_oracle.table[input_word]
        if metric == "exact":
            distance = int(value != target)
        elif metric == "hamming":
            distance = (value ^ target).bit_count()
        else:
            distance = abs(value - target)
        return int(distance <= threshold)

    return TruthTableOracle.from_function(
        value_oracle.input_bits,
        1,
        predicate,
        max_input_bits=value_oracle.input_bits,
        name=f"{value_oracle.name}_{metric}_verifier",
    )
