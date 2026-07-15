from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from itertools import product
from operator import mul
from typing import Sequence

from .arithmetic import _resource_estimate
from .comparators import append_equality_to_constant
from .compiler import OracleResourceEstimate
from .fixed_point import FixedPointFormat
from .fixed_point_mlp import _remap_gates
from .fixed_point_mlp_equality import ReversibleFixedPointMLPEqualityOracle
from .models import QuantizedAffineLayer
from .reversible import ReversibleCircuit, ReversibleGate


@dataclass(frozen=True)
class ProductDomainLayout:
    input_registers: tuple[tuple[int, ...], ...]
    target: int
    membership_bits: tuple[int, ...]
    equality_work: tuple[int, ...]
    conjunction_work: tuple[int, ...]

    @property
    def input_wires(self) -> tuple[int, ...]:
        return tuple(wire for register in self.input_registers for wire in register)

    @property
    def work_wires(self) -> tuple[int, ...]:
        return self.membership_bits + self.equality_work + self.conjunction_work


class ReversibleProductDomainPredicateOracle:
    """Clean membership predicate for a finite product of feature-code domains.

    Each feature domain is compiled as a parity of mutually exclusive constant
    equalities.  The feature membership bits are ANDed into the target and then
    all membership computations are reversed.  Complexity is exponential only
    in the per-feature word width or explicit domain size, not in total feature
    dimension.
    """

    output_bits = 1

    def __init__(
        self,
        input_format: FixedPointFormat,
        domains: Sequence[Sequence[int]],
        *,
        max_enumeration_bits: int = 16,
    ) -> None:
        validated: list[tuple[int, ...]] = []
        for index, domain in enumerate(domains):
            values = tuple(sorted(set(int(value) for value in domain)))
            if not values:
                raise ValueError(f"domain {index} must be non-empty")
            for value in values:
                input_format.require_code(value)
            validated.append(values)
        if not validated:
            raise ValueError("at least one feature domain is required")
        if max_enumeration_bits <= 0:
            raise ValueError("max_enumeration_bits must be positive")

        self.input_format = input_format
        self.domains = tuple(validated)
        self.domain_sets = tuple(frozenset(domain) for domain in self.domains)
        self.feature_count = len(self.domains)
        self.max_enumeration_bits = int(max_enumeration_bits)

        offset = 0
        input_registers: list[tuple[int, ...]] = []
        for _ in self.domains:
            register = tuple(range(offset, offset + input_format.bits))
            input_registers.append(register)
            offset += input_format.bits
        target = offset
        offset += 1
        membership_bits = tuple(range(offset, offset + self.feature_count))
        offset += len(membership_bits)
        equality_work = tuple(
            range(offset, offset + max(0, input_format.bits - 2))
        )
        offset += len(equality_work)
        conjunction_work = tuple(
            range(offset, offset + max(0, self.feature_count - 2))
        )
        offset += len(conjunction_work)
        self.layout = ProductDomainLayout(
            tuple(input_registers),
            target,
            membership_bits,
            equality_work,
            conjunction_work,
        )
        self.circuit = ReversibleCircuit(offset)

        membership_gates: list[tuple[ReversibleGate, ...]] = []
        for register, membership, domain in zip(
            self.layout.input_registers,
            self.layout.membership_bits,
            self.domains,
        ):
            start = len(self.circuit.gates)
            for code in domain:
                append_equality_to_constant(
                    self.circuit,
                    register,
                    membership,
                    self.layout.equality_work,
                    input_format.code_to_word(code),
                )
            membership_gates.append(self.circuit.gates[start:])

        append_equality_to_constant(
            self.circuit,
            self.layout.membership_bits,
            self.layout.target,
            self.layout.conjunction_work,
            (1 << self.feature_count) - 1,
        )
        for gates in reversed(membership_gates):
            self.circuit.append_inverse(gates)

    @property
    def input_bits(self) -> int:
        return self.feature_count * self.input_format.bits

    @property
    def candidate_count(self) -> int:
        return reduce(mul, (len(domain) for domain in self.domains), 1)

    def encode_inputs(self, values: Sequence[int]) -> int:
        codes = tuple(int(value) for value in values)
        if len(codes) != self.feature_count:
            raise ValueError(f"expected {self.feature_count} feature codes")
        word = 0
        for index, code in enumerate(codes):
            self.input_format.require_code(code)
            word |= self.input_format.code_to_word(code) << (
                index * self.input_format.bits
            )
        return word

    def decode_input_word(self, input_word: int) -> tuple[int, ...]:
        word = int(input_word)
        if word < 0 or word >= (1 << self.input_bits):
            raise ValueError("input_word is outside the candidate register")
        mask = self.input_format.mask
        return tuple(
            self.input_format.word_to_code(
                (word >> (index * self.input_format.bits)) & mask
            )
            for index in range(self.feature_count)
        )

    def evaluate_predicate(self, input_word: int) -> int:
        values = self.decode_input_word(input_word)
        return int(
            all(value in domain for value, domain in zip(values, self.domain_sets))
        )

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
        return tuple(
            sorted(self.encode_inputs(values) for values in product(*self.domains))
        )

    def phase_sign(self, input_word: int) -> int:
        return -1 if self.apply(input_word, 0)[1] else 1

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
                "finite product-domain predicate: per-feature clean constant "
                "equalities, clean feature-membership conjunction, and reverse "
                "membership cleanup"
            ),
        )


@dataclass(frozen=True)
class DomainRestrictedMLPEqualityLayout:
    input_wires: tuple[int, ...]
    target: int
    domain_bit: int
    equality_bit: int
    domain_work: tuple[int, ...]
    equality_work: tuple[int, ...]

    @property
    def work_wires(self) -> tuple[int, ...]:
        return (
            (self.domain_bit, self.equality_bit)
            + self.domain_work
            + self.equality_work
        )


class ReversibleDomainRestrictedMLPEqualityOracle:
    """AND a product-domain predicate with exact fixed-point MLP output equality."""

    output_bits = 1

    def __init__(
        self,
        hidden_layer: QuantizedAffineLayer,
        output_layer: QuantizedAffineLayer,
        target_codes: Sequence[int],
        domains: Sequence[Sequence[int]],
        *,
        require_no_overflow: bool = True,
        max_enumeration_bits: int = 16,
    ) -> None:
        self.domain = ReversibleProductDomainPredicateOracle(
            hidden_layer.input_format,
            domains,
            max_enumeration_bits=max_enumeration_bits,
        )
        if self.domain.feature_count != hidden_layer.input_dimension:
            raise ValueError("domain dimension does not match MLP input")
        self.equality = ReversibleFixedPointMLPEqualityOracle(
            hidden_layer,
            output_layer,
            target_codes,
            require_no_overflow=require_no_overflow,
            max_enumeration_bits=max_enumeration_bits,
        )
        if self.domain.input_bits != self.equality.input_bits:
            raise ValueError("domain and equality predicates use different input widths")
        self.max_enumeration_bits = int(max_enumeration_bits)

        offset = self.input_bits
        target = offset
        offset += 1
        domain_bit = offset
        offset += 1
        equality_bit = offset
        offset += 1
        domain_work = tuple(
            range(offset, offset + len(self.domain.layout.work_wires))
        )
        offset += len(domain_work)
        equality_work = tuple(
            range(offset, offset + len(self.equality.layout.work_wires))
        )
        offset += len(equality_work)
        self.layout = DomainRestrictedMLPEqualityLayout(
            tuple(range(self.input_bits)),
            target,
            domain_bit,
            equality_bit,
            domain_work,
            equality_work,
        )
        self.circuit = ReversibleCircuit(offset)

        domain_mapping: dict[int, int] = {}
        domain_mapping.update(
            zip(self.domain.layout.input_wires, self.layout.input_wires)
        )
        domain_mapping[self.domain.layout.target] = domain_bit
        domain_mapping.update(
            zip(self.domain.layout.work_wires, domain_work)
        )
        domain_gates = _remap_gates(self.domain.circuit.gates, domain_mapping)

        equality_mapping: dict[int, int] = {}
        equality_mapping.update(
            zip(self.equality.layout.input_wires, self.layout.input_wires)
        )
        equality_mapping[self.equality.layout.target] = equality_bit
        equality_mapping.update(
            zip(self.equality.layout.work_wires, equality_work)
        )
        equality_gates = _remap_gates(
            self.equality.circuit.gates, equality_mapping
        )

        self.circuit.extend(domain_gates)
        self.circuit.extend(equality_gates)
        self.circuit.ccx(domain_bit, equality_bit, target)
        self.circuit.append_inverse(equality_gates)
        self.circuit.append_inverse(domain_gates)

    @property
    def input_bits(self) -> int:
        return self.domain.input_bits

    @property
    def domains(self) -> tuple[tuple[int, ...], ...]:
        return self.domain.domains

    @property
    def candidate_count(self) -> int:
        return self.domain.candidate_count

    def encode_inputs(self, values: Sequence[int]) -> int:
        return self.domain.encode_inputs(values)

    def evaluate_predicate(self, input_word: int) -> int:
        return int(
            self.domain.evaluate_predicate(input_word)
            and self.equality.evaluate_predicate(input_word)
        )

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
        return tuple(
            word
            for word in self.domain.marked_inputs()
            if self.equality.evaluate_predicate(word)
        )

    def phase_sign(self, input_word: int) -> int:
        return -1 if self.apply(input_word, 0)[1] else 1

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
                "domain-restricted fixed-point MLP equality predicate: clean "
                "product-domain membership, clean exact-output equality, AND, "
                "and complete reverse cleanup"
            ),
        )
