from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Literal

GateKind = Literal["x", "cx", "ccx"]


@dataclass(frozen=True)
class ReversibleGate:
    """One exact computational-basis X, CNOT, or Toffoli operation."""

    kind: GateKind
    wires: tuple[int, ...]

    def __post_init__(self) -> None:
        expected = {"x": 1, "cx": 2, "ccx": 3}.get(self.kind)
        if expected is None:
            raise ValueError(f"unsupported reversible gate: {self.kind}")
        if len(self.wires) != expected:
            raise ValueError(f"{self.kind} requires {expected} wires")
        if any(wire < 0 for wire in self.wires):
            raise ValueError("wire indices must be non-negative")
        if len(set(self.wires)) != len(self.wires):
            raise ValueError("a gate cannot reuse the same wire")

    @property
    def controls(self) -> tuple[int, ...]:
        return self.wires[:-1]

    @property
    def target(self) -> int:
        return self.wires[-1]

    def apply(self, state: int) -> int:
        value = int(state)
        enabled = all((value >> control) & 1 for control in self.controls)
        return value ^ (1 << self.target) if enabled else value


class ReversibleCircuit:
    """Small exact reversible circuit IR with basis-state execution and inversion."""

    def __init__(self, num_qubits: int, gates: Iterable[ReversibleGate] = ()) -> None:
        if num_qubits <= 0:
            raise ValueError("num_qubits must be positive")
        self.num_qubits = int(num_qubits)
        self._gates: list[ReversibleGate] = []
        self.extend(gates)

    @property
    def gates(self) -> tuple[ReversibleGate, ...]:
        return tuple(self._gates)

    def append(self, gate: ReversibleGate) -> None:
        if any(wire >= self.num_qubits for wire in gate.wires):
            raise ValueError("gate wire is outside the circuit")
        self._gates.append(gate)

    def x(self, target: int) -> None:
        self.append(ReversibleGate("x", (int(target),)))

    def cx(self, control: int, target: int) -> None:
        self.append(ReversibleGate("cx", (int(control), int(target))))

    def ccx(self, first: int, second: int, target: int) -> None:
        self.append(ReversibleGate("ccx", (int(first), int(second), int(target))))

    def extend(self, gates: Iterable[ReversibleGate]) -> None:
        for gate in gates:
            self.append(gate)

    def append_inverse(self, gates: Iterable[ReversibleGate]) -> None:
        self.extend(reversed(tuple(gates)))

    def apply_state(self, state: int) -> int:
        value = int(state)
        if value < 0 or value >= (1 << self.num_qubits):
            raise ValueError("basis state does not fit the circuit")
        for gate in self._gates:
            value = gate.apply(value)
        return value

    def apply_inverse_state(self, state: int) -> int:
        value = int(state)
        if value < 0 or value >= (1 << self.num_qubits):
            raise ValueError("basis state does not fit the circuit")
        for gate in reversed(self._gates):
            value = gate.apply(value)
        return value

    def inverse(self) -> "ReversibleCircuit":
        return ReversibleCircuit(self.num_qubits, reversed(self._gates))

    def gate_counts(self) -> dict[str, int]:
        counts = Counter(gate.kind for gate in self._gates)
        return {kind: int(counts.get(kind, 0)) for kind in ("x", "cx", "ccx")}

    def logical_depth(self) -> int:
        """ASAP depth assuming gates on disjoint wires may execute in parallel."""

        wire_depth = [0] * self.num_qubits
        for gate in self._gates:
            level = 1 + max(wire_depth[wire] for wire in gate.wires)
            for wire in gate.wires:
                wire_depth[wire] = level
        return max(wire_depth, default=0)


def pack_register(state: int, wires: tuple[int, ...] | list[int], word: int) -> int:
    """Set a little-endian register in a basis-state integer."""

    value = int(word)
    if value < 0 or value >= (1 << len(wires)):
        raise ValueError(f"word must fit {len(wires)} bits")
    result = int(state)
    for index, wire in enumerate(wires):
        mask = 1 << int(wire)
        result = result | mask if ((value >> index) & 1) else result & ~mask
    return result


def unpack_register(state: int, wires: tuple[int, ...] | list[int]) -> int:
    """Read a little-endian register from a basis-state integer."""

    value = int(state)
    return sum(((value >> int(wire)) & 1) << index for index, wire in enumerate(wires))
