from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .compiler import OracleResourceEstimate, TruthTableOracle


@dataclass(frozen=True)
class GroverSimulationResult:
    iterations: int
    marked: int
    success_probability: float
    probabilities: tuple[float, ...]

    @property
    def most_likely_inputs(self) -> tuple[int, ...]:
        if not self.probabilities:
            return ()
        maximum = max(self.probabilities)
        return tuple(
            index
            for index, probability in enumerate(self.probabilities)
            if np.isclose(probability, maximum, atol=1e-12, rtol=0.0)
        )


@dataclass(frozen=True)
class GroverResourceEstimate:
    iterations: int
    oracle_calls: int
    logical_qubits: int
    state_preparation_h_gates: int
    total_h_gates: int
    total_x_gates: int
    total_cnot_gates: int
    total_toffoli_gates: int
    total_t_count_upper_bound: int
    total_t_depth_upper_bound: int
    oracle: OracleResourceEstimate
    assumptions: str

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["oracle"] = self.oracle.to_dict()
        return result


def simulate_grover(verifier: TruthTableOracle, iterations: int) -> GroverSimulationResult:
    """Exact state-vector simulation using the compiled predicate truth table."""

    if verifier.output_bits != 1:
        raise ValueError("Grover simulation requires a one-bit verifier")
    if iterations < 0:
        raise ValueError("iterations must be non-negative")
    population = 1 << verifier.input_bits
    amplitudes = np.full(population, 1.0 / np.sqrt(population), dtype=np.complex128)
    marked = verifier.marked_inputs()

    for _ in range(iterations):
        for candidate in marked:
            amplitudes[candidate] *= -1.0
        amplitudes = 2.0 * amplitudes.mean() - amplitudes

    probabilities_array = np.abs(amplitudes) ** 2
    success = float(probabilities_array[list(marked)].sum()) if marked else 0.0
    return GroverSimulationResult(
        iterations=iterations,
        marked=len(marked),
        success_probability=success,
        probabilities=tuple(float(value) for value in probabilities_array),
    )


def _diffusion_resources(input_bits: int) -> tuple[int, int, int, int, int, int]:
    """Return H, X, CNOT, Toffoli, T-count, T-depth for one diffusion step."""

    if input_bits <= 0:
        raise ValueError("input_bits must be positive")
    if input_bits == 1:
        # 2|+><+| - I equals X for a one-qubit search register.
        return 0, 1, 0, 0, 0, 0

    controls = input_bits - 1
    h_gates = 2 * input_bits + 2
    x_gates = 2 * input_bits
    cnot = 1 if controls == 1 else 0
    toffoli = 0 if controls < 2 else 2 * controls - 3
    return h_gates, x_gates, cnot, toffoli, 7 * toffoli, 3 * toffoli


def estimate_grover_resources(
    verifier: TruthTableOracle, iterations: int
) -> GroverResourceEstimate:
    if verifier.output_bits != 1:
        raise ValueError("Grover resources require a one-bit verifier")
    if iterations < 0:
        raise ValueError("iterations must be non-negative")

    oracle = verifier.resource_estimate(phase_kickback=True)
    diff_h, diff_x, diff_cnot, diff_toffoli, diff_t, diff_t_depth = _diffusion_resources(
        verifier.input_bits
    )
    # Prepare the search register in |+>^n and the kickback target in |->.
    state_h = verifier.input_bits + 1
    phase_target_x = 1
    total_h = state_h + iterations * (oracle.h_gates + diff_h)
    total_x = phase_target_x + iterations * (oracle.x_gates + diff_x)
    total_cnot = iterations * (oracle.cnot_gates + diff_cnot)
    total_toffoli = iterations * (oracle.toffoli_gates + diff_toffoli)
    total_t = iterations * (oracle.t_count_upper_bound + diff_t)
    total_t_depth = iterations * (oracle.t_depth_upper_bound + diff_t_depth)
    diffusion_ancillas = max(0, verifier.input_bits - 3)
    logical_qubits = max(
        oracle.logical_qubits,
        verifier.input_bits + 1 + diffusion_ancillas,
    )
    return GroverResourceEstimate(
        iterations=iterations,
        oracle_calls=iterations,
        logical_qubits=logical_qubits,
        state_preparation_h_gates=state_h,
        total_h_gates=total_h,
        total_x_gates=total_x,
        total_cnot_gates=total_cnot,
        total_toffoli_gates=total_toffoli,
        total_t_count_upper_bound=total_t,
        total_t_depth_upper_bound=total_t_depth,
        oracle=oracle,
        assumptions=(
            "uniform state preparation; phase kickback through one clean verifier "
            "per iteration; standard H-X-MCZ-X-H diffusion; exact Toffoli cost "
            "upper bounds inherited from the truth-table synthesis contract"
        ),
    )
