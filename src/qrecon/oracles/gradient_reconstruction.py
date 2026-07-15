from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Sequence

from qrecon.theory import compare_search_queries, optimal_standard_grover_iterations

from .analysis import FiniteIdentifiabilityReport, analyze_finite_oracle
from .compiler import TruthTableOracle, compile_verifier_oracle
from .fixed_point import FixedPointFormat
from .grover import GroverResourceEstimate, estimate_grover_resources, simulate_grover


@dataclass(frozen=True)
class GradientRangeReport:
    minimum: tuple[int, ...]
    maximum: tuple[int, ...]
    no_overflow: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SingleRecordGradientLeakageSpec:
    """Finite exact-gradient channel for one integer linear-regression record.

    For prediction ``z = w·x + b`` and loss ``(z-t)^2/2``, the released gradient is
    ``g_w = (z-t)x`` and ``g_b = z-t``. Candidate words contain both the private
    input vector and private target.
    """

    weights: tuple[int, ...]
    bias: int
    input_format: FixedPointFormat
    target_format: FixedPointFormat
    gradient_format: FixedPointFormat
    max_enumeration_bits: int = 16

    def __post_init__(self) -> None:
        weights = tuple(int(weight) for weight in self.weights)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "bias", int(self.bias))
        if not weights:
            raise ValueError("weights must contain at least one feature")
        if any(
            fmt.fractional_bits != 0
            for fmt in (self.input_format, self.target_format, self.gradient_format)
        ):
            raise ValueError("the exact gradient benchmark currently requires integer formats")
        if not self.input_format.signed or not self.target_format.signed:
            raise ValueError("input and target formats must be signed")
        if not self.gradient_format.signed:
            raise ValueError("gradient format must be signed")
        if self.candidate_bits > int(self.max_enumeration_bits):
            raise ValueError(
                "candidate space exceeds max_enumeration_bits; enlarge the explicit limit"
            )
        report = self.range_report()
        if not report.no_overflow:
            raise OverflowError("gradient_format does not contain every candidate gradient")

    @property
    def feature_count(self) -> int:
        return len(self.weights)

    @property
    def candidate_bits(self) -> int:
        return self.feature_count * self.input_format.bits + self.target_format.bits

    @property
    def observation_components(self) -> int:
        return self.feature_count + 1

    @property
    def observation_bits(self) -> int:
        return self.observation_components * self.gradient_format.bits

    @property
    def population(self) -> int:
        return 1 << self.candidate_bits

    def encode_candidate(self, inputs: Sequence[int], target: int) -> int:
        values = tuple(self.input_format.require_code(value) for value in inputs)
        if len(values) != self.feature_count:
            raise ValueError(f"expected {self.feature_count} input values")
        target_value = self.target_format.require_code(target)
        word = 0
        for index, value in enumerate(values):
            word |= self.input_format.code_to_word(value) << (
                index * self.input_format.bits
            )
        word |= self.target_format.code_to_word(target_value) << (
            self.feature_count * self.input_format.bits
        )
        return word

    def decode_candidate(self, candidate_word: int) -> tuple[tuple[int, ...], int]:
        word = int(candidate_word)
        if word < 0 or word >= self.population:
            raise ValueError("candidate_word is outside the candidate register")
        mask = self.input_format.mask
        inputs = tuple(
            self.input_format.word_to_code(
                (word >> (index * self.input_format.bits)) & mask
            )
            for index in range(self.feature_count)
        )
        target_word = word >> (self.feature_count * self.input_format.bits)
        return inputs, self.target_format.word_to_code(target_word)

    def gradient_components(self, candidate_word: int) -> tuple[int, ...]:
        inputs, target = self.decode_candidate(candidate_word)
        prediction = sum(weight * value for weight, value in zip(self.weights, inputs))
        residual = prediction + self.bias - target
        return tuple(residual * value for value in inputs) + (residual,)

    def pack_observation(self, components: Sequence[int]) -> int:
        values = tuple(self.gradient_format.require_code(value) for value in components)
        if len(values) != self.observation_components:
            raise ValueError(f"expected {self.observation_components} gradient components")
        word = 0
        for index, value in enumerate(values):
            word |= self.gradient_format.code_to_word(value) << (
                index * self.gradient_format.bits
            )
        return word

    def unpack_observation(self, observation_word: int) -> tuple[int, ...]:
        word = int(observation_word)
        if word < 0 or word >= (1 << self.observation_bits):
            raise ValueError("observation_word is outside the gradient register")
        mask = self.gradient_format.mask
        return tuple(
            self.gradient_format.word_to_code(
                (word >> (index * self.gradient_format.bits)) & mask
            )
            for index in range(self.observation_components)
        )

    def observe_word(self, candidate_word: int) -> int:
        return self.pack_observation(self.gradient_components(candidate_word))

    def range_report(self) -> GradientRangeReport:
        minima = [math.inf] * self.observation_components
        maxima = [-math.inf] * self.observation_components
        safe = True
        for candidate in range(self.population):
            values = self.gradient_components(candidate)
            for index, value in enumerate(values):
                minima[index] = min(minima[index], value)
                maxima[index] = max(maxima[index], value)
                safe = safe and self.gradient_format.contains(value)
        return GradientRangeReport(
            tuple(int(value) for value in minima),
            tuple(int(value) for value in maxima),
            safe,
        )

    def compile_value_oracle(self) -> TruthTableOracle:
        return TruthTableOracle.from_function(
            self.candidate_bits,
            self.observation_bits,
            self.observe_word,
            max_input_bits=self.max_enumeration_bits,
            name="single_record_linear_gradient_value_oracle",
        )


def recover_single_record_from_full_gradient(
    weights: Sequence[int],
    bias: int,
    weight_gradient: Sequence[int],
    bias_gradient: int,
) -> tuple[tuple[int, ...], int] | None:
    """Analytically invert the exact one-record linear-regression gradient.

    A zero bias gradient is observation-degenerate and returns ``None``. Nonzero
    gradients are accepted only when every weight-gradient component is exactly
    divisible by the residual.
    """

    model_weights = tuple(int(value) for value in weights)
    gradient = tuple(int(value) for value in weight_gradient)
    residual = int(bias_gradient)
    if len(model_weights) != len(gradient):
        raise ValueError("weights and weight_gradient dimensions differ")
    if residual == 0:
        return None
    if any(value % residual for value in gradient):
        return None
    inputs = tuple(value // residual for value in gradient)
    target = sum(weight * value for weight, value in zip(model_weights, inputs))
    target += int(bias) - residual
    return inputs, target


@dataclass(frozen=True)
class GradientReconstructionReport:
    true_candidate: int
    observed_word: int
    marked_candidates: tuple[int, ...]
    exact_original_identifiable: bool
    finite_identifiability: FiniteIdentifiabilityReport
    target_fibre_size: int
    classical_queries: int | None
    grover_queries: int | None
    grover_success_probability: float
    grover_resources: GroverResourceEstimate

    def to_dict(self) -> dict[str, object]:
        return {
            "true_candidate": self.true_candidate,
            "observed_word": self.observed_word,
            "marked_candidates": list(self.marked_candidates),
            "exact_original_identifiable": self.exact_original_identifiable,
            "finite_identifiability": self.finite_identifiability.to_dict(),
            "target_fibre_size": self.target_fibre_size,
            "classical_queries": self.classical_queries,
            "grover_queries": self.grover_queries,
            "grover_success_probability": self.grover_success_probability,
            "grover_resources": self.grover_resources.to_dict(),
        }


def run_single_record_gradient_reconstruction(
    spec: SingleRecordGradientLeakageSpec,
    inputs: Sequence[int],
    target: int,
    *,
    target_success: float = 0.8,
) -> GradientReconstructionReport:
    value_oracle = spec.compile_value_oracle()
    true_candidate = spec.encode_candidate(inputs, target)
    observed = value_oracle.table[true_candidate]
    verifier = compile_verifier_oracle(value_oracle, observed, metric="exact")
    marked = verifier.marked_inputs()
    comparison = compare_search_queries(
        spec.population, len(marked), target_success=target_success
    )
    iterations = optimal_standard_grover_iterations(spec.population, len(marked)) or 0
    simulation = simulate_grover(verifier, iterations)
    return GradientReconstructionReport(
        true_candidate=true_candidate,
        observed_word=observed,
        marked_candidates=marked,
        exact_original_identifiable=len(marked) == 1,
        finite_identifiability=analyze_finite_oracle(value_oracle),
        target_fibre_size=len(marked),
        classical_queries=comparison.classical_queries,
        grover_queries=comparison.grover_queries,
        grover_success_probability=simulation.success_probability,
        grover_resources=estimate_grover_resources(verifier, iterations),
    )
