"""Bit-exact quantized models and finite coherent-oracle baselines."""

from .analysis import FiniteIdentifiabilityReport, analyze_finite_oracle
from .anf import ANFOracle, MonomialGate, SynthesisComparison, compare_exact_syntheses
from .compiler import (
    MintermGate,
    OracleResourceEstimate,
    TruthTableOracle,
    compile_model_value_oracle,
    compile_verifier_oracle,
)
from .fixed_point import (
    FixedPointFormat,
    rescale_code,
    round_half_away_from_zero,
    round_shift_right,
)
from .grover import (
    GroverResourceEstimate,
    GroverSimulationResult,
    estimate_grover_resources,
    simulate_grover,
)
from .models import (
    LayerRangeReport,
    NetworkRangeReport,
    QuantizedAffineLayer,
    QuantizedNetwork,
    quantized_binary_logistic_regression,
)

__all__ = [
    "ANFOracle",
    "FiniteIdentifiabilityReport",
    "FixedPointFormat",
    "GroverResourceEstimate",
    "GroverSimulationResult",
    "LayerRangeReport",
    "MintermGate",
    "MonomialGate",
    "NetworkRangeReport",
    "OracleResourceEstimate",
    "QuantizedAffineLayer",
    "QuantizedNetwork",
    "SynthesisComparison",
    "TruthTableOracle",
    "analyze_finite_oracle",
    "compare_exact_syntheses",
    "compile_model_value_oracle",
    "compile_verifier_oracle",
    "estimate_grover_resources",
    "quantized_binary_logistic_regression",
    "rescale_code",
    "round_half_away_from_zero",
    "round_shift_right",
    "simulate_grover",
]
