"""Bit-exact quantized models and finite coherent-oracle baselines."""

from .analysis import FiniteIdentifiabilityReport, analyze_finite_oracle
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
    "FiniteIdentifiabilityReport",
    "FixedPointFormat",
    "GroverResourceEstimate",
    "GroverSimulationResult",
    "LayerRangeReport",
    "MintermGate",
    "NetworkRangeReport",
    "OracleResourceEstimate",
    "QuantizedAffineLayer",
    "QuantizedNetwork",
    "TruthTableOracle",
    "analyze_finite_oracle",
    "compile_model_value_oracle",
    "compile_verifier_oracle",
    "estimate_grover_resources",
    "quantized_binary_logistic_regression",
    "rescale_code",
    "round_half_away_from_zero",
    "round_shift_right",
    "simulate_grover",
]
