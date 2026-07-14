"""Bit-exact quantized models and coherent-oracle compiler backends."""

from .analysis import FiniteIdentifiabilityReport, analyze_finite_oracle
from .anf import ANFOracle, MonomialGate, SynthesisComparison, compare_exact_syntheses
from .arithmetic import (
    AffineRangeReport,
    AffineRowRange,
    ReversibleIntegerAffinePredicateOracle,
    ReversibleIntegerAffineValueOracle,
    affine_range_report,
    append_cdkm_fixed_adder,
    compile_structure_preserving_affine_oracle,
    compile_structure_preserving_threshold_oracle,
)
from .compiler import (
    MintermGate,
    OracleResourceEstimate,
    TruthTableOracle,
    compile_model_value_oracle,
    compile_verifier_oracle,
)
from .costing import (
    ClassicalSearchCosts,
    EndToEndSearchCostReport,
    FaultTolerantGateCosts,
    QuantumSearchCosts,
    QuantumSearchPlan,
    compare_end_to_end_search_costs,
    maximum_t_cost_for_fixed_plan,
    minimum_instances_for_fixed_plan_advantage,
    optimize_quantum_search_plan,
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
from .reversible import (
    ReversibleCircuit,
    ReversibleGate,
    pack_register,
    unpack_register,
)

__all__ = [
    "ANFOracle",
    "AffineRangeReport",
    "AffineRowRange",
    "ClassicalSearchCosts",
    "EndToEndSearchCostReport",
    "FaultTolerantGateCosts",
    "FiniteIdentifiabilityReport",
    "FixedPointFormat",
    "GroverResourceEstimate",
    "GroverSimulationResult",
    "LayerRangeReport",
    "MintermGate",
    "MonomialGate",
    "NetworkRangeReport",
    "OracleResourceEstimate",
    "QuantumSearchCosts",
    "QuantumSearchPlan",
    "QuantizedAffineLayer",
    "QuantizedNetwork",
    "ReversibleCircuit",
    "ReversibleGate",
    "ReversibleIntegerAffinePredicateOracle",
    "ReversibleIntegerAffineValueOracle",
    "SynthesisComparison",
    "TruthTableOracle",
    "affine_range_report",
    "analyze_finite_oracle",
    "append_cdkm_fixed_adder",
    "compare_end_to_end_search_costs",
    "compare_exact_syntheses",
    "compile_model_value_oracle",
    "compile_structure_preserving_affine_oracle",
    "compile_structure_preserving_threshold_oracle",
    "compile_verifier_oracle",
    "estimate_grover_resources",
    "maximum_t_cost_for_fixed_plan",
    "minimum_instances_for_fixed_plan_advantage",
    "optimize_quantum_search_plan",
    "pack_register",
    "quantized_binary_logistic_regression",
    "rescale_code",
    "round_half_away_from_zero",
    "round_shift_right",
    "simulate_grover",
    "unpack_register",
]
