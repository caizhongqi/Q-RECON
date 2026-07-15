"""Reproducible benchmark generators and matched classical/quantum reports."""

from .fixed_point_mlp import (
    FixedPointMLPBenchmarkConfig,
    FixedPointMLPBenchmarkResult,
    FixedPointMLPInstance,
    build_fixed_point_mlp_instance,
    run_fixed_point_mlp_benchmark,
    run_fixed_point_mlp_benchmark_matrix,
)
from .statistics import (
    BenchmarkQualityGate,
    ConfidenceInterval,
    FixedPointMLPConfigSummary,
    FixedPointMLPMatrixSummary,
    LogLogScalingFit,
    ProportionSummary,
    ScalarSummary,
    benchmark_environment_manifest,
    fit_loglog_scaling,
    summarize_fixed_point_mlp_benchmark_matrix,
    summarize_proportion,
    summarize_scalar,
)

__all__ = [
    "BenchmarkQualityGate",
    "ConfidenceInterval",
    "FixedPointMLPBenchmarkConfig",
    "FixedPointMLPBenchmarkResult",
    "FixedPointMLPConfigSummary",
    "FixedPointMLPInstance",
    "FixedPointMLPMatrixSummary",
    "LogLogScalingFit",
    "ProportionSummary",
    "ScalarSummary",
    "benchmark_environment_manifest",
    "build_fixed_point_mlp_instance",
    "fit_loglog_scaling",
    "run_fixed_point_mlp_benchmark",
    "run_fixed_point_mlp_benchmark_matrix",
    "summarize_fixed_point_mlp_benchmark_matrix",
    "summarize_proportion",
    "summarize_scalar",
]
