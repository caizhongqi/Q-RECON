"""Reproducible benchmark generators and matched classical/quantum reports."""

from .fixed_point_mlp import (
    FixedPointMLPBenchmarkConfig,
    FixedPointMLPBenchmarkResult,
    FixedPointMLPInstance,
    build_fixed_point_mlp_instance,
    run_fixed_point_mlp_benchmark,
    run_fixed_point_mlp_benchmark_matrix,
)

__all__ = [
    "FixedPointMLPBenchmarkConfig",
    "FixedPointMLPBenchmarkResult",
    "FixedPointMLPInstance",
    "build_fixed_point_mlp_instance",
    "run_fixed_point_mlp_benchmark",
    "run_fixed_point_mlp_benchmark_matrix",
]
