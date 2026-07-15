from __future__ import annotations

import argparse
import json

from qrecon.benchmarks import (
    FixedPointMLPBenchmarkConfig,
    run_fixed_point_mlp_benchmark_matrix,
    summarize_fixed_point_mlp_benchmark_matrix,
)


def _configs() -> tuple[FixedPointMLPBenchmarkConfig, ...]:
    return tuple(
        FixedPointMLPBenchmarkConfig(
            input_dimension=dimension,
            hidden_width=2,
            output_dimension=2,
            input_bits=2,
            fractional_bits=0,
            domain_codes=(-1, 0, 1),
            max_basis_verification_bits=6,
        )
        for dimension in (1, 2, 3)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--z3",
        action="store_true",
        help="run the optional exact SMT baseline",
    )
    parser.add_argument(
        "--publication",
        action="store_true",
        help="use the larger predeclared seed/bootstrap thresholds",
    )
    args = parser.parse_args()

    if args.publication:
        seeds = tuple(range(20))
        bootstrap_samples = 5000
        minimum_seeds = 20
        protocol = "publication-oriented"
    else:
        seeds = (3, 5, 7)
        bootstrap_samples = 250
        minimum_seeds = 3
        protocol = "ci-smoke"

    results = run_fixed_point_mlp_benchmark_matrix(
        _configs(),
        seeds,
        use_z3=args.z3,
    )
    report = summarize_fixed_point_mlp_benchmark_matrix(
        results,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=20260715,
        minimum_seeds_per_config=minimum_seeds,
        minimum_candidate_scales=3,
    )
    payload = {
        "protocol": protocol,
        "z3_enabled": args.z3,
        "summary": report.to_dict(),
        "results": [result.to_dict() for result in results],
        "smoke_warning": (
            None
            if args.publication
            else "CI smoke thresholds validate the reporting pipeline only; they are not the "
            "minimum evidence package for a paper claim."
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
