from __future__ import annotations

import argparse
import json
import statistics

from qrecon.benchmarks import (
    FixedPointMLPBenchmarkConfig,
    run_fixed_point_mlp_benchmark_matrix,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--z3",
        action="store_true",
        help="run the optional exact SMT baseline",
    )
    args = parser.parse_args()

    configs = (
        FixedPointMLPBenchmarkConfig(
            input_dimension=2,
            hidden_width=2,
            output_dimension=2,
            input_bits=2,
            fractional_bits=0,
            domain_codes=(-1, 0, 1),
            max_basis_verification_bits=4,
        ),
        FixedPointMLPBenchmarkConfig(
            input_dimension=2,
            hidden_width=3,
            output_dimension=2,
            input_bits=2,
            fractional_bits=0,
            domain_codes=(-1, 0, 1),
            max_basis_verification_bits=4,
        ),
    )
    results = run_fixed_point_mlp_benchmark_matrix(
        configs,
        seeds=(3, 5, 7),
        use_z3=args.z3,
    )
    payloads = [result.to_dict() for result in results]
    summary = {
        "instances": len(results),
        "z3_enabled": args.z3,
        "all_classical_oracle_solution_sets_match": all(
            result.classical_oracle_solution_sets_match for result in results
        ),
        "all_completed_z3_sets_match": (
            all(
                result.branch_and_bound_z3_solution_sets_match is True
                for result in results
            )
            if args.z3
            else None
        ),
        "uniquely_identifiable_fraction": sum(
            result.uniquely_identifiable_on_domain for result in results
        )
        / len(results),
        "mean_fibre_size": statistics.mean(
            result.solution_count for result in results
        ),
        "mean_branch_and_bound_leaf_fraction": statistics.mean(
            result.branch_and_bound.leaf_fraction for result in results
        ),
        "median_branch_and_bound_seconds": statistics.median(
            result.branch_and_bound_seconds for result in results
        ),
        "median_oracle_build_seconds": statistics.median(
            result.oracle_build_seconds for result in results
        ),
        "claim_boundary": (
            "CI timings are reproducibility diagnostics, not hardware-comparable "
            "cost evidence. Publication experiments require pinned runners, warmup, "
            "repetitions, confidence intervals, and a common cost model."
        ),
    }
    print(json.dumps({"summary": summary, "results": payloads}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
