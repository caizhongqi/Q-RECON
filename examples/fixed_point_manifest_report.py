from __future__ import annotations

import argparse
import json

from qrecon.benchmarks import (
    FixedPointMLPBenchmarkConfig,
    FixedPointMLPBenchmarkManifest,
    run_fixed_point_mlp_manifest,
    summarize_manifest_execution,
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
        help="run the optional exact SMT baseline in every repeated measurement",
    )
    parser.add_argument(
        "--publication",
        action="store_true",
        help="use the larger predeclared seed, warmup, and repetition profile",
    )
    args = parser.parse_args()

    if args.publication:
        seeds = tuple(range(20))
        repeats_per_seed = 7
        warmup_runs = 2
        bootstrap_samples = 5000
        minimum_seeds = 20
        protocol = "publication-oriented-reference"
    else:
        seeds = (3, 5)
        repeats_per_seed = 2
        warmup_runs = 1
        bootstrap_samples = 200
        minimum_seeds = 2
        protocol = "ci-smoke"

    manifest = FixedPointMLPBenchmarkManifest(
        configurations=_configs(),
        seeds=seeds,
        repeats_per_seed=repeats_per_seed,
        warmup_runs=warmup_runs,
        use_z3=args.z3,
        z3_timeout_ms=10_000 if args.z3 else None,
        target_success=0.75,
    )
    execution = run_fixed_point_mlp_manifest(manifest)
    summary = summarize_manifest_execution(
        execution,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=20260715,
        minimum_seeds_per_config=minimum_seeds,
        minimum_candidate_scales=3,
        require_complete_cells=args.publication,
    )
    payload = {
        "protocol": protocol,
        "manifest_sha256": manifest.sha256,
        "manifest": manifest.to_dict(),
        "execution": execution.to_dict(),
        "summary": summary.to_dict(),
        "smoke_warning": (
            None
            if args.publication
            else "CI smoke validates execution, recording, failure visibility, and hierarchical "
            "aggregation only. It permits incomplete cells so recorded solver failures remain "
            "visible; it is not publication-level timing evidence."
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
