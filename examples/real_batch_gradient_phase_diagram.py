from __future__ import annotations

import argparse
import json
from pathlib import Path

from qrecon.benchmarks import (
    CandidateQuantizationSpec,
    RealBatchGradientManifest,
    run_real_batch_gradient_phase_diagram,
)


def _offline_manifest() -> RealBatchGradientManifest:
    return RealBatchGradientManifest(
        dataset="synthetic_forecasting",
        candidate_count=8,
        context=8,
        horizon=1,
        feature_count=4,
        feature_selection="uniform_stride",
        target_coordinate=0,
        target_batch_indices=(0, 1),
        quantizations=(
            CandidateQuantizationSpec(8, 4, True, "saturate"),
            CandidateQuantizationSpec(10, 6, True, "saturate"),
        ),
        model_weights=(1, -1, 2, -2),
        model_bias=0,
        gradient_bits=32,
        target_success=0.9,
        bbht_attempts_per_stage=2,
        max_exact_population=64,
        max_exact_batches=64,
        max_basis_verification_bits=6,
        reusable_instances=(1, 10, 100),
        access_contract="explicit_compilation",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the versioned real-candidate two-record gradient phase diagram."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="JSON manifest. Omit to run the deterministic offline smoke configuration.",
    )
    args = parser.parse_args()
    manifest = (
        _offline_manifest()
        if args.manifest is None
        else RealBatchGradientManifest.from_json(
            args.manifest.read_text(encoding="utf-8")
        )
    )
    report = run_real_batch_gradient_phase_diagram(manifest)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
