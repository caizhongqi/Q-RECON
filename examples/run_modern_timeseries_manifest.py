from __future__ import annotations

import argparse
import json
from pathlib import Path

from qrecon.benchmarks import (
    load_modern_timeseries_manifest,
    run_modern_timeseries_reconstruction_benchmark,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a canonical modern time-series reconstruction manifest"
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--require-quality-gate",
        action="store_true",
        help="exit non-zero unless the publication completeness gate passes",
    )
    args = parser.parse_args()

    manifest = load_modern_timeseries_manifest(str(args.manifest))
    report = run_modern_timeseries_reconstruction_benchmark(manifest)
    payload = report.to_dict()
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is None:
        print(encoded)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    if args.require_quality_gate and not report.quality_gate.passed:
        raise SystemExit("publication completeness gate did not pass")


if __name__ == "__main__":
    main()
