from __future__ import annotations

import argparse
import json

from .config import load_config
from .experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Q-RECON experiment runner")
    parser.add_argument("--config", required=True, help="YAML experiment configuration")
    args = parser.parse_args()
    report = run_experiment(load_config(args.config))
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True))


if __name__ == "__main__":
    main()

