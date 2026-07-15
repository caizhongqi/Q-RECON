from __future__ import annotations

import json

from qrecon.benchmarks.tensor_candidates import audit_quantized_candidate_tensor
from qrecon.data import synthetic_forecasting


def main() -> None:
    dataset = synthetic_forecasting(samples=8, context=4, horizon=1, seed=23)
    inputs, _ = dataset.tensors
    audit = audit_quantized_candidate_tensor(
        inputs,
        bits_per_value=8,
        fractional_bits=4,
        signed=True,
        overflow="saturate",
        max_minterm_table_bits=4096,
    )
    payload = {
        "dataset": "synthetic_forecasting",
        "quantization": audit.to_dict(include_codes=False),
        "claim_boundary": (
            "Quantization is deterministic post-processing. Any collisions or saturation "
            "reported here lower the exact-index information ceiling before coherent search; "
            "the nested loading report then prices the resulting explicit fixed-point table."
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
