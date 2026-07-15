from __future__ import annotations

import json

from qrecon.theory import (
    certified_explicit_table_no_advantage_region,
    certify_explicit_table_no_advantage,
)


def main() -> None:
    entries = 1024
    word_bits = 32
    one_shot = certify_explicit_table_no_advantage(
        entries,
        word_bits,
        classical_variable_upper_bound=20_000,
    )
    amortized_region = certified_explicit_table_no_advantage_region(
        entries,
        word_bits,
        quantum_variable_lower_bound=1_000,
        classical_variable_upper_bound=5_000,
    )
    print(
        json.dumps(
            {
                "one_shot": one_shot.to_dict(),
                "certified_no_advantage_region": (
                    None if amortized_region is None else amortized_region.to_dict()
                ),
                "access_model": "explicit classical table compiled exactly",
                "claim_boundary": (
                    "absence of a certificate is not evidence of quantum advantage"
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
