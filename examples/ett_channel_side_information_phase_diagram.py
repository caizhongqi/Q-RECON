from __future__ import annotations

import hashlib
import json

from qrecon.theory.channel_side_information import (
    channel_side_information_phase_diagram,
)


CHANNELS = ("HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT")


def main() -> None:
    # The validated ETTm1/ETTh1 publication artifacts certify distinct complete
    # private signatures on the generic seven-channel windows. Symbolic unique
    # signatures keep this side-information calculation independent of raw data.
    signatures = tuple(f"distinct-private-signature:{name}" for name in CHANNELS)
    regimes = {
        "ordered_channel_labels_private": ("private",) * len(CHANNELS),
        "channel_family_public": (
            "high-load",
            "high-load",
            "medium-load",
            "medium-load",
            "low-load",
            "low-load",
            "oil-temperature",
        ),
        "full_semantic_labels_public": CHANNELS,
    }
    diagram = channel_side_information_phase_diagram(signatures, regimes)
    payload = {
        "schema_version": "qrecon.channel-side-information.v1",
        "channels": list(CHANNELS),
        "regimes": {
            name: bound.to_dict() for name, bound in diagram.items()
        },
        "recovery_targets": {
            "exact_labeled_order": (
                "Use each regime's uniform_exact_ordered_recovery_ceiling."
            ),
            "modulo_residual_permutation": {
                "information_theoretic_ceiling": 1.0,
                "warning": (
                    "Collapsing an orbit to one target class removes this ambiguity "
                    "but does not prove that a representative is computationally recoverable."
                ),
            },
        },
        "validated_empirical_basis": {
            "datasets": ["ETTm1", "ETTm2", "ETTh1"],
            "architectures": ["iTransformer", "shared-head PatchTST"],
            "primary_windows": 120,
            "generic_distinct_channel_orbit": 5040,
        },
        "claim_boundary": (
            "Public side information is modeled as a restriction on admissible channel "
            "permutations. The calculation isolates permutation-induced uncertainty; "
            "other observation collisions or computational barriers can remain."
        ),
    }
    basis = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["report_sha256"] = hashlib.sha256(basis.encode("utf-8")).hexdigest()
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
