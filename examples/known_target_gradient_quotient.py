from __future__ import annotations

import json

import numpy as np

from qrecon.theory import (
    construct_known_target_orbit_representative,
    evaluate_linear_gradient_oracle,
    known_target_orbit_invariants_from_statistics,
    recover_linear_gradient_oracle_statistics,
)


rng = np.random.default_rng(41)
inputs = rng.normal(size=(8, 3))
targets = rng.normal(size=(8, 2))


def oracle(weights: np.ndarray, bias: np.ndarray):
    return evaluate_linear_gradient_oracle(inputs, targets, weights, bias)


recovery = recover_linear_gradient_oracle_statistics(targets, 3, oracle)
invariants = known_target_orbit_invariants_from_statistics(
    recovery.statistics, targets
)
representative = construct_known_target_orbit_representative(
    recovery.statistics, targets
)

print(
    json.dumps(
        {
            "probe_recovery": recovery.to_dict(),
            "orbit_invariants": invariants.to_dict(),
            "representative": representative.tolist(),
        },
        indent=2,
    )
)
