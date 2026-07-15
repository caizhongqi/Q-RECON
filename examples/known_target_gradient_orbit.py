from __future__ import annotations

import json

import numpy as np

from qrecon.theory import construct_known_target_rotation_collision


rng = np.random.default_rng(29)
inputs = rng.normal(size=(7, 3))
targets = rng.normal(size=(7, 1))
weights = rng.normal(size=(1, 3))
bias = rng.normal(size=1)

report = construct_known_target_rotation_collision(
    inputs,
    targets,
    weights,
    bias,
    angle=0.31,
    axes=(0, 2),
)
print(json.dumps(report.to_dict(), indent=2))
