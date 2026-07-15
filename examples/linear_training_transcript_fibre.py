from __future__ import annotations

import json

import numpy as np

from qrecon.theory import target_stabilizer_rotation
from qrecon.theory.linear_training_transcripts import (
    LinearOptimizerConfig,
    maximum_training_transcript_difference,
    simulate_linear_training,
)


rng = np.random.default_rng(71)
inputs = rng.normal(size=(8, 3))
targets = rng.normal(size=(8, 2))
rotated_inputs = target_stabilizer_rotation(targets, 0.27, axes=(0, 2)) @ inputs
weights = rng.normal(size=(2, 3))
bias = rng.normal(size=2)
config = LinearOptimizerConfig(
    optimizer="adam",
    learning_rate=0.025,
    weight_decay=0.01,
    decay_bias=True,
)

left = simulate_linear_training(
    inputs, targets, weights, bias, steps=10, config=config
)
right = simulate_linear_training(
    rotated_inputs, targets, weights, bias, steps=10, config=config
)

print(
    json.dumps(
        {
            "input_displacement": float(np.linalg.norm(inputs - rotated_inputs)),
            "maximum_transcript_difference": maximum_training_transcript_difference(
                left, right
            ),
            "steps": len(left.snapshots),
            "final_loss": left.snapshots[-1].loss,
        },
        indent=2,
    )
)
