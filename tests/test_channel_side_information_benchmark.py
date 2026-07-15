from __future__ import annotations

import math

import pytest
import torch

from qrecon.benchmarks import analyze_public_calibration_side_information


def _labeled_windows(samples: int = 12) -> tuple[torch.Tensor, torch.Tensor]:
    context = 32
    horizon = 4
    total = context + horizon
    time = torch.linspace(0.0, 4.0 * math.pi, total)
    inputs: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for sample in range(samples):
        phase = 0.03 * sample
        channel0 = torch.sin(time + phase)
        channel1 = 0.7 * torch.sin(2.0 * time + 0.2 + phase) + 0.04 * time
        channel2 = 0.5 * torch.cos(0.5 * time - 0.3 + phase) - 0.03 * time
        matrix = torch.stack((channel0, channel1, channel2), dim=-1)
        inputs.append(matrix[:context])
        targets.append(matrix[context:])
    return torch.stack(inputs), torch.stack(targets)


def test_public_calibration_matches_distinct_channel_dynamics():
    inputs, targets = _labeled_windows()
    points, names = analyze_public_calibration_side_information(
        inputs,
        targets,
        calibration_indices=tuple(range(6)),
        evaluation_indices=(6, 7, 8, 9),
        permutation_seeds=(11, 13, 17),
        spectral_bins=4,
    )
    assert len(points) == 12
    assert len(names) >= 15
    assert all(point.exact_labeled_order_recovered for point in points)
    assert all(point.channel_accuracy == 1.0 for point in points)
    assert all(point.assignment_margin > 0.0 for point in points)
    assert all(point.orbit_size == math.factorial(3) for point in points)
    assert all(
        point.no_side_information_exact_ceiling == pytest.approx(1.0 / 6.0)
        for point in points
    )


def test_public_calibration_split_must_be_disjoint():
    inputs, targets = _labeled_windows()
    with pytest.raises(ValueError, match="disjoint"):
        analyze_public_calibration_side_information(
            inputs,
            targets,
            calibration_indices=(0, 1, 2),
            evaluation_indices=(2, 3),
            permutation_seeds=(5,),
        )
