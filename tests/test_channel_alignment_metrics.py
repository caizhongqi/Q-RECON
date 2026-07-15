import pytest
import torch

from qrecon.metrics import permutation_invariant_channel_metrics


def test_channel_alignment_recovers_exact_global_permutation():
    reference = torch.tensor(
        [
            [
                [0.0, 10.0, 20.0],
                [1.0, 11.0, 21.0],
                [2.0, 12.0, 22.0],
            ]
        ]
    )
    permutation = [2, 0, 1]
    estimate = reference[..., permutation]
    report = permutation_invariant_channel_metrics(
        reference, estimate, tolerance=0.0
    )
    assert report.assignment == (1, 2, 0)
    assert report.identity_assignment is False
    assert report.assignment_total_mse == pytest.approx(0.0)
    assert report.exact_tensor_within_tolerance is True
    assert report.channel_success_count == 3
    assert report.channel_success_rate == pytest.approx(1.0)
    assert report.aligned_metrics["mse"] == pytest.approx(0.0)


def test_channel_alignment_uses_one_assignment_across_batch_and_time():
    first = torch.tensor(
        [
            [[0.0, 5.0], [1.0, 6.0]],
            [[2.0, 7.0], [3.0, 8.0]],
        ]
    )
    estimate = first[..., [1, 0]].clone()
    estimate[1, 0, 0] += 0.2
    report = permutation_invariant_channel_metrics(
        first, estimate, tolerance=0.25
    )
    assert report.assignment == (1, 0)
    assert report.exact_tensor_within_tolerance is True
    assert report.channel_success_count == 2
    assert report.aligned_metrics["max_absolute_error"] == pytest.approx(0.2)


def test_channel_alignment_reports_partial_channel_success():
    reference = torch.zeros(1, 3, 3)
    estimate = reference.clone()
    estimate[..., 1] = 0.2
    report = permutation_invariant_channel_metrics(
        reference, estimate, tolerance=0.1
    )
    assert report.identity_assignment is True
    assert report.exact_tensor_within_tolerance is False
    assert report.channel_success_count == 2
    assert report.channel_success_rate == pytest.approx(2 / 3)


def test_channel_alignment_rejects_incompatible_inputs():
    with pytest.raises(ValueError, match="identical shapes"):
        permutation_invariant_channel_metrics(
            torch.zeros(1, 2, 3), torch.zeros(1, 2, 2)
        )
    with pytest.raises(ValueError, match="trailing channel"):
        permutation_invariant_channel_metrics(torch.zeros(3), torch.zeros(3))
    with pytest.raises(ValueError, match="size 12"):
        permutation_invariant_channel_metrics(
            torch.zeros(1, 2, 13), torch.zeros(1, 2, 13)
        )
