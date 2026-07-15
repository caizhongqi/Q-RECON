from __future__ import annotations

import math

import pytest
import torch

from qrecon.models import ITransformer, PatchTST
from qrecon.theory.channel_permutation import (
    apply_channel_permutation,
    channel_permutation_fibre_bound,
    channel_permutation_gradient_witness,
    channel_permutation_orbit_size,
    tensor_channel_permutation_fibre_bound,
    validate_channel_permutation,
)


def test_channel_permutation_orbit_counts_duplicate_private_channels():
    assert channel_permutation_orbit_size((1, 1, 1)) == 6
    assert channel_permutation_orbit_size((2, 1)) == 3
    bound = channel_permutation_fibre_bound(("a", "a", "b"))
    assert bound.channels == 3
    assert bound.multiplicities == (2, 1)
    assert bound.orbit_size == 3
    assert bound.uniform_exact_ordered_recovery_ceiling == pytest.approx(1.0 / 3.0)


def test_tensor_orbit_uses_both_history_and_target_signatures():
    inputs = torch.tensor(
        [[[1.0, 1.0, 4.0], [2.0, 2.0, 5.0], [3.0, 3.0, 6.0]]]
    )
    targets = torch.tensor([[[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]])
    # The first two histories match, but their targets differ, so all three private
    # channel records are distinct and the labeled orbit has size 3!.
    bound = tensor_channel_permutation_fibre_bound(inputs, targets)
    assert bound.multiplicities == (1, 1, 1)
    assert bound.orbit_size == 6


def test_itransformer_full_gradient_is_channel_permutation_invariant():
    torch.manual_seed(17)
    model = ITransformer(
        context=4,
        horizon=2,
        input_channels=3,
        d_model=6,
        n_heads=2,
        e_layers=1,
        d_ff=12,
        dropout=0.0,
        revin=False,
    )
    inputs = torch.randn(1, 4, 3)
    targets = torch.randn(1, 2, 3)
    witness = channel_permutation_gradient_witness(
        model, inputs, targets, (2, 0, 1)
    )
    assert witness.nontrivial_private_collision
    assert witness.fibre_bound.orbit_size == math.factorial(3)
    assert witness.fibre_bound.uniform_exact_ordered_recovery_ceiling == pytest.approx(
        1.0 / 6.0
    )
    assert witness.prediction_equivariance_max_abs_error < 1e-6
    assert witness.loss_absolute_difference < 1e-6
    assert witness.gradient_max_abs_difference < 2e-6
    assert witness.gradient_relative_l2_difference < 2e-6


def test_channel_independent_patchtst_full_gradient_is_permutation_invariant():
    torch.manual_seed(23)
    model = PatchTST(
        context=6,
        horizon=2,
        input_channels=3,
        patch_len=3,
        stride=2,
        padding_patch=True,
        d_model=6,
        n_heads=2,
        e_layers=1,
        d_ff=12,
        dropout=0.0,
        head_dropout=0.0,
        revin=False,
        individual_head=False,
    )
    inputs = torch.randn(1, 6, 3)
    targets = torch.randn(1, 2, 3)
    witness = channel_permutation_gradient_witness(
        model, inputs, targets, (1, 2, 0)
    )
    assert witness.nontrivial_private_collision
    assert witness.fibre_bound.orbit_size == 6
    assert witness.prediction_equivariance_max_abs_error < 1e-6
    assert witness.loss_absolute_difference < 1e-6
    assert witness.gradient_max_abs_difference < 2e-6
    assert witness.gradient_relative_l2_difference < 2e-6


def test_channel_permutation_validation_and_application():
    tensor = torch.tensor([[[1.0, 2.0, 3.0]]])
    assert validate_channel_permutation((2, 0, 1), 3) == (2, 0, 1)
    permuted = apply_channel_permutation(tensor, (2, 0, 1))
    assert torch.equal(permuted, torch.tensor([[[3.0, 1.0, 2.0]]]))
    with pytest.raises(ValueError, match="every channel"):
        validate_channel_permutation((0, 0, 1), 3)
