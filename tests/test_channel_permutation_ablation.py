from __future__ import annotations

import torch

from qrecon.models import ITransformer, PatchTST, RevIN
from qrecon.theory import channel_permutation_gradient_witness


def test_non_affine_revin_preserves_itransformer_gradient_collision():
    torch.manual_seed(41)
    model = ITransformer(
        6,
        2,
        input_channels=3,
        d_model=6,
        n_heads=2,
        e_layers=1,
        d_ff=12,
        dropout=0.0,
        revin=False,
    )
    model.revin = RevIN(3, affine=False)
    inputs = torch.randn(1, 6, 3)
    targets = torch.randn(1, 2, 3)
    witness = channel_permutation_gradient_witness(
        model, inputs, targets, (2, 0, 1)
    )
    assert witness.nontrivial_private_collision
    assert witness.prediction_equivariance_max_abs_error < 1e-6
    assert witness.gradient_relative_l2_difference < 2e-6


def test_channel_specific_patchtst_heads_break_gradient_collision():
    torch.manual_seed(43)
    model = PatchTST(
        6,
        2,
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
        individual_head=True,
    )
    inputs = torch.randn(1, 6, 3)
    targets = torch.randn(1, 2, 3)
    witness = channel_permutation_gradient_witness(
        model, inputs, targets, (1, 2, 0)
    )
    assert witness.input_displacement_l2 > 0.0
    assert witness.prediction_equivariance_max_abs_error > 1e-4
    assert witness.loss_absolute_difference > 1e-5
    assert witness.gradient_relative_l2_difference > 1e-4
