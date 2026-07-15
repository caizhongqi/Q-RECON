from __future__ import annotations

import pytest
import torch

from qrecon.models import ITransformer, PatchTST
from qrecon.theory.channel_permutation_training import (
    channel_permutation_training_transcript_witness,
)


@pytest.mark.parametrize("optimizer", ["sgd", "momentum", "adam", "adamw"])
def test_itransformer_training_transcript_is_channel_permutation_invariant(
    optimizer: str,
):
    torch.manual_seed(71)
    model = ITransformer(
        4,
        2,
        input_channels=3,
        d_model=6,
        n_heads=2,
        e_layers=1,
        d_ff=12,
        dropout=0.0,
        revin=False,
    )
    inputs = torch.randn(2, 4, 3)
    targets = torch.randn(2, 2, 3)
    witness = channel_permutation_training_transcript_witness(
        model,
        inputs,
        targets,
        (2, 0, 1),
        optimizer=optimizer,
        steps=3,
        learning_rate=1e-3,
        weight_decay=1e-4,
    )
    assert witness.fibre_bound.orbit_size == 6
    assert witness.maximum_loss_absolute_difference < 2e-6
    assert witness.maximum_gradient_absolute_difference < 3e-6
    assert witness.maximum_parameter_absolute_difference < 3e-6
    assert witness.maximum_optimizer_state_absolute_difference < 3e-6
    assert witness.final_model_delta_difference.relative_l2_difference < 3e-5


def test_shared_patchtst_adamw_model_delta_is_permutation_invariant():
    torch.manual_seed(73)
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
        individual_head=False,
    )
    inputs = torch.randn(2, 6, 3)
    targets = torch.randn(2, 2, 3)
    witness = channel_permutation_training_transcript_witness(
        model,
        inputs,
        targets,
        (1, 2, 0),
        optimizer="adamw",
        steps=3,
        learning_rate=1e-3,
        weight_decay=1e-4,
    )
    assert witness.maximum_gradient_absolute_difference < 3e-6
    assert witness.maximum_parameter_absolute_difference < 3e-6
    assert witness.final_model_delta_difference.relative_l2_difference < 3e-5


def test_channel_specific_patchtst_training_trajectory_breaks_symmetry():
    torch.manual_seed(79)
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
    inputs = torch.randn(2, 6, 3)
    targets = torch.randn(2, 2, 3)
    witness = channel_permutation_training_transcript_witness(
        model,
        inputs,
        targets,
        (1, 2, 0),
        optimizer="adamw",
        steps=2,
        learning_rate=1e-3,
    )
    assert witness.maximum_loss_absolute_difference > 1e-5
    assert witness.maximum_gradient_absolute_difference > 1e-4
    assert witness.final_model_delta_difference.relative_l2_difference > 1e-4
