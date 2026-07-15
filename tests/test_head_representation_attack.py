from __future__ import annotations

import pytest
import torch

from qrecon.attacks import (
    HeadRepresentationInversionAttack,
    capture_final_linear_input,
    leak_gradients,
    recover_single_effective_head_input,
)
from qrecon.models import ITransformer, PatchTST, TransformerForecaster
from qrecon.quantum import DirectPrior


def _tiny_models():
    return (
        TransformerForecaster(
            6,
            2,
            d_model=4,
            n_heads=1,
            e_layers=1,
            d_ff=8,
            dropout=0.0,
            revin=True,
        ),
        PatchTST(
            6,
            2,
            patch_len=3,
            stride=2,
            d_model=4,
            n_heads=1,
            e_layers=1,
            d_ff=8,
            dropout=0.0,
            revin=True,
        ),
        ITransformer(
            6,
            2,
            d_model=4,
            n_heads=1,
            e_layers=1,
            d_ff=8,
            dropout=0.0,
            revin=True,
        ),
    )


@pytest.mark.parametrize("model", _tiny_models())
def test_final_head_gradient_exactly_recovers_one_effective_representation(model):
    torch.manual_seed(17)
    model.eval()
    x = torch.randn(1, 6)
    target = torch.randn(1, 2)
    observed = leak_gradients(model, x, target, "forecasting")
    captured = capture_final_linear_input(model, x).reshape(1, -1)[0]
    leakage = recover_single_effective_head_input(
        model,
        observed,
        effective_samples=1,
    )
    assert leakage.recovered_feature.shape == captured.shape
    assert torch.allclose(leakage.recovered_feature, captured, atol=1e-5, rtol=1e-5)
    assert leakage.rank_one_relative_residual < 1e-5


def test_shared_multivariate_head_rejects_multiple_effective_samples():
    model = PatchTST(
        6,
        2,
        input_channels=3,
        patch_len=3,
        stride=2,
        d_model=4,
        n_heads=1,
        e_layers=1,
        d_ff=8,
        dropout=0.0,
    ).eval()
    x = torch.randn(1, 6, 3)
    target = torch.randn(1, 2, 3)
    observed = leak_gradients(model, x, target, "forecasting")
    with pytest.raises(ValueError, match="one effective sample"):
        recover_single_effective_head_input(
            model,
            observed,
            effective_samples=3,
        )


def test_head_representation_inversion_reduces_released_feature_loss():
    torch.manual_seed(23)
    model = PatchTST(
        6,
        2,
        patch_len=3,
        stride=2,
        d_model=4,
        n_heads=1,
        e_layers=1,
        d_ff=8,
        dropout=0.0,
        revin=True,
    ).eval()
    reference = torch.tensor([[0.1, 0.4, -0.2, 0.8, 0.3, -0.5]])
    target = torch.tensor([[0.2, -0.1]])
    observed = leak_gradients(model, reference, target, "forecasting")
    attack = HeadRepresentationInversionAttack(
        model,
        observed,
        DirectPrior(tuple(reference.shape), "timeseries", bounded=False),
        steps=10,
        learning_rate=0.05,
        regularization=0.0,
        record_every=1,
    )
    result = attack.run()
    assert result.reconstruction.shape == reference.shape
    assert result.best_representation_loss <= result.history[0]["representation_loss"]
    assert result.best_objective <= result.final_objective + 1e-12
    assert result.leakage.rank_one_relative_residual < 1e-5
