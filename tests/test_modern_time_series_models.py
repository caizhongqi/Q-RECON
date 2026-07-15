import pytest
import torch

from qrecon.attacks.gradient_inversion import _regularizer
from qrecon.data import synthetic_multivariate_forecasting
from qrecon.experiment import _build_model
from qrecon.models import (
    ITransformer,
    PatchTST,
    RevIN,
    TransformerForecaster,
    build_forecasting_model,
    patchtst_geometry,
)


def _tiny_models():
    return (
        TransformerForecaster(
            4,
            2,
            d_model=4,
            n_heads=1,
            e_layers=1,
            d_ff=8,
            dropout=0.0,
        ),
        PatchTST(
            4,
            2,
            patch_len=2,
            stride=1,
            d_model=4,
            n_heads=1,
            e_layers=1,
            d_ff=8,
            dropout=0.0,
        ),
        ITransformer(
            4,
            2,
            d_model=4,
            n_heads=1,
            e_layers=1,
            d_ff=8,
            dropout=0.0,
        ),
    )


def test_modern_forecasters_support_gradient_inversion_second_order_autograd():
    for model in _tiny_models():
        x = torch.randn(1, 4, requires_grad=True)
        target = torch.randn(1, 2)
        prediction = model(x)
        assert prediction.shape == target.shape
        loss = (prediction - target).square().mean()
        gradients = torch.autograd.grad(
            loss, tuple(model.parameters()), create_graph=True
        )
        objective = sum(gradient.square().mean() for gradient in gradients)
        input_gradient = torch.autograd.grad(objective, x)[0]
        assert torch.isfinite(input_gradient).all()


@pytest.mark.parametrize("architecture", ["transformer", "patchtst", "itransformer"])
def test_forecasting_factory_builds_configured_architectures(architecture: str):
    config = {
        "architecture": architecture,
        "d_model": 8,
        "n_heads": 2,
        "e_layers": 1,
        "d_ff": 16,
        "dropout": 0.0,
        "patch_len": 4,
        "stride": 2,
    }
    model = build_forecasting_model(8, 2, 3, config)
    prediction = model(torch.randn(2, 8, 3))
    assert prediction.shape == (2, 2, 3)


def test_factory_installs_parameter_free_revin_when_declared():
    model = build_forecasting_model(
        8,
        2,
        3,
        {
            "architecture": "itransformer",
            "d_model": 8,
            "n_heads": 2,
            "e_layers": 1,
            "d_ff": 16,
            "dropout": 0.0,
            "revin": True,
            "revin_affine": False,
        },
    )
    assert isinstance(model.revin, RevIN)
    assert not model.revin.affine
    assert tuple(model.revin.parameters()) == ()
    x = torch.randn(2, 8, 3)
    permutation = torch.tensor([2, 0, 1])
    model.eval()
    assert torch.allclose(
        model(x[:, :, permutation]),
        model(x)[:, :, permutation],
        atol=1e-6,
        rtol=0,
    )


def test_patchtst_and_itransformer_are_channel_permutation_equivariant():
    constructors = (
        lambda: PatchTST(
            6,
            2,
            input_channels=3,
            patch_len=3,
            stride=2,
            d_model=6,
            n_heads=2,
            e_layers=1,
            d_ff=12,
            dropout=0.0,
        ),
        lambda: ITransformer(
            6,
            2,
            input_channels=3,
            d_model=6,
            n_heads=2,
            e_layers=1,
            d_ff=12,
            dropout=0.0,
        ),
    )
    x = torch.randn(2, 6, 3)
    permutation = torch.tensor([2, 0, 1])
    for constructor in constructors:
        model = constructor().eval()
        reference = model(x)
        permuted = model(x[:, :, permutation])
        assert torch.allclose(
            permuted, reference[:, :, permutation], atol=1e-6, rtol=0
        )


def test_multivariate_generator_and_experiment_builder_are_wired():
    dataset = synthetic_multivariate_forecasting(
        samples=3,
        context=8,
        horizon=2,
        channels=4,
        seed=19,
    )
    x, y = dataset.tensors
    assert x.shape == (3, 8, 4)
    assert y.shape == (3, 2, 4)
    model = _build_model(
        dataset,
        "forecasting",
        {
            "architecture": "itransformer",
            "d_model": 8,
            "n_heads": 2,
            "e_layers": 1,
            "d_ff": 16,
            "dropout": 0.0,
        },
    )
    assert isinstance(model, ITransformer)
    assert model(x).shape == y.shape


def test_multivariate_timeseries_regularizer_uses_the_time_axis():
    series = torch.tensor(
        [[[0.0, 0.0], [1.0, 10.0], [2.0, 20.0]]]
    )
    # Temporal differences are [1, 10] at both transitions, so the mean
    # squared difference is (1 + 100 + 1 + 100) / 4 = 50.5.
    assert _regularizer(series, "timeseries") == pytest.approx(50.5)


def test_univariate_mlp_is_rejected_for_multivariate_data():
    with pytest.raises(ValueError, match="univariate"):
        build_forecasting_model(8, 2, 4, {"architecture": "mlp"})


def test_patchtst_geometry_matches_end_padding_contract():
    geometry = patchtst_geometry(96, 16, 8, padding_patch=True)
    assert geometry.padding == 8
    assert geometry.patch_count == 12
