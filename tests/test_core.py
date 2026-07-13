import torch

from qrecon.attacks import (
    infer_class_label_from_last_bias,
    invert_first_linear_gradient,
    leak_gradients,
)
from qrecon.data import synthetic_forecasting
from qrecon.identifiability import gradient_jacobian_report
from qrecon.metrics import reconstruction_metrics
from qrecon.models import ForecastMLP, ImageMLP
from qrecon.quantum import DirectPrior, quantum_resource_estimate


def test_gradient_leak_and_identifiability_report():
    dataset = synthetic_forecasting(samples=4, context=4, horizon=1)
    x, target = (tensor[:1] for tensor in dataset.tensors)
    model = ForecastMLP(4, 1, hidden=5)
    gradients = leak_gradients(model, x, target, "forecasting")
    assert gradients
    report = gradient_jacobian_report(model, x, target, "forecasting")
    assert report.input_dimension == 4
    assert report.observation_dimension == sum(p.numel() for p in model.parameters())
    assert 0 <= report.numerical_rank <= 4


def test_prior_metrics_and_resources():
    prior = DirectPrior((1, 4), "timeseries")
    reconstruction = prior()
    metrics = reconstruction_metrics(torch.zeros_like(reconstruction), reconstruction)
    assert metrics["mse"] >= 0
    resources = quantum_resource_estimate(6, 2, 1000)
    assert resources["logical_qubits"] == 6
    assert resources["shots_per_forward"] == 1000


def test_analytic_forecasting_recovery_is_numerically_exact():
    torch.manual_seed(3)
    model = ForecastMLP(6, 2, hidden=7)
    x = torch.randn(1, 6)
    target = torch.randn(1, 2)
    gradients = leak_gradients(model, x, target, "forecasting")
    recovered = invert_first_linear_gradient(model, gradients, tuple(x.shape))
    assert torch.allclose(recovered, x, atol=1e-6, rtol=0)


def test_analytic_image_recovery_and_label_inference():
    torch.manual_seed(5)
    model = ImageMLP((3, 4, 4), classes=2, hidden=8)
    x = torch.rand(1, 3, 4, 4)
    target = torch.tensor([1])
    gradients = leak_gradients(model, x, target, "classification")
    recovered = invert_first_linear_gradient(model, gradients, tuple(x.shape))
    inferred = infer_class_label_from_last_bias(model, gradients)
    assert torch.allclose(recovered, x, atol=1e-6, rtol=0)
    assert inferred.item() == target.item()
