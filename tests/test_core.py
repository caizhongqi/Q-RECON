import torch

from qrecon.attacks import leak_gradients
from qrecon.data import synthetic_forecasting
from qrecon.identifiability import gradient_jacobian_report
from qrecon.metrics import reconstruction_metrics
from qrecon.models import ForecastMLP
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

