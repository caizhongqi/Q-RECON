import math

import pytest
import torch

from qrecon.theory.head_gradient_stability import (
    HeadPerturbationNormBounds,
    certify_head_representation_perturbation,
    combine_head_perturbation_bounds,
    common_scale_invariance_error,
    gaussian_head_bounds,
    recover_head_representation,
    uniform_quantization_head_bounds,
)


def _clean_pair():
    bias = torch.tensor([0.75, -0.5, 0.25], dtype=torch.float64)
    feature = torch.tensor([1.25, -0.5, 2.0, 0.75], dtype=torch.float64)
    weight = bias.unsqueeze(1) * feature.unsqueeze(0)
    return weight, bias, feature


def test_common_nonzero_scale_leaves_head_ratio_invariant():
    weight, bias, feature = _clean_pair()
    recovered = recover_head_representation(weight, bias)
    assert torch.allclose(recovered, feature, atol=1e-12, rtol=0)
    for scale in (0.01, 0.25, 3.5, -2.0):
        assert common_scale_invariance_error(weight, bias, scale) < 1e-12
    with pytest.raises(ValueError):
        common_scale_invariance_error(weight, bias, 0.0)


def test_deterministic_posterior_bound_contains_actual_error():
    weight, bias, feature = _clean_pair()
    bias_error = torch.tensor([0.01, -0.02, 0.015], dtype=torch.float64)
    weight_error = torch.tensor(
        [
            [0.01, -0.02, 0.01, 0.0],
            [0.0, 0.01, -0.01, 0.02],
            [-0.01, 0.0, 0.01, -0.01],
        ],
        dtype=torch.float64,
    )
    observed_bias = bias + bias_error
    observed_weight = weight + weight_error
    bounds = HeadPerturbationNormBounds(
        bias_l2=float(bias_error.norm()),
        weight_frobenius=float(weight_error.norm()),
        provenance="deterministic test perturbation",
    )
    certificate = certify_head_representation_perturbation(
        observed_weight, observed_bias, bounds
    )
    recovered = recover_head_representation(observed_weight, observed_bias)
    actual = float((recovered - feature).norm())
    assert certificate.certifiable
    assert certificate.posterior_l2_error_bound is not None
    assert actual <= certificate.posterior_l2_error_bound + 1e-12
    assert certificate.bias_error_ratio < 1.0


def test_large_bias_uncertainty_is_reported_as_noncertifiable():
    weight, bias, _ = _clean_pair()
    bounds = HeadPerturbationNormBounds(
        bias_l2=10.0,
        weight_frobenius=0.0,
        provenance="too-large uncertainty",
    )
    certificate = certify_head_representation_perturbation(weight, bias, bounds)
    assert not certificate.certifiable
    assert certificate.posterior_l2_error_bound is None


def test_quantization_and_gaussian_bounds_have_declared_geometry():
    quantization = uniform_quantization_head_bounds(3, 4, 0.02)
    assert quantization.bias_l2 == pytest.approx(0.01 * math.sqrt(3))
    assert quantization.weight_frobenius == pytest.approx(0.01 * math.sqrt(12))
    assert quantization.failure_probability is None

    gaussian = gaussian_head_bounds(3, 4, 1e-3, 0.05)
    assert gaussian.bias_l2 > 0.0
    assert gaussian.weight_frobenius > gaussian.bias_l2
    assert gaussian.failure_probability == pytest.approx(0.05)

    combined = combine_head_perturbation_bounds(quantization, gaussian)
    assert combined.bias_l2 == pytest.approx(
        quantization.bias_l2 + gaussian.bias_l2
    )
    assert combined.weight_frobenius == pytest.approx(
        quantization.weight_frobenius + gaussian.weight_frobenius
    )
    assert combined.failure_probability == pytest.approx(0.05)


def test_gaussian_bound_covers_a_seeded_draw_in_the_test_configuration():
    torch.manual_seed(123)
    output_dimension = 5
    feature_dimension = 7
    sigma = 0.01
    delta = 1e-4
    bias_noise = sigma * torch.randn(output_dimension)
    weight_noise = sigma * torch.randn(output_dimension, feature_dimension)
    bound = gaussian_head_bounds(
        output_dimension, feature_dimension, sigma, delta
    )
    assert float(bias_noise.norm()) <= bound.bias_l2
    assert float(weight_noise.norm()) <= bound.weight_frobenius
