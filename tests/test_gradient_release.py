from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from qrecon.attacks import (
    GradientReleaseSpec,
    ReleasedGradientInversionAttack,
    clip_gradient_tuple,
    gradient_tuple_l2_norm,
    last_biased_linear_parameter_indices,
    leak_gradients,
    quantize_gradient_tuple,
    release_gradients,
)
from qrecon.quantum import DirectPrior


def _model() -> nn.Module:
    torch.manual_seed(13)
    return nn.Sequential(
        nn.Linear(3, 4),
        nn.Tanh(),
        nn.Linear(4, 1),
    ).eval()


def _observation(model: nn.Module):
    x = torch.tensor([[0.2, -0.4, 0.7]])
    target = torch.tensor([[0.3]])
    return x, target, leak_gradients(model, x, target, "forecasting")


def test_global_gradient_clipping_respects_declared_norm():
    gradients = (torch.tensor([3.0, 4.0]), torch.tensor([0.0]))
    clipped, factor = clip_gradient_tuple(gradients, 2.0)
    assert factor == pytest.approx(0.4)
    assert float(gradient_tuple_l2_norm(clipped)) == pytest.approx(2.0, rel=1e-6)
    unchanged, unchanged_factor = clip_gradient_tuple(gradients, 10.0)
    assert unchanged_factor == pytest.approx(1.0)
    assert torch.equal(unchanged[0], gradients[0])


def test_gradient_quantization_has_public_scale_and_straight_through_gradient():
    source = torch.tensor([-2.0, -0.25, 0.25, 2.0], requires_grad=True)
    quantized, scale, saturated, count = quantize_gradient_tuple(
        (source,),
        3,
        scale=0.5,
        straight_through=True,
    )
    assert scale == pytest.approx(0.5)
    assert saturated == 2
    assert count == 4
    assert torch.allclose(quantized[0].detach(), torch.tensor([-1.5, -0.0, 0.0, 1.5]))
    quantized[0].sum().backward()
    assert torch.equal(source.grad, torch.ones_like(source))


def test_release_is_deterministic_for_a_fixed_noise_seed_and_preserves_contract():
    model = _model()
    _, _, exact = _observation(model)
    spec = GradientReleaseSpec(
        clip_norm=0.5,
        noise_std=0.01,
        noise_seed=41,
        quantization_bits=8,
    )
    first = release_gradients(model, exact, spec)
    second = release_gradients(model, exact, spec)
    third = release_gradients(
        model,
        exact,
        GradientReleaseSpec(
            clip_norm=0.5,
            noise_std=0.01,
            noise_seed=43,
            quantization_bits=8,
        ),
    )
    assert all(torch.equal(left, right) for left, right in zip(first.gradients, second.gradients))
    assert any(not torch.equal(left, right) for left, right in zip(first.gradients, third.gradients))
    assert first.clip_norm == pytest.approx(0.5)
    assert first.clipped_l2_norm <= 0.5 + 1e-6
    assert first.quantization_scale is not None
    assert first.deterministic_attack_contract()["clip_norm"] == pytest.approx(0.5)


def test_partial_release_can_select_only_the_final_biased_linear():
    model = _model()
    _, _, exact = _observation(model)
    weight_index, bias_index = last_biased_linear_parameter_indices(model)
    release = release_gradients(
        model,
        exact,
        GradientReleaseSpec(
            visible_parameter_indices=(weight_index, bias_index),
        ),
    )
    assert release.visible_parameter_indices == (weight_index, bias_index)
    assert release.visible_parameter_names[-2:] == ("2.weight", "2.bias")
    assert len(release.gradients) == 2


def test_release_aware_inversion_runs_with_clip_quantization_and_partial_visibility():
    model = _model()
    reference, target, exact = _observation(model)
    weight_index, bias_index = last_biased_linear_parameter_indices(model)
    spec = GradientReleaseSpec(
        clip_norm=0.5,
        quantization_bits=8,
        visible_parameter_indices=(weight_index, bias_index),
    )
    release = release_gradients(model, exact, spec)
    attack = ReleasedGradientInversionAttack(
        model,
        release,
        spec,
        DirectPrior(tuple(reference.shape), "timeseries", bounded=False),
        task="forecasting",
        mode="timeseries",
        known_target=target,
        target_shape=tuple(target.shape),
        steps=3,
        learning_rate=0.03,
        regularization=0.0,
        match_mode="hybrid",
        layer_weighting="parameter",
        gradient_clip_norm=10.0,
        record_every=1,
    )
    result = attack.run()
    assert result.reconstruction.shape == reference.shape
    assert math.isfinite(result.best_objective)
    assert math.isfinite(result.final_objective)
    assert result.best_objective <= result.final_objective + 1e-12
    assert attack.transform_report.visible_parameter_indices == (
        weight_index,
        bias_index,
    )
    assert attack.transform_report.quantization_straight_through


def test_invalid_release_declarations_are_rejected():
    with pytest.raises(ValueError, match="quantization_bits"):
        GradientReleaseSpec(quantization_bits=1)
    with pytest.raises(ValueError, match="requires quantization_bits"):
        GradientReleaseSpec(quantization_scale=0.1)
    with pytest.raises(ValueError, match="must be unique"):
        GradientReleaseSpec(visible_parameter_indices=(1, 1))
