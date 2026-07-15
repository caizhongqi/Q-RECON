from __future__ import annotations

import torch

from qrecon.models import ITransformer


def _gradient_tuple(model, inputs, targets, *, channel_weights=None):
    prediction = model(inputs)
    squared = (prediction - targets).square()
    if channel_weights is None:
        loss = squared.mean()
    else:
        weights = channel_weights.reshape(1, 1, -1).to(squared)
        loss = (squared * weights).mean()
    gradients = torch.autograd.grad(loss, tuple(model.parameters()))
    return tuple(gradient.detach() for gradient in gradients)


def _maximum_difference(left, right):
    return max(float((a - b).abs().max()) for a, b in zip(left, right))


def test_channel_symmetric_loss_is_necessary_for_the_gradient_fibre():
    torch.manual_seed(20260715)
    model = ITransformer(
        context=8,
        horizon=3,
        input_channels=3,
        d_model=12,
        n_heads=3,
        e_layers=2,
        d_ff=24,
        dropout=0.0,
        revin=False,
    ).double()
    inputs = torch.randn(2, 8, 3, dtype=torch.float64)
    targets = torch.randn(2, 3, 3, dtype=torch.float64)
    permutation = torch.tensor([2, 0, 1])
    permuted_inputs = inputs.index_select(-1, permutation)
    permuted_targets = targets.index_select(-1, permutation)

    symmetric_reference = _gradient_tuple(model, inputs, targets)
    symmetric_permuted = _gradient_tuple(model, permuted_inputs, permuted_targets)
    assert _maximum_difference(symmetric_reference, symmetric_permuted) < 1e-12

    # Fixed semantic channel weights attach identities to output slots. Permuting
    # private channels while leaving public slot weights fixed changes the loss and
    # splits the gradient fibre.
    semantic_weights = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64)
    asymmetric_reference = _gradient_tuple(
        model, inputs, targets, channel_weights=semantic_weights
    )
    asymmetric_permuted = _gradient_tuple(
        model, permuted_inputs, permuted_targets, channel_weights=semantic_weights
    )
    assert _maximum_difference(asymmetric_reference, asymmetric_permuted) > 1e-6

    # If the weights are themselves private channel-associated data and move with the
    # same permutation, symmetry is restored. This separates public slot semantics
    # from a simultaneous relabeling of the entire private object.
    associated_weights = semantic_weights.index_select(0, permutation)
    jointly_permuted = _gradient_tuple(
        model, permuted_inputs, permuted_targets, channel_weights=associated_weights
    )
    assert _maximum_difference(asymmetric_reference, jointly_permuted) < 1e-12
