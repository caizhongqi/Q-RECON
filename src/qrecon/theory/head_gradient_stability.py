from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class HeadPerturbationNormBounds:
    """Norm bounds for perturbations of final-head bias and weight gradients."""

    bias_l2: float
    weight_frobenius: float
    provenance: str
    failure_probability: float | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("bias_l2", self.bias_l2),
            ("weight_frobenius", self.weight_frobenius),
        ):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.failure_probability is not None and not (
            math.isfinite(float(self.failure_probability))
            and 0.0 < float(self.failure_probability) < 1.0
        ):
            raise ValueError("failure_probability must lie strictly between zero and one")
        if not self.provenance.strip():
            raise ValueError("provenance must be non-empty")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class HeadRepresentationPerturbationCertificate:
    """A posteriori deterministic error certificate for recovered head features.

    Let the clean one-effective-sample final-head gradients be
    ``G = u z^T`` and ``g = u``. After a common public scale ``alpha > 0`` and
    perturbations ``E`` and ``e``, the released pair is

    ``G_tilde = alpha G + E`` and ``g_tilde = alpha g + e``.

    The standard ratio estimator is
    ``z_hat = g_tilde^T G_tilde / ||g_tilde||^2``. If norm bounds on ``e`` and
    ``E`` are available, then

    ``||z_hat-z|| <= a ||z|| + b``, where
    ``a = ||e||/||g_tilde||`` and ``b = ||E||_F/||g_tilde||``.

    When ``a < 1``, the observable a posteriori bound is

    ``||z_hat-z|| <= (a ||z_hat|| + b)/(1-a)``.
    """

    observed_bias_l2: float
    recovered_feature_l2: float
    bias_error_ratio: float
    weight_error_normalized: float
    posterior_l2_error_bound: float | None
    perturbation_bounds: HeadPerturbationNormBounds

    @property
    def certifiable(self) -> bool:
        return self.posterior_l2_error_bound is not None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["perturbation_bounds"] = self.perturbation_bounds.to_dict()
        payload["certifiable"] = self.certifiable
        return payload


def recover_head_representation(
    weight_gradient: torch.Tensor,
    bias_gradient: torch.Tensor,
    *,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    """Recover a one-effective-sample final Linear input from its gradients."""

    weight = weight_gradient.detach()
    bias = bias_gradient.detach()
    if weight.ndim != 2 or bias.ndim != 1:
        raise ValueError("weight_gradient must be a matrix and bias_gradient a vector")
    if weight.shape[0] != bias.shape[0]:
        raise ValueError("weight and bias gradient output dimensions differ")
    if not math.isfinite(float(epsilon)) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    energy = bias.square().sum()
    if float(energy) <= epsilon:
        raise ZeroDivisionError("observed bias gradient is zero")
    return torch.matmul(bias.unsqueeze(0), weight).squeeze(0) / energy


def common_scale_invariance_error(
    weight_gradient: torch.Tensor,
    bias_gradient: torch.Tensor,
    scale: float,
    *,
    epsilon: float = 1e-12,
) -> float:
    """Numerically audit invariance under one common nonzero clipping scale."""

    alpha = float(scale)
    if not math.isfinite(alpha) or alpha == 0.0:
        raise ValueError("scale must be finite and nonzero")
    reference = recover_head_representation(
        weight_gradient, bias_gradient, epsilon=epsilon
    )
    scaled = recover_head_representation(
        alpha * weight_gradient,
        alpha * bias_gradient,
        epsilon=epsilon,
    )
    return float((scaled - reference).norm())


def certify_head_representation_perturbation(
    observed_weight_gradient: torch.Tensor,
    observed_bias_gradient: torch.Tensor,
    bounds: HeadPerturbationNormBounds,
    *,
    epsilon: float = 1e-12,
) -> HeadRepresentationPerturbationCertificate:
    """Return the deterministic observable perturbation certificate."""

    recovered = recover_head_representation(
        observed_weight_gradient,
        observed_bias_gradient,
        epsilon=epsilon,
    )
    observed_bias_norm = float(observed_bias_gradient.detach().norm())
    if observed_bias_norm <= epsilon:
        raise ZeroDivisionError("observed bias gradient is zero")
    a = float(bounds.bias_l2) / observed_bias_norm
    b = float(bounds.weight_frobenius) / observed_bias_norm
    posterior = None
    if a < 1.0:
        posterior = (a * float(recovered.norm()) + b) / (1.0 - a)
    return HeadRepresentationPerturbationCertificate(
        observed_bias_l2=observed_bias_norm,
        recovered_feature_l2=float(recovered.norm()),
        bias_error_ratio=a,
        weight_error_normalized=b,
        posterior_l2_error_bound=posterior,
        perturbation_bounds=bounds,
    )


def uniform_quantization_head_bounds(
    output_dimension: int,
    feature_dimension: int,
    step: float,
) -> HeadPerturbationNormBounds:
    """No-saturation error bounds for uniform round-to-nearest quantization."""

    outputs = int(output_dimension)
    features = int(feature_dimension)
    delta = float(step)
    if outputs <= 0 or features <= 0:
        raise ValueError("dimensions must be positive")
    if not math.isfinite(delta) or delta <= 0.0:
        raise ValueError("step must be finite and positive")
    return HeadPerturbationNormBounds(
        bias_l2=0.5 * delta * math.sqrt(outputs),
        weight_frobenius=0.5 * delta * math.sqrt(outputs * features),
        provenance="uniform round-to-nearest quantization without saturation",
        failure_probability=None,
    )


def gaussian_head_bounds(
    output_dimension: int,
    feature_dimension: int,
    noise_std: float,
    failure_probability: float,
) -> HeadPerturbationNormBounds:
    """High-probability Gaussian norm bounds for bias and weight perturbations.

    A union bound splits ``failure_probability`` equally between the bias vector
    and weight matrix. The standard Gaussian concentration inequality
    ``||N|| <= sigma(sqrt(n)+sqrt(2 log(2/delta)))`` is used for each flattened
    tensor.
    """

    outputs = int(output_dimension)
    features = int(feature_dimension)
    sigma = float(noise_std)
    delta = float(failure_probability)
    if outputs <= 0 or features <= 0:
        raise ValueError("dimensions must be positive")
    if not math.isfinite(sigma) or sigma < 0.0:
        raise ValueError("noise_std must be finite and non-negative")
    if not math.isfinite(delta) or not 0.0 < delta < 1.0:
        raise ValueError("failure_probability must lie strictly between zero and one")
    tail = math.sqrt(2.0 * math.log(2.0 / delta))
    return HeadPerturbationNormBounds(
        bias_l2=sigma * (math.sqrt(outputs) + tail),
        weight_frobenius=sigma * (math.sqrt(outputs * features) + tail),
        provenance="Gaussian concentration with a two-event union bound",
        failure_probability=delta,
    )


def combine_head_perturbation_bounds(
    *bounds: HeadPerturbationNormBounds,
    provenance: str | None = None,
) -> HeadPerturbationNormBounds:
    """Combine independent or deterministic component bounds by triangle inequality."""

    if not bounds:
        raise ValueError("at least one perturbation bound is required")
    failure = sum(
        float(bound.failure_probability)
        for bound in bounds
        if bound.failure_probability is not None
    )
    if failure >= 1.0:
        raise ValueError("combined failure probability must be below one")
    label = provenance or " + ".join(bound.provenance for bound in bounds)
    return HeadPerturbationNormBounds(
        bias_l2=sum(float(bound.bias_l2) for bound in bounds),
        weight_frobenius=sum(float(bound.weight_frobenius) for bound in bounds),
        provenance=label,
        failure_probability=(failure if failure > 0.0 else None),
    )
