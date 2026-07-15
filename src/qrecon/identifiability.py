from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .attacks import capture_final_linear_input


@dataclass
class IdentifiabilityReport:
    input_dimension: int
    observation_dimension: int
    numerical_rank: int
    rank_ratio: float
    full_column_rank: bool
    largest_singular_value: float
    smallest_identifiable_singular_value: float
    condition_number: float

    def to_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)


def _matrix_report(
    matrix: torch.Tensor,
    input_dimension: int,
    tolerance: float,
) -> IdentifiabilityReport:
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    matrix = matrix.detach().double()
    singular = torch.linalg.svdvals(matrix)
    largest = float(singular.max()) if singular.numel() else 0.0
    threshold = max(tolerance, tolerance * largest)
    identifiable = singular[singular > threshold]
    rank = int(identifiable.numel())
    smallest = float(identifiable.min()) if rank else 0.0
    condition = largest / smallest if smallest > 0 else float("inf")
    return IdentifiabilityReport(
        input_dimension=int(input_dimension),
        observation_dimension=int(matrix.shape[0]),
        numerical_rank=rank,
        rank_ratio=rank / max(int(input_dimension), 1),
        full_column_rank=rank == int(input_dimension),
        largest_singular_value=largest,
        smallest_identifiable_singular_value=smallest,
        condition_number=condition,
    )


def gradient_jacobian_report(
    model: nn.Module,
    x: torch.Tensor,
    target: torch.Tensor,
    task: str,
    tolerance: float = 1e-6,
) -> IdentifiabilityReport:
    """Local differential identifiability from the full per-sample gradient.

    For a continuously differentiable observation map, full column rank is a
    sufficient local-injectivity certificate: a nonsingular square row minor and
    the inverse-function theorem provide a locally invertible output projection.
    It is not a global uniqueness certificate, and rank deficiency alone does
    not prove non-identifiability.

    This calculation is intentionally limited to small samples because the full
    Jacobian has ``observation_dimension × input_dimension`` entries.
    """

    parameters = tuple(model.parameters())

    def observation(candidate: torch.Tensor) -> torch.Tensor:
        prediction = model(candidate)
        loss = (
            F.cross_entropy(prediction, target.long())
            if task == "classification"
            else F.mse_loss(prediction, target)
        )
        gradients = torch.autograd.grad(loss, parameters, create_graph=True)
        return torch.cat([gradient.reshape(-1) for gradient in gradients])

    jacobian = torch.autograd.functional.jacobian(observation, x, vectorize=True)
    matrix = jacobian.reshape(jacobian.shape[0], -1)
    return _matrix_report(matrix, x.numel(), tolerance)


def head_representation_jacobian_report(
    model: nn.Module,
    x: torch.Tensor,
    *,
    tolerance: float = 1e-6,
    require_single_effective_sample: bool = True,
) -> IdentifiabilityReport:
    """Local input identifiability from the final Linear's input representation.

    For a single effective sample with nonzero final-head bias gradient, the
    representation is recovered exactly from the released head gradients. If the
    representation map ``E(x)`` has full-column-rank Jacobian, then ``E`` is locally
    injective around ``x``. Moreover the exact representation-matching objective
    has Hessian ``2 J_E(x)^T J_E(x)`` at a zero-residual solution, so the true input
    is a strict local minimizer. This remains a local certificate, not a global
    uniqueness theorem.
    """

    initial = capture_final_linear_input(model, x)
    effective_samples = int(initial.numel() // initial.shape[-1])
    if require_single_effective_sample and effective_samples != 1:
        raise ValueError(
            "analytic representation leakage requires exactly one effective final-head sample"
        )

    def observation(candidate: torch.Tensor) -> torch.Tensor:
        return capture_final_linear_input(model, candidate).reshape(-1)

    jacobian = torch.autograd.functional.jacobian(observation, x, vectorize=True)
    matrix = jacobian.reshape(jacobian.shape[0], -1)
    return _matrix_report(matrix, x.numel(), tolerance)
