from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class IdentifiabilityReport:
    input_dimension: int
    observation_dimension: int
    numerical_rank: int
    rank_ratio: float
    largest_singular_value: float
    smallest_identifiable_singular_value: float
    condition_number: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def gradient_jacobian_report(
    model: nn.Module,
    x: torch.Tensor,
    target: torch.Tensor,
    task: str,
    tolerance: float = 1e-6,
) -> IdentifiabilityReport:
    """Local identifiability of x from the full per-sample gradient.

    This is intentionally limited to small samples because the full Jacobian has
    observation_dimension × input_dimension entries.
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
    matrix = jacobian.reshape(jacobian.shape[0], -1).detach().double()
    singular = torch.linalg.svdvals(matrix)
    largest = float(singular.max()) if singular.numel() else 0.0
    threshold = max(tolerance, tolerance * largest)
    identifiable = singular[singular > threshold]
    rank = int(identifiable.numel())
    smallest = float(identifiable.min()) if rank else 0.0
    condition = largest / smallest if smallest > 0 else float("inf")
    input_dimension = x.numel()
    return IdentifiabilityReport(
        input_dimension=input_dimension,
        observation_dimension=matrix.shape[0],
        numerical_rank=rank,
        rank_ratio=rank / max(input_dimension, 1),
        largest_singular_value=largest,
        smallest_identifiable_singular_value=smallest,
        condition_number=condition,
    )

