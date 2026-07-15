from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from qrecon.quantum import DirectPrior


LEARNED_QUANTILE_REFERENCE = {
    "repository": "Capsar/ts-inverse",
    "commit": "2015946906a693f836e6418cdeb3b64a3f6f2d6e",
    "reference_class": "ImprovedGradToInputNN_Quantile",
    "reproduced_components": (
        "separate residual input and target branches",
        "gradient-vector input",
        "multi-quantile input and target outputs",
        "pinball quantile training objective",
        "median initialization and lower/upper optimization bounds",
    ),
    "declared_differences": (
        "Q-RECON uses an explicit disjoint auxiliary split",
        "gradient features are standardized from the auxiliary training split",
        "network widths are manifest-controlled for the victim gradient size",
        "quantile crossing is penalized and sorted at inference",
    ),
}


def flatten_gradient_tuple(gradients: Sequence[torch.Tensor]) -> torch.Tensor:
    if not gradients:
        raise ValueError("gradients must be non-empty")
    return torch.cat(tuple(gradient.detach().reshape(-1) for gradient in gradients))


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape)).encode("ascii"))
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class GradientFeatureNormalizer:
    mean: torch.Tensor
    scale: torch.Tensor

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mean.to(values.device, values.dtype)) / self.scale.to(
            values.device, values.dtype
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "feature_count": int(self.mean.numel()),
            "mean_sha256": _tensor_sha256(self.mean),
            "scale_sha256": _tensor_sha256(self.scale),
            "minimum_scale": float(self.scale.min()),
            "maximum_scale": float(self.scale.max()),
        }


def fit_gradient_feature_normalizer(values: torch.Tensor) -> GradientFeatureNormalizer:
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("gradient feature matrix must be [samples,features] with at least two samples")
    mean = values.mean(dim=0)
    scale = values.std(dim=0, unbiased=False).clamp_min(1e-8)
    return GradientFeatureNormalizer(mean.detach().cpu(), scale.detach().cpu())


class QuantileResidualBlock(nn.Module):
    """Residual MLP block matching the public TS-Inverse inversion model family."""

    def __init__(self, input_features: int, output_features: int, dropout: float) -> None:
        super().__init__()
        if input_features <= 0 or output_features <= 0:
            raise ValueError("residual-block widths must be positive")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must lie in [0,1)")
        self.fc1 = nn.Linear(input_features, output_features)
        self.bn1 = nn.BatchNorm1d(output_features)
        self.fc2 = nn.Linear(output_features, output_features)
        self.bn2 = nn.BatchNorm1d(output_features)
        self.dropout = nn.Dropout(float(dropout))
        self.adapt: nn.Module
        if input_features == output_features:
            self.adapt = nn.Identity()
        else:
            self.adapt = nn.Sequential(
                nn.Linear(input_features, output_features),
                nn.BatchNorm1d(output_features),
            )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        identity = self.adapt(values)
        hidden = self.dropout(torch.relu(self.bn1(self.fc1(values))))
        hidden = self.bn2(self.fc2(hidden))
        return torch.relu(hidden + identity)


def _residual_stack(
    gradient_features: int,
    hidden_sizes: tuple[int, ...],
    dropout: float,
) -> nn.Sequential:
    if not hidden_sizes or any(width <= 0 for width in hidden_sizes):
        raise ValueError("hidden_sizes must contain positive widths")
    layers: list[nn.Module] = []
    previous = int(gradient_features)
    for width in hidden_sizes:
        layers.append(QuantileResidualBlock(previous, int(width), dropout))
        previous = int(width)
    return nn.Sequential(*layers)


class GradientToQuantileNetwork(nn.Module):
    """Gradient-to-input/target quantile network based on TS-Inverse."""

    def __init__(
        self,
        gradient_features: int,
        input_shape: tuple[int, ...],
        target_shape: tuple[int, ...] | None,
        *,
        hidden_sizes: tuple[int, ...] = (256, 128),
        dropout: float = 0.05,
        quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    ) -> None:
        super().__init__()
        if gradient_features <= 0:
            raise ValueError("gradient_features must be positive")
        if not input_shape or any(dimension <= 0 for dimension in input_shape):
            raise ValueError("input_shape must contain positive dimensions")
        if target_shape is not None and (
            not target_shape or any(dimension <= 0 for dimension in target_shape)
        ):
            raise ValueError("target_shape must contain positive dimensions")
        declared_quantiles = tuple(float(value) for value in quantiles)
        if not declared_quantiles or any(
            not 0.0 < value < 1.0 for value in declared_quantiles
        ):
            raise ValueError("quantiles must lie strictly between zero and one")
        if tuple(sorted(declared_quantiles)) != declared_quantiles:
            raise ValueError("quantiles must be sorted")
        if len(set(declared_quantiles)) != len(declared_quantiles):
            raise ValueError("quantiles must be unique")

        self.gradient_features = int(gradient_features)
        self.input_shape = tuple(int(value) for value in input_shape)
        self.target_shape = (
            None if target_shape is None else tuple(int(value) for value in target_shape)
        )
        self.hidden_sizes = tuple(int(value) for value in hidden_sizes)
        self.quantiles = declared_quantiles
        self.input_blocks = _residual_stack(
            self.gradient_features, self.hidden_sizes, float(dropout)
        )
        self.input_head = nn.Linear(
            self.hidden_sizes[-1], math.prod(self.input_shape) * len(self.quantiles)
        )
        if self.target_shape is None:
            self.target_blocks = None
            self.target_head = None
        else:
            self.target_blocks = _residual_stack(
                self.gradient_features, self.hidden_sizes, float(dropout)
            )
            self.target_head = nn.Linear(
                self.hidden_sizes[-1],
                math.prod(self.target_shape) * len(self.quantiles),
            )

    def forward(
        self, gradient_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if gradient_features.ndim != 2 or gradient_features.shape[1] != self.gradient_features:
            raise ValueError(
                f"expected gradient features [batch,{self.gradient_features}]"
            )
        batch = gradient_features.shape[0]
        inputs = self.input_head(self.input_blocks(gradient_features)).reshape(
            batch, *self.input_shape, len(self.quantiles)
        )
        if self.target_blocks is None or self.target_head is None:
            return inputs, None
        targets = self.target_head(self.target_blocks(gradient_features)).reshape(
            batch, *self.target_shape, len(self.quantiles)
        )
        return inputs, targets

    def inference(
        self, gradient_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        self.eval()
        with torch.no_grad():
            inputs, targets = self(gradient_features)
            inputs = torch.sort(inputs, dim=-1).values
            targets = None if targets is None else torch.sort(targets, dim=-1).values
            return inputs, targets


def pinball_quantile_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    quantiles: Sequence[float],
) -> torch.Tensor:
    declared = torch.tensor(
        tuple(float(value) for value in quantiles),
        device=predictions.device,
        dtype=predictions.dtype,
    )
    if predictions.shape[:-1] != targets.shape or predictions.shape[-1] != len(declared):
        raise ValueError("predictions must equal target shape plus one quantile axis")
    error = targets.unsqueeze(-1) - predictions
    return torch.maximum(declared * error, (declared - 1.0) * error).mean()


def quantile_crossing_penalty(predictions: torch.Tensor) -> torch.Tensor:
    if predictions.shape[-1] <= 1:
        return torch.zeros((), device=predictions.device, dtype=predictions.dtype)
    return torch.relu(predictions[..., :-1] - predictions[..., 1:]).mean()


@dataclass(frozen=True)
class QuantileInitializerTrainingReport:
    training_samples: int
    validation_samples: int
    gradient_features: int
    input_shape: tuple[int, ...]
    target_shape: tuple[int, ...] | None
    hidden_sizes: tuple[int, ...]
    quantiles: tuple[float, ...]
    epochs: int
    best_epoch: int
    best_validation_loss: float
    final_training_loss: float
    final_validation_loss: float
    normalizer: dict[str, object]
    model_sha256: str
    provenance: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "training_samples": self.training_samples,
            "validation_samples": self.validation_samples,
            "gradient_features": self.gradient_features,
            "input_shape": list(self.input_shape),
            "target_shape": None if self.target_shape is None else list(self.target_shape),
            "hidden_sizes": list(self.hidden_sizes),
            "quantiles": list(self.quantiles),
            "epochs": self.epochs,
            "best_epoch": self.best_epoch,
            "best_validation_loss": self.best_validation_loss,
            "final_training_loss": self.final_training_loss,
            "final_validation_loss": self.final_validation_loss,
            "normalizer": self.normalizer,
            "model_sha256": self.model_sha256,
            "provenance": self.provenance,
        }


def _model_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def train_gradient_quantile_network(
    gradient_features: torch.Tensor,
    inputs: torch.Tensor,
    targets: torch.Tensor | None,
    *,
    hidden_sizes: tuple[int, ...] = (256, 128),
    dropout: float = 0.05,
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    validation_fraction: float = 0.2,
    epochs: int = 100,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    crossing_weight: float = 1e-2,
    seed: int = 1729,
) -> tuple[
    GradientToQuantileNetwork,
    GradientFeatureNormalizer,
    QuantileInitializerTrainingReport,
]:
    if gradient_features.ndim != 2:
        raise ValueError("gradient_features must have shape [samples,features]")
    if inputs.shape[0] != gradient_features.shape[0]:
        raise ValueError("input and gradient sample counts must match")
    if targets is not None and targets.shape[0] != gradient_features.shape[0]:
        raise ValueError("target and gradient sample counts must match")
    sample_count = int(gradient_features.shape[0])
    if sample_count < 8:
        raise ValueError("at least eight auxiliary samples are required")
    if not 0.0 < float(validation_fraction) < 0.5:
        raise ValueError("validation_fraction must lie in (0,0.5)")
    if epochs <= 0 or batch_size <= 1:
        raise ValueError("epochs must be positive and batch_size must exceed one")
    if not math.isfinite(float(learning_rate)) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be finite and positive")
    if crossing_weight < 0.0:
        raise ValueError("crossing_weight must be non-negative")

    generator = torch.Generator().manual_seed(int(seed))
    permutation = torch.randperm(sample_count, generator=generator)
    validation_count = max(2, int(round(sample_count * validation_fraction)))
    training_count = sample_count - validation_count
    if training_count < 4:
        raise ValueError("auxiliary training split is too small")
    training_indices = permutation[:training_count]
    validation_indices = permutation[training_count:]

    normalizer = fit_gradient_feature_normalizer(
        gradient_features[training_indices].float()
    )
    normalized = normalizer.transform(gradient_features.float())
    input_values = inputs.float()
    target_values = None if targets is None else targets.float()
    if target_values is None:
        dataset = TensorDataset(normalized, input_values)
    else:
        dataset = TensorDataset(normalized, input_values, target_values)

    training_subset = torch.utils.data.Subset(dataset, training_indices.tolist())
    validation_subset = torch.utils.data.Subset(dataset, validation_indices.tolist())
    loader_generator = torch.Generator().manual_seed(int(seed) + 1)
    training_loader = DataLoader(
        training_subset,
        batch_size=min(int(batch_size), training_count),
        shuffle=True,
        drop_last=training_count > int(batch_size),
        generator=loader_generator,
    )
    validation_loader = DataLoader(
        validation_subset,
        batch_size=min(int(batch_size), validation_count),
        shuffle=False,
    )

    torch.manual_seed(int(seed))
    network = GradientToQuantileNetwork(
        gradient_features=int(gradient_features.shape[1]),
        input_shape=tuple(int(value) for value in inputs.shape[1:]),
        target_shape=(
            None if targets is None else tuple(int(value) for value in targets.shape[1:])
        ),
        hidden_sizes=hidden_sizes,
        dropout=dropout,
        quantiles=quantiles,
    )
    optimizer = torch.optim.AdamW(
        network.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )

    def epoch(loader: DataLoader, *, train: bool) -> float:
        network.train(train)
        losses: list[float] = []
        for batch in loader:
            gradients = batch[0]
            batch_inputs = batch[1]
            batch_targets = None if len(batch) == 2 else batch[2]
            if train:
                optimizer.zero_grad(set_to_none=True)
            predicted_inputs, predicted_targets = network(gradients)
            loss = pinball_quantile_loss(
                predicted_inputs, batch_inputs, network.quantiles
            )
            crossing = quantile_crossing_penalty(predicted_inputs)
            if batch_targets is not None:
                assert predicted_targets is not None
                loss = loss + pinball_quantile_loss(
                    predicted_targets, batch_targets, network.quantiles
                )
                crossing = crossing + quantile_crossing_penalty(predicted_targets)
            total = loss + float(crossing_weight) * crossing
            if train:
                total.backward()
                optimizer.step()
            losses.append(float(total.detach()))
        if not losses:
            raise RuntimeError("quantile initializer epoch produced no batches")
        return float(sum(losses) / len(losses))

    best_state: dict[str, torch.Tensor] | None = None
    best_validation = math.inf
    best_epoch = 0
    final_training = math.inf
    final_validation = math.inf
    for epoch_index in range(1, int(epochs) + 1):
        final_training = epoch(training_loader, train=True)
        with torch.no_grad():
            final_validation = epoch(validation_loader, train=False)
        if final_validation < best_validation:
            best_validation = final_validation
            best_epoch = epoch_index
            best_state = copy.deepcopy(network.state_dict())
    if best_state is None:
        raise RuntimeError("quantile initializer did not produce a finite validation model")
    network.load_state_dict(best_state)
    network.eval()

    report = QuantileInitializerTrainingReport(
        training_samples=training_count,
        validation_samples=validation_count,
        gradient_features=int(gradient_features.shape[1]),
        input_shape=tuple(int(value) for value in inputs.shape[1:]),
        target_shape=(
            None if targets is None else tuple(int(value) for value in targets.shape[1:])
        ),
        hidden_sizes=tuple(int(value) for value in hidden_sizes),
        quantiles=tuple(float(value) for value in quantiles),
        epochs=int(epochs),
        best_epoch=best_epoch,
        best_validation_loss=best_validation,
        final_training_loss=final_training,
        final_validation_loss=final_validation,
        normalizer=normalizer.to_dict(),
        model_sha256=_model_sha256(network),
        provenance=json.loads(json.dumps(LEARNED_QUANTILE_REFERENCE)),
    )
    return network, normalizer, report


def predict_gradient_quantiles(
    network: GradientToQuantileNetwork,
    normalizer: GradientFeatureNormalizer,
    gradients: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    flattened = flatten_gradient_tuple(gradients).float().unsqueeze(0)
    return network.inference(normalizer.transform(flattened))


def initialize_direct_prior_from_median(
    median: torch.Tensor,
    *,
    mode: str,
    bounded: bool = True,
    jitter_standard_deviation: float = 0.0,
    seed: int | None = None,
) -> DirectPrior:
    if jitter_standard_deviation < 0.0 or not math.isfinite(
        float(jitter_standard_deviation)
    ):
        raise ValueError("jitter_standard_deviation must be finite and non-negative")
    value = median.detach().float()
    prior = DirectPrior(tuple(value.shape), mode, bounded=bounded)
    if bounded:
        if mode == "timeseries":
            normalized = (value / 3.0).clamp(-0.999999, 0.999999)
            raw = torch.atanh(normalized)
        elif mode == "image":
            normalized = value.clamp(1e-6, 1.0 - 1e-6)
            raw = torch.logit(normalized)
        else:
            raw = value
    else:
        raw = value
    if jitter_standard_deviation > 0.0:
        generator = torch.Generator(device=raw.device)
        if seed is not None:
            generator.manual_seed(int(seed))
        raw = raw + torch.randn(
            raw.shape,
            generator=generator,
            device=raw.device,
            dtype=raw.dtype,
        ) * float(jitter_standard_deviation)
    with torch.no_grad():
        prior.raw.copy_(raw)
    return prior
