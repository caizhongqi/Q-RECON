from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .attacks import (
    GradientInversionAttack,
    infer_class_label_from_last_bias,
    invert_first_linear_gradient,
    leak_gradients,
)
from .data import (
    load_community_forensics,
    load_gifteval,
    load_image_folder,
    load_time_repository,
    synthetic_forecasting,
    synthetic_multivariate_forecasting,
)
from .identifiability import gradient_jacobian_report
from .metrics import reconstruction_metrics
from .models import ImageMLP, SmallLeNet, TinyConvNet, build_forecasting_model
from .quantum import ClassicalPrior, DirectPrior, QuantumPrior, quantum_resource_estimate


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_dataset(config: dict[str, Any]) -> tuple[TensorDataset, str, str]:
    dataset = config["dataset"]
    name = dataset["name"]
    if name == "synthetic_time":
        return synthetic_forecasting(
            samples=dataset.get("max_samples", 32),
            context=dataset["context"],
            horizon=dataset["horizon"],
            seed=config.get("seed", 7),
        ), "forecasting", "timeseries"
    if name == "synthetic_multivariate_time":
        return synthetic_multivariate_forecasting(
            samples=dataset.get("max_samples", 32),
            context=dataset["context"],
            horizon=dataset["horizon"],
            channels=dataset.get("channels", 7),
            seed=config.get("seed", 7),
        ), "forecasting", "timeseries"
    if name == "gift_eval":
        return load_gifteval(
            max_series=dataset.get("max_samples", 128),
            context=dataset["context"],
            horizon=dataset["horizon"],
            streaming=dataset.get("streaming", True),
            split=dataset.get("split", "train"),
            revision=dataset.get("revision"),
        ), "forecasting", "timeseries"
    if name == "time_2026":
        return load_time_repository(
            dataset["root"],
            dataset.get("max_samples", 128),
            dataset["context"],
            dataset["horizon"],
        ), "forecasting", "timeseries"
    if name == "community_forensics_small":
        return load_community_forensics(
            max_images=dataset.get("max_samples", 128),
            image_size=dataset.get("image_size", 32),
            streaming=dataset.get("streaming", True),
            seed=config.get("seed", 17),
            sampling=dataset.get("sampling", "api"),
            real_offset=dataset.get("real_offset", 9000),
        ), "classification", "image"
    if name == "image_folder":
        return load_image_folder(
            dataset["root"], dataset.get("image_size", 32)
        ), "classification", "image"
    raise ValueError(f"unknown dataset: {name}")


def _forecast_dimensions(dataset: TensorDataset) -> tuple[int, int, int]:
    x, y = dataset.tensors
    if x.ndim == 2:
        if y.ndim != 2:
            raise ValueError(
                "univariate forecasting targets must have shape [samples, horizon]"
            )
        return int(x.shape[1]), int(y.shape[1]), 1
    if x.ndim == 3:
        if y.ndim != 3:
            raise ValueError(
                "multivariate forecasting targets must have shape "
                "[samples, horizon, channels]"
            )
        if x.shape[2] != y.shape[2]:
            raise ValueError("forecast input and target channel counts must match")
        return int(x.shape[1]), int(y.shape[1]), int(x.shape[2])
    raise ValueError(
        "forecasting inputs must have shape [samples, context] or "
        "[samples, context, channels]"
    )


def _build_model(dataset: TensorDataset, task: str, config: dict[str, Any]) -> nn.Module:
    x, y = dataset.tensors
    if task == "forecasting":
        context, horizon, input_channels = _forecast_dimensions(dataset)
        return build_forecasting_model(context, horizon, input_channels, config)
    classes = max(2, int(y.max().item()) + 1)
    architecture = str(config.get("architecture", "cnn")).lower()
    if architecture == "mlp":
        return ImageMLP(tuple(x.shape[1:]), classes, config.get("hidden", 128))
    if architecture == "lenet":
        return SmallLeNet(tuple(x.shape[1:]), classes, config.get("width", 6))
    if architecture in {"cnn", "tinyconvnet", "tiny_conv_net"}:
        return TinyConvNet(classes, config.get("width", 24))
    raise ValueError(
        f"unknown image architecture {config.get('architecture')!r}; "
        "supported architectures: mlp, lenet, cnn"
    )


def _train(model: nn.Module, dataset: TensorDataset, task: str, config: dict[str, Any]) -> None:
    loader = DataLoader(
        dataset, batch_size=config.get("batch_size", 16), shuffle=True
    )
    learning_rate = float(config.get("learning_rate", 1e-3))
    optimizer_name = str(config.get("optimizer", "adam")).lower()
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=float(config.get("weight_decay", 1e-2)),
        )
    else:
        raise ValueError("training optimizer must be 'adam' or 'adamw'")
    criterion: nn.Module = (
        nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()
    )
    gradient_clip_norm = config.get("gradient_clip_norm")
    model.train()
    for _ in range(config.get("epochs", 5)):
        for x, target in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), target)
            loss.backward()
            if gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(gradient_clip_norm)
                )
            optimizer.step()
    model.eval()


def _prior(shape: tuple[int, ...], mode: str, config: dict[str, Any]) -> nn.Module:
    kind = config.get("prior", "direct")
    if kind == "direct":
        return DirectPrior(shape, mode, bounded=config.get("bounded", True))
    if kind == "classical":
        return ClassicalPrior(
            shape, mode, config.get("latent_dim", 8), config.get("hidden", 64)
        )
    if kind == "quantum":
        return QuantumPrior(
            shape,
            mode,
            n_qubits=config.get("n_qubits", 6),
            layers=config.get("layers", 2),
            hidden=config.get("hidden", 64),
            shots=config.get("shots"),
        )
    raise ValueError(f"unknown prior: {kind}")


def run_experiment(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config.get("seed", 7))
    _seed_everything(seed)
    dataset, task, mode = _load_dataset(config)
    victim_config = config.get("victim", {})
    model = _build_model(dataset, task, victim_config)
    _train(model, dataset, task, config.get("training", {}))

    true_x, true_target = (tensor[:1].clone() for tensor in dataset.tensors)
    observed = leak_gradients(model, true_x, true_target, task)
    attack_config = config.get("attack", {})
    known_target = true_target if attack_config.get("known_target", True) else None
    if known_target is None and task == "classification":
        known_target = infer_class_label_from_last_bias(model, observed)

    method = attack_config.get("method", "gradient_matching")
    if method == "analytic_linear":
        reconstruction = invert_first_linear_gradient(
            model, observed, tuple(true_x.shape)
        )
        reconstructed_target = (
            known_target.detach()
            if known_target is not None
            else torch.full_like(true_target, float("nan"))
        )
        history = [
            {
                "step": 1.0,
                "objective": 0.0,
                "gradient_match": 0.0,
                "regularizer": 0.0,
            }
        ]
    else:
        prior = _prior(tuple(true_x.shape), mode, attack_config)
        attack = GradientInversionAttack(
            model=model,
            observed_gradients=observed,
            prior=prior,
            task=task,
            mode=mode,
            known_target=known_target,
            target_shape=tuple(true_target.shape),
            steps=attack_config.get("steps", 300),
            learning_rate=attack_config.get("learning_rate", 0.05),
            regularization=attack_config.get("regularization", 1e-3),
            optimizer_name=attack_config.get("optimizer", "adam"),
        )
        result = attack.run()
        reconstruction = result.reconstruction
        reconstructed_target = result.reconstructed_target
        history = result.history

    report: dict[str, Any] = {
        "dataset": config["dataset"]["name"],
        "task": task,
        "victim": {
            "class": type(model).__name__,
            "architecture": victim_config.get(
                "architecture", "mlp" if task == "forecasting" else "cnn"
            ),
            "parameters": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "config": victim_config,
        },
        "attack_method": method,
        "prior": (
            attack_config.get("prior") if method != "analytic_linear" else None
        ),
        "metrics": reconstruction_metrics(true_x, reconstruction, mode=mode),
        "history": history,
    }
    max_identifiability_dim = config.get("identifiability", {}).get(
        "max_input_dimension", 256
    )
    if true_x.numel() <= max_identifiability_dim:
        report["identifiability"] = gradient_jacobian_report(
            model, true_x, true_target, task
        ).to_dict()
    if attack_config.get("prior") == "quantum" and method != "analytic_linear":
        report["quantum_resources"] = quantum_resource_estimate(
            attack_config.get("n_qubits", 6),
            attack_config.get("layers", 2),
            attack_config.get("shots"),
        )

    output_dir = Path(config.get("output_dir", "outputs/latest"))
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "reference": true_x,
            "reconstruction": reconstruction,
            "target": true_target,
            "reconstructed_target": reconstructed_target,
        },
        output_dir / "reconstruction.pt",
    )
    with (output_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=True)
    return report
