from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .attacks import GradientInversionAttack, leak_gradients
from .data import (
    load_community_forensics,
    load_gifteval,
    load_image_folder,
    load_time_repository,
    synthetic_forecasting,
)
from .identifiability import gradient_jacobian_report
from .metrics import reconstruction_metrics
from .models import ForecastMLP, TinyConvNet
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
    if name == "gift_eval":
        return load_gifteval(
            max_series=dataset.get("max_samples", 128),
            context=dataset["context"],
            horizon=dataset["horizon"],
            streaming=dataset.get("streaming", True),
        ), "forecasting", "timeseries"
    if name == "time_2026":
        return load_time_repository(
            dataset["root"], dataset.get("max_samples", 128), dataset["context"], dataset["horizon"]
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
        return load_image_folder(dataset["root"], dataset.get("image_size", 32)), "classification", "image"
    raise ValueError(f"unknown dataset: {name}")


def _build_model(dataset: TensorDataset, task: str, config: dict[str, Any]) -> nn.Module:
    x, y = dataset.tensors
    if task == "forecasting":
        return ForecastMLP(x.shape[-1], y.shape[-1], config.get("hidden", 64))
    classes = max(2, int(y.max().item()) + 1)
    return TinyConvNet(classes, config.get("width", 24))


def _train(model: nn.Module, dataset: TensorDataset, task: str, config: dict[str, Any]) -> None:
    loader = DataLoader(dataset, batch_size=config.get("batch_size", 16), shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("learning_rate", 1e-3))
    criterion: nn.Module = nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()
    model.train()
    for _ in range(config.get("epochs", 5)):
        for x, target in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), target)
            loss.backward()
            optimizer.step()
    model.eval()


def _prior(shape: tuple[int, ...], mode: str, config: dict[str, Any]) -> nn.Module:
    kind = config.get("prior", "direct")
    if kind == "direct":
        return DirectPrior(shape, mode)
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
    model = _build_model(dataset, task, config.get("victim", {}))
    _train(model, dataset, task, config.get("training", {}))

    true_x, true_target = (tensor[:1].clone() for tensor in dataset.tensors)
    observed = leak_gradients(model, true_x, true_target, task)
    prior = _prior(tuple(true_x.shape), mode, config.get("attack", {}))
    known_target = true_target if config.get("attack", {}).get("known_target", True) else None
    attack = GradientInversionAttack(
        model=model,
        observed_gradients=observed,
        prior=prior,
        task=task,
        mode=mode,
        known_target=known_target,
        target_shape=tuple(true_target.shape),
        steps=config.get("attack", {}).get("steps", 300),
        learning_rate=config.get("attack", {}).get("learning_rate", 0.05),
        regularization=config.get("attack", {}).get("regularization", 1e-3),
    )
    result = attack.run()

    report: dict[str, Any] = {
        "dataset": config["dataset"]["name"],
        "task": task,
        "prior": config.get("attack", {}).get("prior", "direct"),
        "metrics": reconstruction_metrics(true_x, result.reconstruction),
        "history": result.history,
    }
    max_identifiability_dim = config.get("identifiability", {}).get("max_input_dimension", 256)
    if true_x.numel() <= max_identifiability_dim:
        report["identifiability"] = gradient_jacobian_report(
            model, true_x, true_target, task
        ).to_dict()
    if config.get("attack", {}).get("prior") == "quantum":
        attack_config = config["attack"]
        report["quantum_resources"] = quantum_resource_estimate(
            attack_config.get("n_qubits", 6), attack_config.get("layers", 2), attack_config.get("shots")
        )

    output_dir = Path(config.get("output_dir", "outputs/latest"))
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "reference": true_x,
            "reconstruction": result.reconstruction,
            "target": true_target,
            "reconstructed_target": result.reconstructed_target,
        },
        output_dir / "reconstruction.pt",
    )
    with (output_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=True)
    return report
