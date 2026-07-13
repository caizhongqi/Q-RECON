from __future__ import annotations

import math
from functools import reduce
from operator import mul

import torch
from torch import nn


def _elements(shape: tuple[int, ...]) -> int:
    return reduce(mul, shape, 1)


def _bound(raw: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "image":
        return torch.sigmoid(raw)
    if mode == "timeseries":
        return 3.0 * torch.tanh(raw)
    return raw


class DirectPrior(nn.Module):
    def __init__(self, shape: tuple[int, ...], mode: str):
        super().__init__()
        self.shape = shape
        self.mode = mode
        self.raw = nn.Parameter(torch.randn(shape) * 0.1)

    def forward(self) -> torch.Tensor:
        return _bound(self.raw, self.mode)


class ClassicalPrior(nn.Module):
    def __init__(
        self,
        shape: tuple[int, ...],
        mode: str,
        latent_dim: int = 8,
        hidden: int = 64,
    ):
        super().__init__()
        self.shape = shape
        self.mode = mode
        self.latent = nn.Parameter(torch.randn(1, latent_dim) * 0.1)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.GELU(), nn.Linear(hidden, _elements(shape))
        )

    def forward(self) -> torch.Tensor:
        raw = self.decoder(self.latent).reshape(self.shape)
        return _bound(raw, self.mode)


class QuantumPrior(nn.Module):
    """Small VQC latent prior followed by a classical readout decoder.

    The VQC operates in latent space; high-dimensional classical data are never
    amplitude encoded. This keeps state preparation explicit and measurable.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        mode: str,
        n_qubits: int = 6,
        layers: int = 2,
        hidden: int = 64,
        shots: int | None = None,
    ):
        super().__init__()
        try:
            import pennylane as qml
        except ImportError as exc:
            raise ImportError("QuantumPrior requires `pip install -e .[quantum]`") from exc

        self.shape = shape
        self.mode = mode
        self.n_qubits = n_qubits
        self.layers = layers
        self.shots = shots
        self.latent = nn.Parameter(torch.randn(1, n_qubits) * 0.1)
        device = qml.device("default.qubit", wires=n_qubits, shots=shots)

        @qml.qnode(device, interface="torch", diff_method="best")
        def circuit(inputs: torch.Tensor, weights: torch.Tensor):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(wire)) for wire in range(n_qubits)]

        shapes = {"weights": (layers, n_qubits, 3)}
        self.quantum_layer = qml.qnn.TorchLayer(circuit, shapes)
        self.decoder = nn.Sequential(
            nn.Linear(n_qubits, hidden), nn.GELU(), nn.Linear(hidden, _elements(shape))
        )

    def forward(self) -> torch.Tensor:
        features = self.quantum_layer(self.latent)
        raw = self.decoder(features).reshape(self.shape)
        return _bound(raw, self.mode)


def quantum_resource_estimate(n_qubits: int, layers: int, shots: int | None) -> dict[str, int | None]:
    # StronglyEntanglingLayers uses three rotations per qubit and one entangler
    # per qubit in each layer. Values are logical-operation estimates.
    return {
        "logical_qubits": n_qubits,
        "single_qubit_rotations": n_qubits + 3 * n_qubits * layers,
        "two_qubit_entanglers": n_qubits * layers,
        "nominal_depth": 1 + 4 * layers,
        "shots_per_forward": shots,
    }

