from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import TensorDataset


def _window(values: np.ndarray, context: int, horizon: int) -> tuple[np.ndarray, np.ndarray] | None:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size < context + horizon:
        return None
    segment = values[-(context + horizon) :]
    x, y = segment[:context], segment[context:]
    mean = float(x.mean())
    scale = float(x.std())
    scale = scale if scale > 1e-6 else 1.0
    return (x - mean) / scale, (y - mean) / scale


def _as_dataset(windows: Iterable[tuple[np.ndarray, np.ndarray]]) -> TensorDataset:
    pairs = list(windows)
    if not pairs:
        raise RuntimeError("no usable forecasting windows were found")
    x = torch.from_numpy(np.stack([pair[0] for pair in pairs])).float()
    y = torch.from_numpy(np.stack([pair[1] for pair in pairs])).float()
    return TensorDataset(x, y)


def load_gifteval(
    max_series: int = 128,
    context: int = 96,
    horizon: int = 24,
    streaming: bool = True,
) -> TensorDataset:
    """Load compact windows from Salesforce/GiftEval.

    The official dataset contains 144k series and 177M points. Streaming is the
    default so a reconstruction experiment does not download the full archive.
    """
    from datasets import load_dataset

    rows = load_dataset("Salesforce/GiftEval", split="train", streaming=streaming)
    result: list[tuple[np.ndarray, np.ndarray]] = []
    for row in rows:
        pair = _window(np.asarray(row["target"]), context, horizon)
        if pair is not None:
            result.append(pair)
        if len(result) >= max_series:
            break
    return _as_dataset(result)


def load_time_repository(
    root: str | Path,
    max_series: int = 128,
    context: int = 96,
    horizon: int = 24,
) -> TensorDataset:
    """Load numeric columns from a local checkout of the 2026 TIME benchmark.

    TIME contains heterogeneous source files. This adapter intentionally treats
    every sufficiently long numeric CSV column as an independent variate.
    """
    import pandas as pd

    root = Path(root)
    result: list[tuple[np.ndarray, np.ndarray]] = []
    for csv_path in sorted(root.rglob("*.csv")):
        frame = pd.read_csv(csv_path)
        for column in frame.select_dtypes(include=["number"]).columns:
            pair = _window(frame[column].to_numpy(), context, horizon)
            if pair is not None:
                result.append(pair)
            if len(result) >= max_series:
                return _as_dataset(result)
    return _as_dataset(result)


def synthetic_forecasting(
    samples: int = 32,
    context: int = 24,
    horizon: int = 6,
    seed: int = 7,
) -> TensorDataset:
    rng = np.random.default_rng(seed)
    windows = []
    time = np.linspace(0, 4 * np.pi, context + horizon, dtype=np.float32)
    for _ in range(samples):
        phase = rng.uniform(0, np.pi)
        trend = rng.uniform(-0.03, 0.03) * np.arange(time.size)
        values = np.sin(time + phase) + 0.25 * np.sin(3 * time) + trend
        values += rng.normal(0, 0.03, size=time.size)
        pair = _window(values, context, horizon)
        assert pair is not None
        windows.append(pair)
    return _as_dataset(windows)

