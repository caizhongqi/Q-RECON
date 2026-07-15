from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

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
    split: str = "train",
    revision: str | None = None,
) -> TensorDataset:
    """Load compact windows from Salesforce/GiftEval.

    ``revision`` is forwarded to Hugging Face Datasets so publication manifests
    can pin a commit SHA or immutable tag instead of silently tracking ``main``.
    """

    from datasets import load_dataset

    kwargs: dict[str, object] = {
        "split": split,
        "streaming": streaming,
    }
    if revision is not None:
        kwargs["revision"] = revision
    rows = load_dataset("Salesforce/GiftEval", **kwargs)
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


def load_multivariate_csv(
    path: str | Path,
    *,
    context: int = 96,
    horizon: int = 24,
    max_windows: int = 128,
    stride: int = 1,
    start: int = 0,
    columns: Sequence[str] | None = None,
) -> TensorDataset:
    """Load deterministic multivariate sliding windows from one CSV file.

    This adapter covers ETT, Electricity, Weather, Traffic and similar benchmark
    exports without silently flattening variables. Numeric columns are selected by
    default; callers may declare an exact ordered column list. Missing values are
    linearly interpolated in both directions and any remaining incomplete rows are
    removed before windowing. Each window is normalized per variable using only
    its context segment, preventing target leakage.
    """

    import pandas as pd

    if context <= 0 or horizon <= 0 or max_windows <= 0 or stride <= 0:
        raise ValueError("context, horizon, max_windows and stride must be positive")
    if start < 0:
        raise ValueError("start must be non-negative")
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    frame = pd.read_csv(csv_path)
    if columns is None:
        selected = frame.select_dtypes(include=["number"]).copy()
    else:
        declared = tuple(str(column) for column in columns)
        if not declared:
            raise ValueError("columns must be non-empty when provided")
        missing = [column for column in declared if column not in frame.columns]
        if missing:
            raise ValueError(f"CSV is missing declared columns: {missing}")
        selected = frame.loc[:, list(declared)].apply(pd.to_numeric, errors="coerce")
    if selected.shape[1] <= 0:
        raise RuntimeError("the CSV contains no usable numeric forecasting columns")

    selected = selected.replace([np.inf, -np.inf], np.nan)
    selected = selected.interpolate(method="linear", limit_direction="both")
    selected = selected.dropna(axis=0, how="any")
    values = selected.to_numpy(dtype=np.float32, copy=True)
    total = context + horizon
    if len(values) < total:
        raise RuntimeError(
            f"CSV has {len(values)} complete rows but requires at least {total}"
        )

    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for offset in range(start, len(values) - total + 1, stride):
        segment = values[offset : offset + total]
        context_values = segment[:context]
        mean = context_values.mean(axis=0, keepdims=True)
        scale = context_values.std(axis=0, keepdims=True)
        scale = np.where(scale > 1e-6, scale, 1.0)
        normalized = ((segment - mean) / scale).astype(np.float32)
        inputs.append(normalized[:context])
        targets.append(normalized[context:])
        if len(inputs) >= max_windows:
            break
    if not inputs:
        raise RuntimeError("no multivariate windows were generated")
    return TensorDataset(
        torch.from_numpy(np.stack(inputs)).float(),
        torch.from_numpy(np.stack(targets)).float(),
    )


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


def synthetic_multivariate_forecasting(
    samples: int = 32,
    context: int = 96,
    horizon: int = 24,
    channels: int = 7,
    seed: int = 7,
) -> TensorDataset:
    """Generate correlated multivariate windows for modern forecaster tests.

    The returned tensors have shapes ``[samples, context, channels]`` and
    ``[samples, horizon, channels]``. Channels combine shared seasonal factors,
    channel-specific frequencies, lags and trends so cross-variate attention is
    non-trivial while the generator remains deterministic.
    """

    if samples <= 0 or context <= 0 or horizon <= 0 or channels <= 0:
        raise ValueError("samples, context, horizon and channels must be positive")
    rng = np.random.default_rng(seed)
    total = context + horizon
    time = np.linspace(0, 8 * np.pi, total, dtype=np.float32)
    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for _ in range(samples):
        shared_phase = rng.uniform(0.0, 2.0 * np.pi)
        shared = np.sin(time + shared_phase) + 0.3 * np.sin(0.25 * time)
        series: list[np.ndarray] = []
        for channel in range(channels):
            lag = channel % max(1, min(8, context // 4))
            frequency = 1.0 + 0.08 * channel
            local = 0.45 * np.sin(frequency * time + rng.uniform(0.0, np.pi))
            trend = rng.uniform(-0.01, 0.01) * np.arange(total, dtype=np.float32)
            values = np.roll(shared, lag) + local + trend
            values += rng.normal(0.0, 0.03, size=total)
            context_values = values[:context]
            mean = float(context_values.mean())
            scale = float(context_values.std())
            scale = scale if scale > 1e-6 else 1.0
            series.append(((values - mean) / scale).astype(np.float32))
        matrix = np.stack(series, axis=-1)
        inputs.append(matrix[:context])
        targets.append(matrix[context:])
    return TensorDataset(
        torch.from_numpy(np.stack(inputs)).float(),
        torch.from_numpy(np.stack(targets)).float(),
    )
