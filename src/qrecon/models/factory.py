from __future__ import annotations

from typing import Any

from torch import nn

from .modern_time_series import ITransformer, PatchTST, TransformerForecaster
from .time_series import ForecastMLP

FORECASTING_ARCHITECTURES = (
    "mlp",
    "transformer",
    "patchtst",
    "itransformer",
)


def _architecture_name(value: object) -> str:
    return str(value).strip().lower().replace("-", "").replace("_", "")


def build_forecasting_model(
    context: int,
    horizon: int,
    input_channels: int,
    config: dict[str, Any],
) -> nn.Module:
    """Build a forecasting victim from a YAML-compatible configuration."""

    architecture = _architecture_name(config.get("architecture", "mlp"))
    if architecture in {"mlp", "forecastmlp"}:
        if input_channels != 1:
            raise ValueError(
                "ForecastMLP supports only univariate [batch, time] inputs; "
                "choose transformer, patchtst or itransformer for multivariate data"
            )
        return ForecastMLP(context, horizon, int(config.get("hidden", 64)))

    common = {
        "context": int(context),
        "horizon": int(horizon),
        "input_channels": int(input_channels),
        "d_model": int(config.get("d_model", 64)),
        "n_heads": int(config.get("n_heads", 4)),
        "e_layers": int(config.get("e_layers", 2)),
        "d_ff": int(config.get("d_ff", 128)),
        "dropout": float(config.get("dropout", 0.1)),
        "activation": str(config.get("activation", "gelu")),
        "revin": bool(config.get("revin", True)),
    }
    if architecture in {
        "transformer",
        "transformerencoder",
        "vanillatransformer",
    }:
        return TransformerForecaster(**common)
    if architecture in {"patchtst", "patchtransformer"}:
        patch_len = int(config.get("patch_len", min(16, context)))
        stride = int(config.get("stride", min(8, patch_len)))
        return PatchTST(
            **common,
            patch_len=patch_len,
            stride=stride,
            padding_patch=bool(config.get("padding_patch", True)),
            head_dropout=float(config.get("head_dropout", 0.0)),
            individual_head=bool(config.get("individual_head", False)),
        )
    if architecture in {"itransformer", "invertedtransformer"}:
        return ITransformer(**common)
    supported = ", ".join(FORECASTING_ARCHITECTURES)
    raise ValueError(
        f"unknown forecasting architecture {config.get('architecture')!r}; "
        f"supported architectures: {supported}"
    )
