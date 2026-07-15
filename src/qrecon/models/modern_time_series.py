from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

ActivationName = Literal["gelu", "relu"]


def _validate_transformer_dimensions(d_model: int, n_heads: int, e_layers: int, d_ff: int) -> None:
    if d_model <= 0 or n_heads <= 0 or e_layers <= 0 or d_ff <= 0:
        raise ValueError("d_model, n_heads, e_layers and d_ff must be positive")
    if d_model % n_heads != 0:
        raise ValueError("d_model must be divisible by n_heads")


def _as_btc(
    x: torch.Tensor,
    *,
    context: int,
    input_channels: int,
) -> tuple[torch.Tensor, bool]:
    """Return ``[batch, time, channels]`` and whether the input was 2-D."""

    if x.ndim == 2:
        if input_channels != 1:
            raise ValueError(
                "2-D forecasting input is only valid when input_channels=1; "
                "use [batch, time, channels] for multivariate data"
            )
        if x.shape[1] != context:
            raise ValueError(f"expected context length {context}, observed {x.shape[1]}")
        return x.unsqueeze(-1), True
    if x.ndim != 3:
        raise ValueError(
            "forecasting input must have shape [batch, time] or "
            "[batch, time, channels]"
        )
    if x.shape[1] != context:
        raise ValueError(f"expected context length {context}, observed {x.shape[1]}")
    if x.shape[2] != input_channels:
        raise ValueError(f"expected {input_channels} input channels, observed {x.shape[2]}")
    return x, False


def _restore_forecast_shape(y: torch.Tensor, squeezed_univariate: bool) -> torch.Tensor:
    return y[..., 0] if squeezed_univariate else y


class RevIN(nn.Module):
    """Reversible instance normalization for time-major multivariate tensors."""

    def __init__(self, channels: int, *, affine: bool = True, eps: float = 1e-5) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        self.channels = int(channels)
        self.affine = bool(affine)
        self.eps = float(eps)
        if self.affine:
            self.weight = nn.Parameter(torch.ones(1, 1, channels))
            self.bias = nn.Parameter(torch.zeros(1, 1, channels))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def normalize(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if x.ndim != 3 or x.shape[-1] != self.channels:
            raise ValueError(
                "RevIN expects [batch, time, channels] with the declared channel count"
            )
        mean = x.mean(dim=1, keepdim=True)
        variance = (x - mean).square().mean(dim=1, keepdim=True)
        stdev = torch.sqrt(variance + self.eps)
        normalized = (x - mean) / stdev
        if self.affine:
            normalized = normalized * self.weight + self.bias
        return normalized, (mean, stdev)

    def denormalize(
        self,
        x: torch.Tensor,
        statistics: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        mean, stdev = statistics
        restored = x
        if self.affine:
            safe_weight = torch.where(
                self.weight >= 0,
                self.weight.clamp_min(self.eps),
                self.weight.clamp_max(-self.eps),
            )
            restored = (restored - self.bias) / safe_weight
        return restored * stdev + mean


class MultiHeadSelfAttention(nn.Module):
    """Unfused attention chosen to preserve reliable higher-order autograd."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.output = nn.Linear(d_model, d_model)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.n_heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        attention = self.attention_dropout(torch.softmax(scores, dim=-1))
        mixed = torch.matmul(attention, value)
        mixed = mixed.transpose(1, 2).reshape(batch, tokens, self.d_model)
        return self.output_dropout(self.output(mixed))


class ForecastEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        activation: ActivationName,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        activation_layer: nn.Module = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            activation_layer,
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        return x + self.feed_forward(self.norm2(x))


class ForecastEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        e_layers: int,
        d_ff: int,
        dropout: float,
        activation: ActivationName,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            ForecastEncoderLayer(d_model, n_heads, d_ff, dropout, activation)
            for _ in range(e_layers)
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)


@dataclass(frozen=True)
class PatchTSTGeometry:
    context: int
    patch_len: int
    stride: int
    padding: int
    patch_count: int


def patchtst_geometry(
    context: int,
    patch_len: int,
    stride: int,
    *,
    padding_patch: bool,
) -> PatchTSTGeometry:
    if context <= 0 or patch_len <= 0 or stride <= 0:
        raise ValueError("context, patch_len and stride must be positive")
    if patch_len > context:
        raise ValueError("patch_len must not exceed context")
    padding = stride if padding_patch else 0
    patch_count = 1 + (context + padding - patch_len) // stride
    if patch_count <= 0:
        raise ValueError("patch configuration produces no patches")
    return PatchTSTGeometry(context, patch_len, stride, padding, patch_count)


class TransformerForecaster(nn.Module):
    """Encoder-only temporal-token Transformer for direct forecasting."""

    def __init__(
        self,
        context: int,
        horizon: int,
        *,
        input_channels: int = 1,
        d_model: int = 64,
        n_heads: int = 4,
        e_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.1,
        activation: ActivationName = "gelu",
        revin: bool = True,
    ) -> None:
        super().__init__()
        _validate_transformer_dimensions(d_model, n_heads, e_layers, d_ff)
        if context <= 0 or horizon <= 0 or input_channels <= 0:
            raise ValueError("context, horizon and input_channels must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if activation not in ("gelu", "relu"):
            raise ValueError("activation must be 'gelu' or 'relu'")
        self.context = int(context)
        self.horizon = int(horizon)
        self.input_channels = int(input_channels)
        self.revin = RevIN(input_channels) if revin else None
        self.input_projection = nn.Linear(input_channels, d_model)
        self.position = nn.Parameter(torch.zeros(1, context, d_model))
        nn.init.trunc_normal_(self.position, std=0.02)
        self.encoder = ForecastEncoder(
            d_model, n_heads, e_layers, d_ff, dropout, activation
        )
        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Dropout(dropout),
            nn.Linear(context * d_model, horizon * input_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence, squeezed = _as_btc(
            x, context=self.context, input_channels=self.input_channels
        )
        statistics = None
        if self.revin is not None:
            sequence, statistics = self.revin.normalize(sequence)
        encoded = self.encoder(self.input_projection(sequence) + self.position)
        forecast = self.head(encoded).reshape(
            sequence.shape[0], self.horizon, self.input_channels
        )
        if self.revin is not None and statistics is not None:
            forecast = self.revin.denormalize(forecast, statistics)
        return _restore_forecast_shape(forecast, squeezed)


class PatchTST(nn.Module):
    """Channel-independent patch Transformer following PatchTST.

    Variates are split into overlapping temporal patches and folded into the
    batch dimension. Patch embedding and encoder weights are shared across all
    variates, matching the channel-independent PatchTST design.
    """

    def __init__(
        self,
        context: int,
        horizon: int,
        *,
        input_channels: int = 1,
        patch_len: int = 16,
        stride: int = 8,
        padding_patch: bool = True,
        d_model: int = 64,
        n_heads: int = 4,
        e_layers: int = 3,
        d_ff: int = 128,
        dropout: float = 0.1,
        head_dropout: float = 0.0,
        activation: ActivationName = "gelu",
        revin: bool = True,
        individual_head: bool = False,
    ) -> None:
        super().__init__()
        _validate_transformer_dimensions(d_model, n_heads, e_layers, d_ff)
        if horizon <= 0 or input_channels <= 0:
            raise ValueError("horizon and input_channels must be positive")
        if not 0.0 <= dropout < 1.0 or not 0.0 <= head_dropout < 1.0:
            raise ValueError("dropout values must lie in [0, 1)")
        if activation not in ("gelu", "relu"):
            raise ValueError("activation must be 'gelu' or 'relu'")
        self.geometry = patchtst_geometry(
            context, patch_len, stride, padding_patch=padding_patch
        )
        self.context = int(context)
        self.horizon = int(horizon)
        self.input_channels = int(input_channels)
        self.individual_head = bool(individual_head)
        self.revin = RevIN(input_channels) if revin else None
        self.patch_projection = nn.Linear(patch_len, d_model)
        self.position = nn.Parameter(
            torch.zeros(1, self.geometry.patch_count, d_model)
        )
        nn.init.trunc_normal_(self.position, std=0.02)
        self.encoder = ForecastEncoder(
            d_model, n_heads, e_layers, d_ff, dropout, activation
        )
        head_input = self.geometry.patch_count * d_model
        if self.individual_head:
            self.heads = nn.ModuleList(
                nn.Sequential(nn.Dropout(head_dropout), nn.Linear(head_input, horizon))
                for _ in range(input_channels)
            )
            self.head = None
        else:
            self.head = nn.Sequential(
                nn.Dropout(head_dropout), nn.Linear(head_input, horizon)
            )
            self.heads = None

    def _patches(self, x: torch.Tensor) -> torch.Tensor:
        channels_first = x.transpose(1, 2)
        if self.geometry.padding:
            channels_first = F.pad(
                channels_first,
                (0, self.geometry.padding),
                mode="replicate",
            )
        patches = channels_first.unfold(
            dimension=-1,
            size=self.geometry.patch_len,
            step=self.geometry.stride,
        )
        if patches.shape[-2] != self.geometry.patch_count:
            raise RuntimeError(
                "PatchTST patch geometry mismatch: "
                f"expected {self.geometry.patch_count}, observed {patches.shape[-2]}"
            )
        return patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence, squeezed = _as_btc(
            x, context=self.context, input_channels=self.input_channels
        )
        statistics = None
        if self.revin is not None:
            sequence, statistics = self.revin.normalize(sequence)
        patches = self._patches(sequence)
        batch, channels, patch_count, patch_len = patches.shape
        tokens = patches.reshape(batch * channels, patch_count, patch_len)
        encoded = self.encoder(self.patch_projection(tokens) + self.position)
        flattened = encoded.reshape(batch, channels, -1)
        if self.individual_head:
            assert self.heads is not None
            forecast = torch.stack(
                [
                    head(flattened[:, channel])
                    for channel, head in enumerate(self.heads)
                ],
                dim=-1,
            )
        else:
            assert self.head is not None
            forecast = self.head(flattened).transpose(1, 2)
        if self.revin is not None and statistics is not None:
            forecast = self.revin.denormalize(forecast, statistics)
        return _restore_forecast_shape(forecast, squeezed)


class ITransformer(nn.Module):
    """Inverted Transformer using complete variates as attention tokens."""

    def __init__(
        self,
        context: int,
        horizon: int,
        *,
        input_channels: int = 1,
        d_model: int = 64,
        n_heads: int = 4,
        e_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.1,
        activation: ActivationName = "gelu",
        revin: bool = True,
    ) -> None:
        super().__init__()
        _validate_transformer_dimensions(d_model, n_heads, e_layers, d_ff)
        if context <= 0 or horizon <= 0 or input_channels <= 0:
            raise ValueError("context, horizon and input_channels must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if activation not in ("gelu", "relu"):
            raise ValueError("activation must be 'gelu' or 'relu'")
        self.context = int(context)
        self.horizon = int(horizon)
        self.input_channels = int(input_channels)
        self.revin = RevIN(input_channels) if revin else None
        self.variate_projection = nn.Linear(context, d_model)
        self.encoder = ForecastEncoder(
            d_model, n_heads, e_layers, d_ff, dropout, activation
        )
        self.head = nn.Linear(d_model, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence, squeezed = _as_btc(
            x, context=self.context, input_channels=self.input_channels
        )
        statistics = None
        if self.revin is not None:
            sequence, statistics = self.revin.normalize(sequence)
        variate_tokens = self.variate_projection(sequence.transpose(1, 2))
        encoded = self.encoder(variate_tokens)
        forecast = self.head(encoded).transpose(1, 2)
        if self.revin is not None and statistics is not None:
            forecast = self.revin.denormalize(forecast, statistics)
        return _restore_forecast_shape(forecast, squeezed)
