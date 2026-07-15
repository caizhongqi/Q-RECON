from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ForecastMLP(nn.Module):
    def __init__(self, context: int, horizon: int, hidden: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(context, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def _validate_transformer_dimensions(
    *, context: int, horizon: int, d_model: int, n_heads: int, num_layers: int
) -> None:
    if context <= 0 or horizon <= 0:
        raise ValueError("context and horizon must be positive")
    if d_model <= 0 or n_heads <= 0 or num_layers <= 0:
        raise ValueError("d_model, n_heads, and num_layers must be positive")
    if d_model % n_heads != 0:
        raise ValueError("d_model must be divisible by n_heads")


def _channel_first(x: torch.Tensor, channels: int) -> tuple[torch.Tensor, bool]:
    if x.ndim == 2:
        if channels != 1:
            raise ValueError("two-dimensional inputs are only valid for channels=1")
        return x.unsqueeze(1), True
    if x.ndim == 3:
        if x.shape[1] != channels:
            raise ValueError(f"expected {channels} channels, observed {x.shape[1]}")
        return x, channels == 1
    raise ValueError("time-series inputs must have shape [batch, context] or [batch, channels, context]")


def _restore_output_shape(y: torch.Tensor, squeeze_channel: bool) -> torch.Tensor:
    return y[:, 0] if squeeze_channel else y


class ReversibleInstanceNorm(nn.Module):
    """Per-sample, per-channel normalization with optional learned affine terms."""

    def __init__(self, channels: int, *, affine: bool = True, eps: float = 1e-5):
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        self.channels = int(channels)
        self.affine = bool(affine)
        self.eps = float(eps)
        if self.affine:
            self.weight = nn.Parameter(torch.ones(1, channels, 1))
            self.bias = nn.Parameter(torch.zeros(1, channels, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def normalize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = x.mean(dim=-1, keepdim=True).detach()
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        scale = torch.sqrt(variance + self.eps).detach()
        normalized = (x - mean) / scale
        if self.affine:
            normalized = normalized * self.weight + self.bias
        return normalized, mean, scale

    def denormalize(
        self, y: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        restored = y
        if self.affine:
            restored = (restored - self.bias) / (self.weight + self.eps)
        return restored * scale + mean


class _SequenceNorm(nn.Module):
    def __init__(self, d_model: int, kind: str):
        super().__init__()
        normalized = kind.lower()
        if normalized == "batch":
            self.kind = "batch"
            self.norm: nn.Module = nn.BatchNorm1d(d_model)
        elif normalized == "layer":
            self.kind = "layer"
            self.norm = nn.LayerNorm(d_model)
        else:
            raise ValueError("norm must be 'batch' or 'layer'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "batch":
            return self.norm(x.transpose(1, 2)).transpose(1, 2)
        return self.norm(x)


class _DifferentiableMultiheadSelfAttention(nn.Module):
    """Explicit scaled dot-product attention with higher-order autograd support.

    PyTorch's optimized ``MultiheadAttention`` may dispatch to flash-attention
    kernels whose second derivative is unavailable. Gradient inversion requires
    differentiating through parameter gradients, so Q-RECON intentionally uses
    the transparent matmul-softmax formulation here.
    """

    def __init__(
        self, d_model: int, n_heads: int, *, attention_dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = self.d_model // self.n_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(self.d_model, 3 * self.d_model)
        self.output = nn.Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = x.shape
        qkv = self.qkv(x).reshape(
            batch, tokens, 3, self.n_heads, self.head_dim
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        weights = self.dropout(torch.softmax(scores, dim=-1))
        attended = torch.matmul(weights, value)
        attended = attended.transpose(1, 2).reshape(batch, tokens, self.d_model)
        return self.output(attended)


class _TimeSeriesEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ff_dim: int,
        *,
        dropout: float,
        attention_dropout: float,
        norm: str,
    ) -> None:
        super().__init__()
        if ff_dim <= 0:
            raise ValueError("ff_dim must be positive")
        if not 0.0 <= dropout < 1.0 or not 0.0 <= attention_dropout < 1.0:
            raise ValueError("dropout probabilities must lie in [0, 1)")
        self.attention = _DifferentiableMultiheadSelfAttention(
            d_model, n_heads, attention_dropout=attention_dropout
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.norm1 = _SequenceNorm(d_model, norm)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.feed_forward_dropout = nn.Dropout(dropout)
        self.norm2 = _SequenceNorm(d_model, norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attended = self.attention(x)
        x = self.norm1(x + self.attention_dropout(attended))
        return self.norm2(x + self.feed_forward_dropout(self.feed_forward(x)))


class _TimeSeriesEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_layers: int,
        ff_dim: int,
        *,
        dropout: float,
        attention_dropout: float,
        norm: str,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            _TimeSeriesEncoderLayer(
                d_model,
                n_heads,
                ff_dim,
                dropout=dropout,
                attention_dropout=attention_dropout,
                norm=norm,
            )
            for _ in range(num_layers)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class TransformerForecaster(nn.Module):
    """Channel-independent point-token Transformer forecasting baseline.

    Inputs use ``[batch, context]`` for one channel or
    ``[batch, channels, context]`` for multivariate channel-independent use.
    """

    def __init__(
        self,
        context: int,
        horizon: int,
        *,
        channels: int = 1,
        d_model: int = 64,
        n_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 128,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        head_dropout: float = 0.0,
        norm: str = "layer",
        revin: bool = True,
        revin_affine: bool = True,
    ) -> None:
        super().__init__()
        _validate_transformer_dimensions(
            context=context,
            horizon=horizon,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
        )
        if channels <= 0:
            raise ValueError("channels must be positive")
        if not 0.0 <= head_dropout < 1.0:
            raise ValueError("head_dropout must lie in [0, 1)")
        self.context = int(context)
        self.horizon = int(horizon)
        self.channels = int(channels)
        self.revin = (
            ReversibleInstanceNorm(channels, affine=revin_affine) if revin else None
        )
        self.input_projection = nn.Linear(1, d_model)
        self.position = nn.Parameter(torch.empty(1, context, d_model))
        nn.init.trunc_normal_(self.position, std=0.02)
        self.encoder = _TimeSeriesEncoder(
            d_model,
            n_heads,
            num_layers,
            ff_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            norm=norm,
        )
        self.embedding_dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Dropout(head_dropout),
            nn.Linear(context * d_model, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        series, squeeze_channel = _channel_first(x, self.channels)
        if series.shape[-1] != self.context:
            raise ValueError(
                f"expected context length {self.context}, observed {series.shape[-1]}"
            )
        if self.revin is not None:
            series, mean, scale = self.revin.normalize(series)
        else:
            mean = scale = None
        batch, channels, _ = series.shape
        tokens = series.reshape(batch * channels, self.context, 1)
        tokens = self.embedding_dropout(self.input_projection(tokens) + self.position)
        encoded = self.encoder(tokens)
        forecast = self.head(encoded).reshape(batch, channels, self.horizon)
        if self.revin is not None:
            assert mean is not None and scale is not None
            forecast = self.revin.denormalize(forecast, mean, scale)
        return _restore_output_shape(forecast, squeeze_channel)


class PatchTSTForecaster(nn.Module):
    """Supervised PatchTST-style forecaster.

    The implementation follows the paper's two central choices: patch tokens and
    channel independence. Each channel is normalized separately, padded with its
    final value, divided into overlapping patches, processed by a shared
    Transformer encoder, and decoded by a flattening head.
    """

    def __init__(
        self,
        context: int,
        horizon: int,
        *,
        channels: int = 1,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 64,
        n_heads: int = 4,
        num_layers: int = 3,
        ff_dim: int = 128,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        head_dropout: float = 0.0,
        norm: str = "batch",
        padding_patch: bool = True,
        revin: bool = True,
        revin_affine: bool = True,
        individual_head: bool = False,
    ) -> None:
        super().__init__()
        _validate_transformer_dimensions(
            context=context,
            horizon=horizon,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
        )
        if channels <= 0:
            raise ValueError("channels must be positive")
        if patch_len <= 0 or stride <= 0:
            raise ValueError("patch_len and stride must be positive")
        if patch_len > context:
            raise ValueError("patch_len cannot exceed context")
        if not 0.0 <= head_dropout < 1.0:
            raise ValueError("head_dropout must lie in [0, 1)")
        self.context = int(context)
        self.horizon = int(horizon)
        self.channels = int(channels)
        self.patch_len = int(patch_len)
        self.stride = int(stride)
        self.padding_patch = bool(padding_patch)
        padded_context = context + stride if padding_patch else context
        self.num_patches = (padded_context - patch_len) // stride + 1
        if self.num_patches <= 0:
            raise ValueError("patch configuration produces no tokens")
        self.revin = (
            ReversibleInstanceNorm(channels, affine=revin_affine) if revin else None
        )
        self.patch_projection = nn.Linear(patch_len, d_model)
        self.position = nn.Parameter(torch.empty(1, self.num_patches, d_model))
        nn.init.trunc_normal_(self.position, std=0.02)
        self.embedding_dropout = nn.Dropout(dropout)
        self.encoder = _TimeSeriesEncoder(
            d_model,
            n_heads,
            num_layers,
            ff_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            norm=norm,
        )
        head_features = self.num_patches * d_model
        self.individual_head = bool(individual_head)
        if self.individual_head:
            self.heads = nn.ModuleList(
                nn.Sequential(
                    nn.Dropout(head_dropout),
                    nn.Linear(head_features, horizon),
                )
                for _ in range(channels)
            )
            self.head = None
        else:
            self.head = nn.Sequential(
                nn.Dropout(head_dropout),
                nn.Linear(head_features, horizon),
            )
            self.heads = nn.ModuleList()

    def _patch(self, x: torch.Tensor) -> torch.Tensor:
        if self.padding_patch:
            x = F.pad(x, (0, self.stride), mode="replicate")
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        if patches.shape[-2] != self.num_patches:
            raise RuntimeError(
                f"expected {self.num_patches} patches, observed {patches.shape[-2]}"
            )
        return patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        series, squeeze_channel = _channel_first(x, self.channels)
        if series.shape[-1] != self.context:
            raise ValueError(
                f"expected context length {self.context}, observed {series.shape[-1]}"
            )
        if self.revin is not None:
            series, mean, scale = self.revin.normalize(series)
        else:
            mean = scale = None
        batch, channels, _ = series.shape
        patches = self._patch(series)
        tokens = patches.reshape(batch * channels, self.num_patches, self.patch_len)
        tokens = self.embedding_dropout(self.patch_projection(tokens) + self.position)
        encoded = self.encoder(tokens).reshape(batch, channels, -1)
        if self.individual_head:
            forecast = torch.stack(
                [head(encoded[:, index]) for index, head in enumerate(self.heads)],
                dim=1,
            )
        else:
            assert self.head is not None
            forecast = self.head(encoded)
        if self.revin is not None:
            assert mean is not None and scale is not None
            forecast = self.revin.denormalize(forecast, mean, scale)
        return _restore_output_shape(forecast, squeeze_channel)


class ITransformerForecaster(nn.Module):
    """Variate-token Transformer following the iTransformer inversion principle.

    This model is most meaningful for multivariate inputs with ``channels > 1``;
    for a single channel the attention sequence contains only one variate token.
    """

    def __init__(
        self,
        context: int,
        horizon: int,
        *,
        channels: int,
        d_model: int = 64,
        n_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 128,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        head_dropout: float = 0.0,
        norm: str = "layer",
        revin: bool = True,
        revin_affine: bool = True,
    ) -> None:
        super().__init__()
        _validate_transformer_dimensions(
            context=context,
            horizon=horizon,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
        )
        if channels <= 0:
            raise ValueError("channels must be positive")
        if not 0.0 <= head_dropout < 1.0:
            raise ValueError("head_dropout must lie in [0, 1)")
        self.context = int(context)
        self.horizon = int(horizon)
        self.channels = int(channels)
        self.revin = (
            ReversibleInstanceNorm(channels, affine=revin_affine) if revin else None
        )
        self.temporal_embedding = nn.Linear(context, d_model)
        self.channel_embedding = nn.Parameter(torch.empty(1, channels, d_model))
        nn.init.trunc_normal_(self.channel_embedding, std=0.02)
        self.embedding_dropout = nn.Dropout(dropout)
        self.encoder = _TimeSeriesEncoder(
            d_model,
            n_heads,
            num_layers,
            ff_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            norm=norm,
        )
        self.head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(d_model, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        series, squeeze_channel = _channel_first(x, self.channels)
        if series.shape[-1] != self.context:
            raise ValueError(
                f"expected context length {self.context}, observed {series.shape[-1]}"
            )
        if self.revin is not None:
            series, mean, scale = self.revin.normalize(series)
        else:
            mean = scale = None
        tokens = self.embedding_dropout(
            self.temporal_embedding(series) + self.channel_embedding
        )
        encoded = self.encoder(tokens)
        forecast = self.head(encoded)
        if self.revin is not None:
            assert mean is not None and scale is not None
            forecast = self.revin.denormalize(forecast, mean, scale)
        return _restore_output_shape(forecast, squeeze_channel)
