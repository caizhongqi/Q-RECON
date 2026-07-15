# Modern Transformer Forecasting Victims

## Scope

Q-RECON now exposes four forecasting victims through the experiment configuration:

| `victim.architecture` | Implementation | Token axis | Intended role |
|---|---|---|---|
| `mlp` | `ForecastMLP` | none | analytic/simple baseline |
| `transformer` | `TransformerForecaster` | time points | conventional temporal-token Transformer baseline |
| `patchtst` | `PatchTST` | temporal patches, independently per variate | long-lookback patch Transformer |
| `itransformer` | `ITransformer` | complete variates | multivariate correlation model |

PatchTST is based on *A Time Series is Worth 64 Words: Long-term Forecasting
with Transformers* (ICLR 2023, arXiv:2211.14730). Its defining choices are
patching and channel independence. iTransformer follows *iTransformer: Inverted
Transformers Are Effective for Time Series Forecasting* (ICLR 2024 Spotlight,
arXiv:2310.06625), which embeds a complete variable history as one token and
applies attention across variables.

PatchTST is an established modern baseline rather than literally the newest
architecture in 2026. Q-RECON includes it because it is widely recognized,
structurally distinct from ordinary temporal-token Transformers, and suitable for
studying whether patching changes gradient leakage and reconstruction fibres.

## Tensor contract

All modern victims accept either:

- univariate input `x: [batch, context]` and return `[batch, horizon]`; or
- multivariate input `x: [batch, context, channels]` and return
  `[batch, horizon, channels]`.

The historical `ForecastMLP` remains univariate-only. Multivariate use with the
MLP is rejected instead of silently flattening channels.

## PatchTST implementation contract

`PatchTST` provides:

- overlapping patches with configurable `patch_len` and `stride`;
- optional end replication padding matching the standard PatchTST patch-count
  convention;
- a shared patch projection and shared Transformer encoder across all variates;
- shared or per-channel forecasting heads;
- optional RevIN normalization and exact forecast denormalization;
- learned patch positional embeddings.

Channels are folded into the batch dimension before the encoder. With a shared
head, permuting input channels therefore permutes output channels identically;
this property is tested explicitly.

## iTransformer implementation contract

`ITransformer` maps each variable's complete lookback window to one token. The
self-attention axis is the variable axis, and a shared projection maps each
encoded variable token to the forecast horizon. No channel-specific positional
embedding is added, preserving variable permutation equivariance and arbitrary
variable ordering.

The current GIFT-Eval window loader is univariate. iTransformer runs on it, but a
single variable gives attention only one token and therefore does not exercise
its central cross-variate inductive bias. The
`synthetic_multivariate_forecasting` generator and
`configs/time_synthetic_multivariate_itransformer.yaml` provide a genuine
multivariate execution path. Publication experiments should add real
multivariate benchmarks such as ETT, Electricity, Weather or Traffic under a
revision-pinned data manifest.

## Gradient-inversion compatibility

The encoder uses explicit matrix-multiply/softmax attention rather than fused
FlashAttention kernels. This is deliberate: Q-RECON's gradient-matching attack
requires derivatives of parameter gradients with respect to candidate inputs.
The unfused path has a stable higher-order autograd contract on CPU and GPU,
whereas fused attention backends can differ in double-backward support.

Tests compute model-parameter gradients with `create_graph=True`, differentiate a
second objective back to the private input, and require finite results for the
Transformer, PatchTST and iTransformer victims.

## Configuration examples

```yaml
victim:
  architecture: patchtst
  patch_len: 16
  stride: 8
  padding_patch: true
  d_model: 64
  n_heads: 4
  e_layers: 3
  d_ff: 128
  dropout: 0.1
  revin: true
```

```yaml
victim:
  architecture: itransformer
  d_model: 64
  n_heads: 4
  e_layers: 2
  d_ff: 128
  dropout: 0.1
  revin: true
```

Runnable configurations:

```bash
qrecon --config configs/time_gifteval_transformer.yaml
qrecon --config configs/time_gifteval_patchtst.yaml
qrecon --config configs/time_synthetic_multivariate_itransformer.yaml
```

The result report records the concrete victim class, architecture configuration
and trainable parameter count. This prevents a run from being described as a
PatchTST experiment when the factory actually instantiated the historical MLP.

## Claim boundary

Adding a modern victim does not establish attack quality or quantum advantage.
For each architecture the paper still needs:

1. a multi-sample, multi-seed reconstruction success matrix;
2. exact-record/equivalence-class and perceptual metrics;
3. architecture-matched classical attacks and solver baselines;
4. gradient clipping, aggregation, quantization and noise conditions;
5. a coherent compiler or an explicit statement that the experiment is only
   classical white-box gradient inversion;
6. end-to-end data-loading, oracle, measurement and fault-tolerant costs before
   any quantum-advantage claim.
