# Verification record

Verified locally on macOS arm64 with Python 3.12, PyTorch 2.13.0 and
PennyLane 0.45.1.

## Unit tests

```text
4 passed
```

The tests cover gradient extraction, local Jacobian-rank analysis,
reconstruction metrics, analytic input recovery, label inference and quantum
logical-resource estimates.

## Synthetic end-to-end experiment

`configs/smoke.yaml` completed training, gradient leakage and reconstruction.
The gradient-matching component decreased from approximately `0.1264` to
`0.0033` in 20 attack steps. The local gradient Jacobian had numerical rank
`8/8` with condition number approximately `9.50`.

## Real dataset loading and reconstruction

- GIFT-Eval: loaded two normalized forecasting windows with shapes `(2, 16)`
  and `(2, 4)` using Hugging Face streaming.
- Community Forensics Small: loaded balanced real/fake samples and decoded them
  to image tensors.
- Tiny GIFT-Eval reconstruction: final gradient matching loss `0.00509`.
- Tiny Community Forensics reconstruction: final gradient matching loss
  `0.000589`.

These tiny runs validate the execution path, not paper-level attack quality.
Full experiments require multiple seeds, stronger classical baselines,
confidence intervals and end-to-end resource accounting.

## Quantum second-order path

A two-qubit, one-layer `QuantumPrior` completed differentiable gradient
inversion for two optimization steps. This validates second-order derivatives
through the VQC and victim-gradient matching objective.

## Exact batch-one linear-layer recovery

For a first biased linear layer and a single training sample,

```text
grad(W) = delta outer x
grad(b) = delta
```

so the raw input is recovered by a least-squares ratio across output rows.

### GIFT-Eval + ForecastMLP

- MSE: `4.06e-15`
- maximum absolute error: `2.38e-7`
- values within `1e-6`: `100%`
- bitwise floating-point equality: `58.33%`

### Community Forensics + ImageMLP

- MSE: `3.60e-16`
- maximum absolute error: `1.19e-7`
- pixels within `1e-6`: `100%`
- recovered 8-bit pixels equal to reference: `100%`
- automatically inferred class label: correct

Floating-point bitwise equality is lower because multiplication followed by
division can differ in the least significant bits. The recovered image becomes
identical after conversion back to its 8-bit representation.

## Convolutional victim

The shallow sigmoid LeNet-style victim with LBFGS inversion achieved:

- PSNR: `57.34 dB`
- maximum absolute error: `0.0040`
- pixels within `0.01`: `100%`
- recovered 8-bit pixels equal to reference: `86.46%`

The deeper three-convolution `TinyConvNet` remains substantially harder and is
not claimed as exact recovery. This architecture-dependent gap is an explicit
research result rather than being hidden behind gradient-matching loss.
