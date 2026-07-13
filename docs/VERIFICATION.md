# Verification record

Verified locally on macOS arm64 with Python 3.12, PyTorch 2.13.0 and
PennyLane 0.45.1.

## Unit tests

```text
2 passed
```

The tests cover gradient extraction, local Jacobian-rank analysis,
reconstruction metrics and quantum logical-resource estimates.

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

