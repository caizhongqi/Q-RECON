# Verification record

## Continuous integration — 2026-07-14

GitHub Actions run `29360446047` validated commit
`73bb6ee6fb2bc5927931dad2e4cf26f28c3e2871` on Ubuntu with Python 3.10 and 3.12.
The independent PennyLane job used Python 3.12.

The archived Python 3.12 JUnit report contains:

```text
60 tests
0 failures
0 errors
0 skipped
```

All three jobs completed successfully:

- unit tests on Python 3.10;
- unit tests on Python 3.12;
- differentiable PennyLane `QuantumPrior` forward/backward smoke test.

Both unit-test jobs also compiled every Python file and executed the public theory
and compiler examples. The workflow archived machine-readable reports for:

- information/query bounds;
- finite truth-table oracle synthesis;
- compiler scaling;
- structure-preserving Affine cost accounting;
- reversible MLP search;
- exact training-gradient reconstruction.

The test suite covers:

- deterministic/noisy Bayes recovery and target equivalence;
- data processing, conditional min-entropy, Helstrom and privacy bounds;
- local differential identifiability;
- aggregate-gradient collision constructions;
- fixed-point word semantics and range proofs;
- mixed-polarity minterm and ANF exact oracle synthesis;
- reversible ripple-carry addition and constant shift-add Affine circuits;
- exact equality comparators;
- two-layer and arbitrary-depth reversible ReLU MLPs;
- shared arithmetic-work liveness;
- signed variable-by-variable multiplication;
- finite and structure-preserving training-gradient value/equality/phase oracles;
- clean ancillas, inverse restoration and basis-permutation checks;
- Grover success curves driven by compiled phase netlists;
- end-to-end setup/query/fault-tolerant cost planning.

## Earlier local environment

The first reconstruction prototype was verified locally on macOS arm64 with
Python 3.12, PyTorch 2.13.0 and PennyLane 0.45.1.

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
inversion for two optimization steps. The CI smoke test additionally verifies a
fresh forward pass, scalar loss, backward pass and finite gradients for every
trainable parameter.

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
