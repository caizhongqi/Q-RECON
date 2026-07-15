# Modern Forecasting Gradient-Release and Defense Protocol

## 1. Purpose

Modern forecasting reconstruction should not be evaluated only under exact full
floating-point gradients. Q-RECON defines an auditable release channel covering:

- global L2 clipping;
- additive Gaussian noise;
- symmetric signed quantization with a public scale;
- partial visibility of parameter tensors.

All operations, seeds, scales and visible parameter names are emitted in the
machine-readable report. Failed attacks remain in the denominator.

This protocol studies classical white-box leakage robustness. It does not make a
coherent quantum oracle available for free.

## 2. Fixed release order

Given exact per-batch gradients

\[
g=(g_1,\ldots,g_P),
\]

the declared release map applies operations in the following fixed order.

### 2.1 Global clipping

For threshold `C`,

\[
\widetilde g_i
=\alpha g_i,
\qquad
\alpha=\min\left\{1,
\frac{C}{\sqrt{\sum_i\|g_i\|_2^2}+\varepsilon}
\right\}.
\]

The report records the raw norm, clipped norm, threshold and scaling factor.
Candidate gradients pass through the same public clipping operation during
inversion.

### 2.2 Gaussian noise

Independent noise is added to every tensor:

\[
\widehat g_i=\widetilde g_i+\mathcal N(0,\sigma^2I).
\]

The noise seed is retained only so an artifact can be reproduced. The attack does
not receive or replay the private realization. It optimizes against the noisy
released values.

### 2.3 Symmetric signed quantization

For `b` bits, let

\[
q_{\max}=2^{b-1}-1.
\]

With public scale `s`, each value is released as

\[
Q_s(v)=s\,\operatorname{clip}
\left(\operatorname{round}(v/s),-q_{\max},q_{\max}\right).
\]

If the manifest does not predeclare a scale, the release derives one global scale
from the post-noise maximum absolute value and publishes it. Saturation count and
rate are reported. The attack uses a declared straight-through estimator only as
an optimization surrogate; the forward value is the exact quantized release.

### 2.4 Parameter visibility

Finally, the channel selects a declared subset of parameter tensors. The report
contains both ordered indices and exact parameter names. The standard matrix
includes a `last_head_only` setting that releases only the weight and bias of the
final biased Linear.

## 3. Paired defense matrix

`standard_modern_gradient_defenses()` declares:

1. `full_exact` — exact complete gradients;
2. `global_clip_1` — complete gradients clipped to global norm one;
3. `symmetric_int8` — complete gradients with global-scale signed 8-bit
   quantization;
4. `gaussian_noise_1e-3` — complete gradients with standard deviation `1e-3`;
5. `last_head_only` — exact weight and bias gradients of the final Linear only.

Every condition uses the same dataset tensor, trained model, attack batches and
restart seeds. A noise realization is fixed once per `(condition, batch)` and
shared across attack restarts so restart comparisons are paired.

The numeric thresholds above are calibration defaults, not universal defenses.
Publication studies must sweep clipping norms, noise levels, bit widths and
visible-layer scopes around predeclared values.

## 4. Defense-aware attack

`ReleasedGradientInversionAttack` computes candidate exact gradients, applies the
known deterministic release map, selects the same visible tensors and compares
the result with the actual released observation. Gaussian noise is not subtracted
or regenerated.

A valid comparison therefore does not commit either of these errors:

- matching exact candidate gradients directly to clipped/quantized observations;
- supplying the noise realization to the attack while calling the setting noisy.

## 5. Modern-model calibration

The revision-pinned GIFT-Eval/PatchTST calibration is executable with:

```bash
python examples/gifteval_patchtst_defense_suite.py \
  > outputs/gifteval-patchtst-defense-suite.json
```

It reports MSE, relative L2 error, correlation, sMAPE, tolerance success, attack
completion, release norms, quantization saturation and duration for each paired
condition.

The CI version is deliberately small. GitHub-hosted runner timing is correctness
and regression evidence, not publication-grade performance evidence.

## 6. Interaction with final-head representation leakage

For an exact batch-one, one-effective-sample release containing the final Linear
weight and bias gradients, the final hidden representation is algebraically
recoverable when the bias gradient is nonzero. Clipping by a common nonzero scalar
does not remove this ratio leakage because the scale cancels. Exact full-tensor
noise or quantization generally perturbs it, while omitting either final-head
weight or bias destroys the direct ratio.

Therefore `last_head_only` is not automatically a privacy defense. It may retain
the strongest analytic leakage while removing gradients that help generic
full-gradient matching. Q-RECON reports both the release-aware generic attack and
the dedicated head-representation baseline.

## 7. Required publication sweeps

For every modern victim and dataset, a strict study should include:

- batch sizes 1, 2, 4 and 8 where computationally feasible;
- multiple clipping thresholds spanning inactive to severe clipping;
- at least 4, 8 and 16-bit quantization;
- multiple noise levels normalized to the released gradient norm;
- full, final-head, encoder-only and selected-layer visibility;
- DLG-L2, InvG-cosine, hybrid, temporal-prior and analytic representation
  baselines where their assumptions hold;
- exact/equivalence-class success and perceptual/forecasting metrics;
- paired independent seeds, failures, confidence intervals and fixed-hardware
  timing.

## 8. Claim boundary

A lower reconstruction score under one defense setting is not a privacy theorem.
The paper must distinguish:

- empirical attack failure;
- local or global non-identifiability;
- loss of analytic decoder assumptions;
- optimization failure under a still-identifiable channel;
- increased classical or quantum cost.

Similarly, a defense that defeats one gradient matcher does not rule out the
analytic final-head representation attack or a structure-aware solver.
