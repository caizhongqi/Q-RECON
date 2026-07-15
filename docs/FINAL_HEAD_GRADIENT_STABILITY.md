# Final-Head Gradient Leakage: Clipping Invariance and Perturbation Bounds

## 1. Scope

Let the final victim layer be a biased Linear map

\[
y=Wz+b,
\]

and assume exactly one effective sample reaches the layer. For any scalar loss,
write the output derivative as \(u\). The final-head gradients are

\[
G=\nabla_W L=u z^\top,
\qquad
g=\nabla_b L=u.
\]

If \(g\neq0\), the hidden representation is recovered exactly by

\[
z=\frac{g^\top G}{\|g\|_2^2}.
\]

The theorem concerns the hidden representation presented to the final Linear.
Recovering the original time series from that representation is a separate
encoder-inversion problem.

## 2. Global clipping does not remove the ratio leakage

Suppose a release applies one common nonzero scale \(\alpha\) to every parameter
gradient, as in global L2 clipping:

\[
\widetilde G=\alpha G,
\qquad
\widetilde g=\alpha g,
\qquad
\alpha>0.
\]

Then

\[
\frac{\widetilde g^\top\widetilde G}
     {\|\widetilde g\|_2^2}
=
\frac{\alpha^2 g^\top G}{\alpha^2\|g\|_2^2}
=z.
\]

Therefore global clipping by a public or unknown common nonzero scalar does not
remove one-effective-sample final-head leakage. Clipping to exactly zero does
remove the ratio but also destroys the released update.

This result does not apply unchanged to coordinate-wise clipping, per-layer
scales that differ between the final weight and bias, or an aggregate of multiple
effective samples.

## 3. Deterministic perturbation identity

After common scaling, let the released final-head pair be

\[
\widetilde g=\alpha g+e,
\qquad
\widetilde G=\alpha G+E
               =\alpha g z^\top+E.
\]

The ratio estimator is

\[
\widehat z=
\frac{\widetilde g^\top\widetilde G}
     {\|\widetilde g\|_2^2}.
\]

Using \(\alpha g=\widetilde g-e\),

\[
\widehat z-z
=-\frac{\widetilde g^\top e}{\|\widetilde g\|_2^2}z
 +\frac{\widetilde g^\top E}{\|\widetilde g\|_2^2}.
\]

Consequently,

\[
\|\widehat z-z\|_2
\le
\frac{\|e\|_2}{\|\widetilde g\|_2}\|z\|_2
+
\frac{\|E\|_F}{\|\widetilde g\|_2}.
\]

Define

\[
a=\frac{\|e\|_2}{\|\widetilde g\|_2},
\qquad
b=\frac{\|E\|_F}{\|\widetilde g\|_2}.
\]

When \(a<1\), \(\|z\|_2\le\|\widehat z\|_2+
\|\widehat z-z\|_2\) gives the observable a posteriori certificate

\[
\boxed{
\|\widehat z-z\|_2
\le
\frac{a\|\widehat z\|_2+b}{1-a}
}.
\]

If \(a\ge1\), this particular certificate is vacuous; that does not prove the
representation is unrecoverable.

## 4. Quantization bound

For round-to-nearest uniform quantization with step \(\Delta\), no saturation,
output dimension \(c\), and feature dimension \(d\), every scalar error is at
most \(\Delta/2\). Hence

\[
\|e\|_2\le\frac{\Delta}{2}\sqrt c,
\qquad
\|E\|_F\le\frac{\Delta}{2}\sqrt{cd}.
\]

Saturation invalidates these bounds and must be reported separately.

## 5. Gaussian bound

Assume independent Gaussian release noise with standard deviation \(\sigma\)
for the final bias and weight tensors. For a standard Gaussian vector of
dimension \(n\),

\[
\Pr\left[
\|N\|_2
\le
\sigma\left(\sqrt n+\sqrt{2\log(1/\eta)}\right)
\right]
\ge1-\eta.
\]

Splitting total failure probability \(\delta\) equally between the bias vector
and flattened weight matrix gives, with probability at least \(1-\delta\),

\[
\|e\|_2
\le
\sigma\left(\sqrt c+\sqrt{2\log(2/\delta)}\right),
\]

\[
\|E\|_F
\le
\sigma\left(\sqrt{cd}+\sqrt{2\log(2/\delta)}\right).
\]

Quantization and noise bounds can be combined by the triangle inequality; their
failure probabilities are combined by a union bound.

## 6. Consequences for PatchTST defenses

For a univariate, batch-one PatchTST with a shared final biased Linear, the final
head has one effective sample. Therefore:

- releasing both final weight and bias gradients exposes the head input exactly
  under exact arithmetic;
- global L2 clipping does not remove that exposure;
- `last_head_only` is not automatically a privacy defense;
- noise and quantization should be evaluated against both generic gradient
  matching and the dedicated ratio/representation attack;
- exact representation recovery does not imply exact reconstruction of the
  original time series because the Transformer encoder may be non-injective or
  hard to invert.

For multivariate channel-independent PatchTST, a shared head generally receives
`batch * channels` effective samples. The rank-one decoder must then be rejected
unless the head is channel-specific or the release separates contributions.

## 7. Executable mapping

`qrecon.theory.head_gradient_stability` provides:

- `recover_head_representation`;
- `common_scale_invariance_error`;
- `uniform_quantization_head_bounds`;
- `gaussian_head_bounds`;
- `combine_head_perturbation_bounds`;
- `certify_head_representation_perturbation`.

The tests verify exact scaling invariance, deterministic certificate soundness,
non-certifiable large bias perturbations, quantization geometry, Gaussian
concentration geometry, and combined bounds.

## 8. Claim boundary

The result is a leakage/stability theorem, not an end-to-end input-recovery
guarantee and not a quantum result. It should be paired with:

1. empirical representation-to-input inversion;
2. full-gradient and release-aware optimization baselines;
3. multiple effective-sample analysis;
4. clipping, quantization and normalized-noise sweeps;
5. a clear distinction between information loss and optimization failure.
