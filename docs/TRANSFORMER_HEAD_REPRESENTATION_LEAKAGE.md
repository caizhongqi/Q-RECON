# Exact Final-Head Representation Leakage in Transformer Forecasters

## 1. Setting

Let a forecasting victim decompose as

\[
E_\phi:x\mapsto z\in\mathbb R^d,
\qquad
h_{W,b}(z)=Wz+b\in\mathbb R^c,
\]

where `E` is a Transformer, PatchTST or iTransformer encoder and the final
prediction head is a biased Linear layer. Let a scalar differentiable loss be

\[
L(x,t;\phi,W,b)=\ell(h_{W,b}(E_\phi(x)),t).
\]

The attacker observes exact gradients of the final head. The theorem below does
not require knowledge of the target, the loss derivative, or the preceding
encoder parameters.

## 2. Exact single-effective-sample theorem

### Theorem 1

Assume the final Linear is evaluated on exactly one effective sample and let

\[
\delta=\nabla_b L.
\]

If `delta` is nonzero, then the exact head input is

\[
z=
\frac{(\nabla_b L)^\top(\nabla_W L)}
     {\|\nabla_b L\|_2^2}.
\]

### Proof

For one effective sample, standard reverse-mode differentiation gives

\[
\nabla_W L=\delta z^\top,
\qquad
\nabla_b L=\delta.
\]

Left multiplying the weight gradient by `delta^T` yields

\[
\delta^\top\nabla_WL
=\|\delta\|_2^2 z^\top.
\]

Division by the nonzero squared norm recovers `z`. `□`

The executable implementation is
`recover_single_effective_head_input`. It also computes

\[
\frac{\|\nabla_W L-\delta z^\top\|_F}
     {\|\nabla_W L\|_F},
\]

which should be near zero under the theorem assumptions. This residual is a
consistency check; it does not prove that the effective-sample count is one.

## 3. What counts as an effective sample

PyTorch `Linear` sums parameter gradients over every leading dimension of its
input. Consequently:

- a univariate batch-one `TransformerForecaster` has one effective head sample;
- a univariate batch-one shared-head `PatchTST` has one effective sample;
- a univariate batch-one `ITransformer` has one variable token and one effective
  sample;
- a shared-head PatchTST with batch size `B` and `C` variables has `B*C`
  effective samples;
- a shared-head iTransformer likewise has `B*C` effective samples;
- an individual PatchTST head has `B` effective samples per channel-specific
  Linear.

For multiple effective samples,

\[
\nabla_WL=\sum_s\delta_s z_s^\top,
\qquad
\nabla_bL=\sum_s\delta_s,
\]

and the single-sample ratio generally does not separate the individual
representations. Q-RECON therefore rejects the exact decoder instead of silently
applying it to an aggregated shared head.

## 4. Representation leakage is not input recovery

The theorem reveals `z=E(x)`, not automatically `x`. Define the representation
fibre

\[
\mathcal F_E(z)=\{x':E(x')=z\}.
\]

No classical or quantum method can identify the original input beyond the prior
mass inside this fibre from the representation alone.

### Theorem 2 — local representation identifiability

Suppose `E` is continuously differentiable and its Jacobian

\[
J_E(x)=\frac{\partial E(x)}{\partial x}
\]

has full column rank. Then `E` is locally injective around `x`. Moreover, for

\[
R(x')=\|E(x')-E(x)\|_2^2,
\]

we have

\[
\nabla^2 R(x)=2J_E(x)^\top J_E(x),
\]

which is positive definite. Thus the true input is a strict local minimizer of
exact representation matching.

### Proof sketch

A full-column-rank Jacobian contains a nonsingular square row minor. Projecting
onto those output coordinates and applying the inverse-function theorem gives a
locally invertible projection, hence local injectivity of `E`. At zero residual,
all Hessian terms multiplied by `E(x')-E(x)` vanish, leaving
`2 J_E^T J_E`, which is positive definite under full column rank. `□`

`head_representation_jacobian_report` evaluates this local certificate. It is not
a global uniqueness result.

## 5. Strong classical baseline

`HeadRepresentationInversionAttack` performs:

1. exact analytic recovery of the final-head representation;
2. first-order optimization of a candidate input against that representation;
3. optional time-series smoothness regularization;
4. best-iterate selection using only representation loss, never private-reference
   error.

Unlike full gradient matching, this baseline does not differentiate through
parameter gradients. It is therefore materially cheaper and must be included
before comparing a quantum search method against generic DLG-style inversion.

## 6. Application to current victims

The theorem applies directly to the current revision-pinned GIFT-Eval
batch-one/univariate experiments for:

- `TransformerForecaster`;
- shared-head `PatchTST`;
- univariate `ITransformer`.

It does not justify exact individual representation recovery for multivariate
shared-head PatchTST/iTransformer or batch size greater than one. Those settings
need a mixture-identifiability theorem, channel-specific heads, additional
observations, or a different attack.

## 7. Claim boundary

Valid claim:

> Under one-effective-sample and nonzero-bias-gradient conditions, the final
> forecasting-head gradients exactly disclose the encoder representation; input
> recovery is governed by the representation fibre and local/global properties of
> the encoder.

Invalid claims:

- that a rank-one numerical residual alone proves one effective sample;
- that representation recovery implies globally unique raw-input recovery;
- that the theorem applies unchanged to a bias-free head;
- that it separates multiple records or variables sharing one Linear;
- that it establishes quantum advantage.
