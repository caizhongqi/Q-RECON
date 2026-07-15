# Aggregate-Gradient Non-Identifiability for Linear Regression

## 1. Scope

This document gives a Q-RECON-specific collision theorem for complete aggregate
gradients. It moves beyond the batch-one identity used by the analytic recovery
baseline and proves that, for biased linear regression with mean squared loss,
a batch of size at least two generally belongs to a continuous family of
training batches with exactly the same released weight and bias gradients.

The result concerns reconstruction of the complete input-target batch when both
inputs and regression targets are private. It does not by itself prove the same
claim for fixed known targets, classification labels, nonlinear networks,
discrete candidate grids, or per-sample gradients.

## 2. Model and observation

Let

\[
X\in\mathbb R^{B\times d},
\qquad
Y\in\mathbb R^{B\times c},
\]

be a private batch. Consider the biased linear model

\[
f_{\Theta,b}(x)=\Theta x+b,
\]

with

\[
\Theta\in\mathbb R^{c\times d},
\qquad
b\in\mathbb R^c,
\]

and mean half-squared loss

\[
L(X,Y)=\frac{1}{2B}
\sum_{i=1}^{B}\|\Theta x_i+b-y_i\|_2^2.
\]

Write the residual matrix as

\[
\Delta=X\Theta^\top+\mathbf 1b^\top-Y
\in\mathbb R^{B\times c}.
\]

The released full aggregate gradient is

\[
G_\Theta=\nabla_\Theta L
=\frac1B\Delta^\top X,
\qquad
G_b=\nabla_b L
=\frac1B\Delta^\top\mathbf 1.
\]

## 3. Batch-mixing collision theorem

### Theorem 1

Let \(A\in\mathrm{GL}(B,\mathbb R)\) satisfy

\[
A\mathbf 1=\mathbf 1.
\]

Define

\[
X'=AX,
\qquad
\Delta'=A^{-\top}\Delta,
\]

and choose

\[
Y'=X'\Theta^\top+\mathbf 1b^\top-\Delta'.
\]

Then \((X',Y')\) produces exactly the same full aggregate weight and bias
gradients as \((X,Y)\):

\[
\nabla_\Theta L(X',Y')=\nabla_\Theta L(X,Y),
\]

\[
\nabla_b L(X',Y')=\nabla_b L(X,Y).
\]

### Proof

By construction, the residual matrix of \((X',Y')\) is \(\Delta'\). Therefore

\[
\begin{aligned}
B\nabla_\Theta L(X',Y')
&=(\Delta')^\top X'\\
&=\Delta^\top A^{-1}AX\\
&=\Delta^\top X\\
&=B\nabla_\Theta L(X,Y).
\end{aligned}
\]

Since \(A\mathbf 1=\mathbf 1\), invertibility gives
\(A^{-1}\mathbf 1=\mathbf 1\). Hence

\[
\begin{aligned}
B\nabla_b L(X',Y')
&=(\Delta')^\top\mathbf 1\\
&=\Delta^\top A^{-1}\mathbf 1\\
&=\Delta^\top\mathbf 1\\
&=B\nabla_b L(X,Y).
\end{aligned}
\]

Thus both released gradients are identical. \(\square\)

## 4. Continuous nontrivial collision family

For any two batch positions, embed the block

\[
A_\alpha=
\begin{pmatrix}
1-\alpha&\alpha\\
\alpha&1-\alpha
\end{pmatrix}
\]

into the identity matrix. It satisfies

\[
A_\alpha\mathbf 1=\mathbf 1
\]

and is invertible whenever \(\alpha\ne\tfrac12\). For
\(0<\alpha<1\), the transformed inputs are convex combinations of the selected
samples. Consequently, if the input domain is convex, the transformed inputs
remain in the domain.

Except when the selected samples and residual structure are invariant under the
mixing, varying \(\alpha\) gives infinitely many distinct batches with exactly
the same aggregate gradient. The executable implementation verifies the gradient
identity numerically to a caller-specified tolerance and reports the input and
target displacement.

## 5. Identifiability consequence

Under the observation map

\[
\mathcal O(X,Y)=(G_\Theta,G_b),
\]

Theorem 1 constructs an explicit orbit inside an observation fibre. Therefore,
without a prior or side information that selects one representative from this
orbit, the complete batch is not globally identifiable from the aggregate
gradient alone.

This result explains why the batch-one ratio

\[
x=\frac{\nabla_b^\top\nabla_\Theta}{\|\nabla_b\|_2^2}
\]

cannot be extended mechanically to aggregated batches. At batch size one there
is no nontrivial batch mixing group; at batch size at least two, the factorization
\(\Delta^\top X\) has a gauge symmetry constrained only by preservation of the
all-ones vector when the bias gradient is also visible.

## 6. Executable theorem witness

The implementation in `qrecon.theory.batch_collisions` provides:

- `linear_squared_loss_gradients` for the exact released observation;
- `validate_batch_mixing_matrix` for the condition
  \(A\mathbf 1=\mathbf 1\) and invertibility;
- `symmetric_pair_mixing` for a continuous explicit collision family;
- `construct_linear_batch_collision` for generating \((X',Y')\) and checking
  both gradient equalities.

Regression tests cover random multivariate batches, distinct values of
\(\alpha\), convex-domain preservation for the transformed inputs, and rejection
of invalid mixing matrices.

## 7. Claim boundary and next theorem

The theorem is exact but deliberately narrow. It establishes non-identifiability
of the joint input-target batch for biased linear regression under full aggregate
gradient leakage. The next theoretical extensions are:

1. characterize collisions when targets are fixed or known;
2. impose discrete or structured candidate domains and determine when the
   continuous orbit intersects them more than once;
3. extend the symmetry analysis to generalized linear models and softmax loss;
4. quantify how clipping, quantization, noise, and multiple training rounds alter
   the observation fibres;
5. derive Bayes recovery ceilings under explicit priors on the collision orbit.

No claim about arbitrary CNNs, classification batches, or federated updates is
made by this theorem alone.
