# Quantitative Stability of the Known-Target Linear Gradient Oracle

## 1. Motivation

Exact fibre results answer a binary question: do two batches induce precisely the
same full gradient oracle? Real systems add finite precision, clipping, and
noise, so a useful theory also needs a quantitative statement for *nearby*
oracles.

This document specializes the biased linear model with mean half-squared loss.
The targets `Y` are known and common to the two candidate batches. A parameter
query consists of weights `Theta` and bias `b`, and returns the complete weight
and bias gradients.

## 2. Quotient statistics

For batch size `m`, define

\[
G=X^\top X,\qquad s=X^\top\mathbf 1,\qquad C=Y^\top X.
\]

The complete gradient oracle is

\[
\nabla_\Theta L_X(\Theta,b)
=\frac1m\left(\Theta G+b s^\top-C\right),
\]

\[
\nabla_b L_X(\Theta,b)
=\frac1m\left(\Theta s+mb-Y^\top\mathbf 1\right).
\]

For two input batches with common targets, write

\[
\Delta G=G_0-G_1,\qquad
\Delta s=s_0-s_1,\qquad
\Delta C=C_0-C_1.
\]

Exact observation equivalence is the special case in which all three
differences vanish.

## 3. Uniform one-query stability

Assume the attacker may query only a declared bounded parameter domain

\[
\|\Theta\|_2\le R_\Theta,
\qquad
\|b\|_2\le R_b.
\]

### Theorem 1 — bounded transcript-mean separation

For every allowed parameter query,

\[
\|\Delta\nabla_\Theta L\|_F
\le
\frac{
R_\Theta\|\Delta G\|_F
+R_b\|\Delta s\|_2
+\|\Delta C\|_F}{m},
\]

and

\[
\|\Delta\nabla_b L\|_2
\le
\frac{R_\Theta\|\Delta s\|_2}{m}.
\]

Consequently the Euclidean distance between the vectorized full-gradient means
is at most

\[
B=\sqrt{B_\Theta^2+B_b^2},
\]

where the two terms are the right-hand sides above.

#### Proof

Subtract the two affine gradient formulas. For the weight gradient,

\[
\Delta\nabla_\Theta L
=\frac1m(\Theta\Delta G+b\Delta s^\top-\Delta C).
\]

Apply the triangle inequality,
`||Theta Delta G||_F <= ||Theta||_2 ||Delta G||_F`, and
`||b Delta s^T||_F=||b||_2||Delta s||_2`. The bias result follows from

\[
\Delta\nabla_b L=\Theta\Delta s/m.
\]

Combining the two orthogonal transcript blocks gives the Euclidean bound.
\(\square\)

The executable implementation is
`uniform_gradient_query_difference_bound`. Regression tests draw many random
queries inside the declared spectral/L2 balls and independently verify that the
actual gradient distance never exceeds the certificate.

## 4. Gaussian-noisy observation channel

Suppose each classical query response is the vectorized full gradient plus
independent isotropic Gaussian noise

\[
Z\sim\mathcal N(0,\sigma^2 I).
\]

For one fixed query whose two means differ by Euclidean distance `d`, equal-prior
binary discrimination has exact success

\[
P_{\mathrm{one}}^*=\Phi\!\left(\frac d{2\sigma}\right),
\]

implemented by `equal_covariance_gaussian_binary_success`.

The attacker may adapt future parameter queries to every previous noisy answer.
The uniform bound from Theorem 1 remains valid conditionally on any history.

### Theorem 2 — arbitrary adaptive transcript upper bound

If every allowed conditional query has mean separation at most `B`, then after
`q` adaptive Gaussian-noisy queries,

\[
P_{\mathrm{succ}}^*
\le
\min\left\{1,
\frac12+\frac{\sqrt q\,B}{4\sigma}
\right\}.
\]

#### Proof

For equal-covariance Gaussian responses, the conditional KL divergence of one
query is at most

\[
B^2/(2\sigma^2).
\]

The chain rule for KL divergence applies to an adaptive transcript because the
query selected after a history is a deterministic or randomized function of
that history, and the same uniform conditional bound holds for every such
choice. Thus

\[
D_{\mathrm{KL}}(P_0^{(q)}\|P_1^{(q)})
\le qB^2/(2\sigma^2).
\]

Pinsker's inequality gives

\[
\operatorname{TV}(P_0^{(q)},P_1^{(q)})
\le \sqrt q B/(2\sigma).
\]

Equal-prior Bayes success is `(1+TV)/2`, yielding the result. \(\square\)

This is an information upper bound for every classical adaptive strategy, not a
performance claim about a particular reconstruction optimizer.

## 5. Necessary query count

For a desired binary success `p>1/2`, Theorem 2 implies the necessary condition

\[
q\ge
\left(
\frac{4\sigma(p-1/2)}{B}
\right)^2.
\]

`necessary_gaussian_queries_for_binary_success` returns the smallest integer
consistent with this condition. If `B=0`, all noisy response distributions are
identical for every adaptive strategy, so no number of queries can exceed random
guessing; the implementation returns `None` for every target above one half.

The bound is necessary, not sufficient. Meeting the count does not construct a
successful attack.

## 6. Interpretation for quantized and defended gradients

The statistic distance

\[
(\|\Delta G\|_F,\|\Delta s\|_2,\|\Delta C\|_F)
\]

is the natural quotient-space metric for this oracle family. It separates three
phenomena:

1. **exact fibre:** all components are zero, so unlimited exact classical or
   coherent access to an encoding of this same function cannot identify the
   original orbit member;
2. **near fibre:** the components are small, producing a controlled transcript
   separation on bounded query domains;
3. **far quotient classes:** the statistics differ enough that noise may no
   longer hide the candidates.

Clipping or quantization can be analyzed only after their exact channel semantics
are declared. A small pre-quantization difference alone does not guarantee equal
quantized words near bin boundaries. Experiments must compute quantization-margin
certificates or evaluate the induced discrete channel directly rather than
silently applying the Gaussian theorem.

## 7. Quantum-access boundary

Theorem 2 concerns a classical Gaussian response channel. It must not be applied
to a clean coherent unitary. For a noisy coherent implementation, the correct
object is an operational channel distance, including arbitrary reference
systems, followed by a hybrid argument over oracle calls.

The exact fibre result remains stronger: if two candidates compile to the same
unitary oracle, no quantum algorithm can distinguish them. For near-equal
unitaries, Q-RECON's operational-error bound must be instantiated separately.

## 8. Publication role

This stability result strengthens the identifiability thesis in three ways:

- it turns exact collisions into a continuous quotient-space geometry;
- it applies to arbitrary adaptive classical query policies;
- it yields explicit noise/query phase diagrams independent of optimization
  quality.

A top-tier evaluation should pair it with calibrated noise levels, bounded
parameter-query policies, empirical statistic-distance distributions, and exact
fibre/equivalence reporting on the final task.
