# Packed Classical Probes for Known-Target Gradient Oracles

## 1. Strengthened query bound

For a biased linear model with input dimension \(d\), output dimension \(c\),
fixed known targets, and mean half-squared loss, the complete full-gradient oracle
can be recovered with

\[
1+\left\lceil\frac d c\right\rceil
\]

deterministic classical parameter queries.

This improves the coordinate-at-a-time \(d+1\) construction by using the \(c\)
output rows as parallel probe channels. It is an explicit classical baseline that
must be used before comparing the channel with a coherent-query algorithm.

## 2. Probe schedule

Let

\[
S=X^\top X,\qquad m=X^\top\mathbf1,\qquad R=Y^\top X.
\]

The zero query \((\Theta,b)=(0,0)\) returns

\[
\nabla_\Theta L(0,0)=-R/B.
\]

For round \(r\), assign output row \(j\) the input basis vector

\[
\Theta^{(r)}_{j,:}=e_{rc+j}^\top
\]

whenever \(rc+j<d\); unused output rows remain zero. With \(b=0\), subtracting
the zero-query response gives

\[
B\left(
\nabla_\Theta L(\Theta^{(r)},0)
-
\nabla_\Theta L(0,0)
\right)_{j,:}
=e_{rc+j}^\top S,
\]

and

\[
B\left(
\nabla_b L(\Theta^{(r)},0)
-
\nabla_b L(0,0)
\right)_j
=m_{rc+j}.
\]

Each round therefore recovers up to \(c\) rows of \(S\) and the matching
coordinates of \(m\). After \(\lceil d/c\rceil\) rounds, \((S,m,R)\) is
complete and every future oracle response can be evaluated offline.

## 3. Consequence

A baseline that searches over parameter queries or repeatedly calls the gradient
oracle without exploiting this affine structure is not competitive. The relevant
classical query upper bound is independent of batch size and equals
\(1+\lceil d/c\rceil\), followed by polynomial-time matrix arithmetic.

This result does not resolve the target-stabilizer fibre. It learns the complete
oracle, and therefore the exact quotient information, but every batch in the same
fibre remains indistinguishable even under coherent access.

## 4. Executable implementation

`qrecon.theory.known_target_packed_probes` provides:

- `build_packed_linear_gradient_probe_plan`, which emits the auditable parameter
  matrices;
- `recover_linear_gradient_oracle_statistics_packed`, which executes the plan,
  checks Gram symmetry, reconstructs the sufficient statistics, and verifies the
  complete probe transcript.

Tests cover basis coverage for \(d=7,c=3\), exact emulation on unseen random
queries, the two-query regime \(c\ge d\), and rejection of inconsistent oracle
callbacks.

## 5. Claim boundary

The bound assumes exact full gradients, a biased linear model, mean half-squared
loss, public fixed targets, and freely chosen real-valued parameter queries. It is
an explicit upper bound, not a general query-minimality theorem for all possible
encodings or noisy access models.
