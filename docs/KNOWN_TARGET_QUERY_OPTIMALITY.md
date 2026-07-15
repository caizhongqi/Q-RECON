# Exact Query Complexity of Known-Target Linear Gradient Oracles

## 1. Main theorem

Consider exact full-gradient access to biased linear regression with input
dimension \(d\), output dimension \(c\), fixed known targets
\(Y\in\mathbb R^{B\times c}\), and mean half-squared loss. Assume

\[
\operatorname{rank}[\mathbf1,Y]=c+1,
\qquad B\ge c+2.
\]

The exact worst-case deterministic number of parameter queries required to learn
the complete gradient-oracle function is

\[
Q^*(d,c)=1+\left\lceil\frac dc\right\rceil.
\]

The same lower bound holds for zero-error randomized algorithms by fixing their
randomness. The theorem is about exact noiseless real-valued query access; noisy,
quantized, partial, or restricted-parameter interfaces require separate bounds.

## 2. Upper bound

The packed probe construction in `KNOWN_TARGET_PACKED_PROBES.md` uses one zero
query to reveal \(R=Y^\top X\), followed by
\(\lceil d/c\rceil\) queries. In each later query, the \(c\) output rows carry
up to \(c\) distinct input basis vectors, recovering the matching rows of
\(S=X^\top X\) and coordinates of \(m=X^\top\mathbf1\). Therefore

\[
Q^*(d,c)\le1+\left\lceil\frac dc\right\rceil.
\]

## 3. Physical lower-bound construction

Let a deterministic adaptive algorithm issue \(q\) weight queries

\[
\Theta^{(1)},\ldots,\Theta^{(q)}\in\mathbb R^{c\times d}
\]

while interacting with the zero input batch \(X_0=0\). Query biases may be
arbitrary. Suppose

\[
q<1+\left\lceil\frac dc\right\rceil.
\]

Then \(c(q-1)<d\). Stack all within-output-row differences

\[
D=
\begin{bmatrix}
\Theta^{(2)}-\Theta^{(1)}\\
\vdots\\
\Theta^{(q)}-\Theta^{(1)}
\end{bmatrix}.
\]

There exists a unit vector \(v\ne0\) with \(Dv=0\). Hence every query has the
same image

\[
\Theta^{(r)}v=a.
\]

Because \([\mathbf1,Y]\) has full column rank, any vector
\([0,a]^\top\) has a sample-index preimage. Choose \(w\) satisfying

\[
\mathbf1^\top w=0,
\qquad Y^\top w=a.
\]

When \(a\ne0\), set

\[
u=\frac{w}{\|w\|^2}.
\]

Then

\[
\mathbf1^\top u=0,
\qquad Y^\top u=\|u\|^2a.
\]

When \(a=0\), the assumption \(B\ge c+2\) provides a nonzero vector
\(u\in\operatorname{span}([\mathbf1,Y])^\perp\), which satisfies the same
equality.

Define a nonzero rank-one alternative batch

\[
X_1=uv^\top.
\]

Its sufficient statistics relative to \(X_0=0\) are

\[
X_1^\top X_1=\|u\|^2vv^\top,
\qquad X_1^\top\mathbf1=0,
\qquad Y^\top X_1=\|u\|^2av^\top.
\]

For every supplied query,

\[
\Theta^{(r)}X_1^\top X_1-Y^\top X_1
=
\|u\|^2(\Theta^{(r)}v-a)v^\top=0,
\]

and the bias-gradient difference is also zero because
\(X_1^\top\mathbf1=0\). Thus \(X_0\) and \(X_1\) return exactly the same full
gradients for all \(q\) queries, irrespective of the query biases.

Since the transcript is identical, an adaptive algorithm makes the same later
queries in both worlds. It cannot identify the complete oracle after \(q\)
queries. Therefore

\[
Q^*(d,c)\ge1+\left\lceil\frac dc\right\rceil.
\]

Combining the upper and lower bounds proves equality. \(\square\)

## 4. Interpretation

This theorem separates two questions that are often conflated:

1. **Learning the oracle:** exactly possible in
   \(1+\lceil d/c\rceil\) classical queries.
2. **Recovering the original batch:** generally impossible beyond the
   target-stabilizer quotient, even after the entire oracle is known.

Consequently, a quantum algorithm cannot claim a query advantage by treating
parameter settings as an unstructured search space. The strongest classical
baseline first executes the optimal packed probe plan, reconstructs
\((X^\top X,X^\top\mathbf1,Y^\top X)\), and then works entirely offline.
Coherent access cannot recover the unresolved orbit member because all members
induce the same unitary oracle.

## 5. Executable lower-bound witness

`qrecon.theory.known_target_probe_optimality` provides:

- `exact_known_target_probe_query_count`;
- `construct_physical_probe_lower_bound_witness`, which accepts any
  subcritical collection of parameter queries and constructs the explicit
  nonzero rank-one batch that produces the same transcript as the zero batch;
- numerical certificates for target constraints, gradient equality, nullspace
  rank, and input displacement.

Tests cover generic random subcritical probes, arbitrary query biases, the
zero-shared-image branch, the exact count formula, and theorem-assumption
rejection.

## 6. Claim boundary

The lower bound assumes the real-valued batch domain contains the constructed
rank-one witness. A separate bounded or discrete domain can remove it; in that
case the domain intersection must be analyzed explicitly. The theorem also does
not apply unchanged to cross-entropy, nonlinear victims, incomplete gradients,
noise, clipping, quantized parameter queries, or a rank-deficient target
constraint matrix.
