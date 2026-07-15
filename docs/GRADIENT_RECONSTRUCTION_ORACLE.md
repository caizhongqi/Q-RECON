# Exact Training-Gradient Reconstruction Oracle

## 1. Reconstruction task

This artifact connects the coherent compiler to an actual training-data leakage
objective rather than a classifier-output toy. Consider one private integer
record \((x,t)\), a released biased linear model

\[
z=w^\top x+b,
\]

and squared loss

\[
L(x,t)=\frac12(z-t)^2.
\]

Let

\[
r=z-t.
\]

The exact per-record gradient observation is

\[
g_w=rx,
\qquad
g_b=r.
\]

The candidate register contains both the private feature vector and private
target. A public observation word concatenates all signed fixed-width components
of \((g_w,g_b)\). The verifier marks exactly the candidates whose complete
packed gradient equals the released word.

## 2. Identifiability dichotomy

### Theorem 1 — analytic recovery for nonzero bias gradient

If \(g_b\ne0\) and the exact integer gradient is internally consistent, then the
private record is unique and can be recovered by

\[
x_i=\frac{(g_w)_i}{g_b},
\qquad
t=w^\top x+b-g_b.
\]

### Proof

The bias gradient equals the residual: \(g_b=r\). Every weight-gradient component
satisfies \((g_w)_i=rx_i\). Division by the nonzero residual therefore determines
each feature exactly. Substitution into \(r=w^\top x+b-t\) determines the target.
Any second candidate with the same full gradient has the same residual, features,
and target. \(\square\)

The implementation rejects inconsistent observations when a weight-gradient
component is not exactly divisible by \(g_b\).

### Theorem 2 — zero-residual collision fibre

If \(g_b=0\), then \(g_w=0\) for every input. All representable records satisfying

\[
t=w^\top x+b
\]

produce the identical all-zero gradient observation. Unless the candidate prior
contains only one such record modulo the declared target equivalence, exact
original-record recovery is information-theoretically impossible.

### Corollary — no useful Grover regime for this leakage

For a full exact single-record gradient of a biased linear squared-loss model:

1. when \(g_b\ne0\), a classical decoder recovers the record in \(O(d)\) arithmetic
   operations, so unstructured quantum search is dominated by direct inversion;
2. when \(g_b=0\), the observation has a collision fibre and no classical or
   quantum post-processing can identify the original record beyond the Bayes
   ceiling.

Thus this leakage model contains no regime in which Grover search establishes a
meaningful end-to-end reconstruction advantage. Its value is instead as a
compiler correctness benchmark and as a negative theorem guiding the project
toward aggregated, partial, noisy, or nonlinear observations.

## 3. Structure-preserving value oracle

The gate-level backend implements

\[
U_g|x,t\rangle|y\rangle|0^a\rangle
=
|x,t\rangle|y\oplus g(x,t)\rangle|0^a\rangle
\]

using only X, CNOT, and Toffoli gates:

1. a clean affine value oracle computes
   \(r=w^\top x+b-t\) into a signed residual register;
2. the residual is copied into the output word for \(g_b\);
3. for every feature, a reversible variable-by-variable multiplier computes
   \(rx_i\) into a reusable accumulator;
4. the product is copied into the corresponding weight-gradient output word and
   the multiplier is reversed;
5. the residual oracle is reversed.

All output words remain, while the residual, product accumulator, scratch, carry,
and affine work return to zero.

## 4. Signed modular multiplication theorem

Let \(a\) and \(r\) be two's-complement words and let the product width be \(q\).
Their bit patterns are elements of the ring \(\mathbb Z/2^q\mathbb Z\). The
compiler sign-extends the shorter multiplicand and, for every multiplier bit,
conditionally copies a shifted multiplicand into scratch, adds it with a clean
ripple-carry adder, and erases scratch.

### Theorem 3 — product correctness

On clean work registers, the emitted circuit maps

\[
|a\rangle|r\rangle|p\rangle|0\rangle
\mapsto
|a\rangle|r\rangle|p+ar\bmod2^q\rangle|0\rangle.
\]

### Proof

For multiplier bit \(j\), the controlled scratch word is
\(r_j(a2^j)\bmod2^q\). The clean adder accumulates that word and restores its
carry; reversing the controlled copy clears scratch. Summing over all multiplier
bits gives multiplication in \(\mathbb Z/2^q\mathbb Z\). Two's-complement encoding
is the canonical residue representation, so the modular result equals signed
multiplication modulo the word width. Under the interval no-overflow certificate,
the residue is exactly the intended mathematical integer product. \(\square\)

For product width \(q\), one forward multiplier call uses

\[
3q^2+q
\]

Toffolis and \(4q^2\) CNOTs under the repository's ripple-adder convention. A
clean product output requires compute, \(q\) copy CNOTs, and inverse compute, for

\[
6q^2+2q\text{ Toffolis},
\qquad
8q^2+q\text{ CNOTs}
\]

per feature.

## 5. Exact-gradient equality and phase oracle

Given a public released gradient word \(g^*\), the clean verifier computes the
value oracle, applies a full-word equality tree, and reverses the value oracle:

\[
U_v|x,t\rangle|z\rangle|0^a\rangle
=
|x,t\rangle|z\oplus\mathbf1[g(x,t)=g^*]\rangle|0^a\rangle.
\]

Preparing the target in \(|-\rangle\) yields the phase oracle used by the Grover
simulator. Unlike the earlier finite truth-table backend, the phase sign here is
produced by a polynomial-size arithmetic netlist containing the actual residual
and gradient products.

If there are \(d\) features, candidate word width \(b\), and gradient component
width \(q\), the value-oracle layout uses

\[
(d+1)b+(d+1)q+3q+1
\]

logical qubits: candidate words, gradient outputs, residual, shared product
accumulator, scratch, and one carry helper. The equality verifier additionally
stores the value word and an equality tree before Bennett cleanup. Machine-readable
reports give exact emitted X/CNOT/Toffoli counts, logical depth, T-count and
T-depth bounds.

## 6. Global fibre and search evaluation

The same public reference channel is also exhaustively enumerated on small
candidate spaces. For every released gradient, the benchmark reports:

- the complete observation fibre;
- global injectivity and fibre-size histogram;
- uniform-prior Bayes exact-recovery ceiling;
- analytic inversion outcome;
- matched classical and Grover query counts;
- success of Grover simulation driven by the arithmetic phase circuit;
- full fault-tolerant logical-resource estimates.

Regression tests include both sides of the dichotomy: a unique nonzero-residual
record and the multi-record zero-gradient fibre.

## 7. Research consequence

This result closes the first end-to-end Q-RECON loop:

\[
\text{training leakage}
\rightarrow
\text{global identifiability analysis}
\rightarrow
\text{clean arithmetic value/verifier/phase oracle}
\rightarrow
\text{Grover execution and resource accounting}.
\]

It also demonstrates why query acceleration alone is not a publishable advantage
claim: the same task has a linear-time analytic classical solution whenever it is
identifiable. The next advantage search must therefore use a leakage map that is
both identifiable on the declared candidate space and not already algebraically
invertible—such as structured aggregate gradients, partial nonlinear-network
observations, or a constrained latent candidate verifier—with the strongest
classical solver measured under the same success criterion and cost unit.
