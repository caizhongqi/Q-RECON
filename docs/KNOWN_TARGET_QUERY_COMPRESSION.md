# Finite Query Compression and Quotient Reconstruction

## 1. Why this result matters

`KNOWN_TARGET_GRADIENT_ORACLE_FIBRES.md` proves that the full biased-linear MSE
gradient oracle with fixed known targets identifies an input batch only up to a
target-stabilizer orthogonal orbit. This document adds the complementary positive
and negative result:

1. the **entire infinite parameter-query oracle can be learned with only
   \(d+1\) classical queries**, where \(d\) is the input dimension; and
2. those recovered statistics reconstruct exactly the identifiable quotient of
   the batch, while making the unresolved orthogonal orbit explicit.

Therefore this access model does not support a meaningful unstructured-search
advantage. A classical attacker can compress the oracle in polynomially many
queries and emulate every future response offline, yet no classical or quantum
algorithm can select the original member of a nontrivial fibre from the oracle
alone.

## 2. Model and sufficient statistics

For

\[
f_{\Theta,b}(x)=\Theta x+b
\]

and

\[
L_{X,Y}(\Theta,b)=\frac1{2B}
\|X\Theta^\top+\mathbf1b^\top-Y\|_F^2,
\]

the gradient function is determined by

\[
S=X^\top X,\qquad m=X^\top\mathbf1,
\qquad R=Y^\top X,
\]

because

\[
B\nabla_\Theta L=\Theta S+b m^\top-R,
\]

\[
B\nabla_b L=\Theta m+B b-Y^\top\mathbf1.
\]

Targets \(Y\), batch size \(B\), input dimension \(d\), and output dimension
\(c\ge1\) are public in this theorem.

## 3. A \(d+1\)-query exact compression algorithm

### Theorem 1 — finite classical recovery of the full oracle

The complete statistics \((S,m,R)\), and hence every future gradient-oracle
response, can be recovered with \(d+1\) deterministic classical queries.

### Query 0

Set \(\Theta=0\) and \(b=0\). Then

\[
\nabla_\Theta L(0,0)=-\frac1B R,
\]

so

\[
R=-B\nabla_\Theta L(0,0).
\]

### Queries \(1,\ldots,d\)

For coordinate \(k\), let \(\Theta^{(k)}\) be zero except that the first output
row equals \(e_k^\top\), and set \(b=0\). Relative to Query 0,

\[
B\left(
\nabla_\Theta L(\Theta^{(k)},0)
-
\nabla_\Theta L(0,0)
\right)_{1,:}
=e_k^\top S,
\]

which recovers row \(k\) of \(S\). The first bias-gradient component gives

\[
B\left(
\nabla_b L(\Theta^{(k)},0)
-
\nabla_b L(0,0)
\right)_1
=m_k.
\]

After all \(d\) basis probes, \(S,m,R\) are known exactly. Substitution into the
gradient formulas simulates any later query without further access to \(X\).
\(\square\)

The theorem is an upper bound, not a claim that \(d+1\) is query-minimal. Its
importance is that the full continuous parameter oracle collapses to an explicit
polynomial classical transcript.

## 4. Exact quotient reconstruction

Let

\[
C=[\mathbf1,Y]
\]

and let \(P_C\) be the orthogonal projector onto
\(\operatorname{span}(C)\). From \(m\) and \(R\), the oracle identifies

\[
C^\top X=
\begin{bmatrix}
\mathbf1^\top X\\
Y^\top X
\end{bmatrix}.
\]

Hence it identifies the constrained component

\[
X_\parallel=P_CX
=C(C^\top C)^+
\begin{bmatrix}
m^\top\\R
\end{bmatrix}.
\]

It also identifies the target-orthogonal residual Gram matrix

\[
G_\perp
=(X-X_\parallel)^\top(X-X_\parallel)
=S-X_\parallel^\top X_\parallel.
\]

### Theorem 2 — complete quotient information

The pair

\[
(X_\parallel,G_\perp)
\]

is determined exactly by the oracle and known targets. Conversely, any matrix

\[
\widehat X=X_\parallel+U_\perp Z
\]

with \(U_\perp\) an orthonormal basis of
\(\operatorname{span}(C)^\perp\) and

\[
Z^\top Z=G_\perp
\]

has the same complete gradient oracle. Thus the pair is a complete coordinate
system for the quotient by the target-stabilizer orbit.

### Proof

The forward direction follows from the recovered sufficient statistics. For the
converse,

\[
C^\top\widehat X=C^\top X_\parallel=C^\top X
\]

and

\[
\widehat X^\top\widehat X
=X_\parallel^\top X_\parallel+Z^\top Z
=S.
\]

The complete-fibre theorem then gives identical gradient oracles. \(\square\)

The implementation constructs one deterministic real-valued representative by
factoring \(G_\perp\) and embedding it in a deterministic basis of the
orthogonal complement. This is a quotient representative, not evidence that the
original ordered batch was recovered.

## 5. Consequences for quantum claims

### Corollary 3 — no Grover-style advantage for learning this oracle

A classical algorithm obtains a complete emulator in \(d+1\) queries. Therefore
a query-complexity comparison that treats each possible parameter setting as an
unstructured candidate space is invalid for this channel: the affine structure
must be used by the classical baseline.

### Corollary 4 — quotient recovery does not resolve the fibre

After the statistics are learned, all remaining uncertainty is exactly the
orthogonal factor of the residual component. Coherent access does not remove it,
because every batch in the same fibre induces the same unitary oracle. The
correct recovery target is therefore either:

- the quotient \((X_\parallel,G_\perp)\);
- a declared equivalence class containing the whole allowed orbit intersection;
  or
- the original batch only under an additional domain theorem proving that the
  orbit intersects the candidate domain once.

## 6. Executable implementation

`qrecon.theory.known_target_quotient` provides:

- `recover_linear_gradient_oracle_statistics`: executes the \(d+1\)-query probe
  plan, checks Gram symmetry, and verifies that the recovered statistics
  reproduce the entire probe transcript;
- `known_target_orbit_invariants_from_statistics`: computes
  \(X_\parallel\) and \(G_\perp\);
- `construct_known_target_orbit_representative`: constructs and revalidates one
  representative of the exact fibre.

Tests additionally draw unseen random parameter queries and confirm that the
recovered emulator matches direct gradients, reject inconsistent callbacks, and
verify that the quotient representative preserves all full-oracle statistics.

## 7. Claim boundary

The result assumes exact, noiseless full gradients of biased linear regression
under mean half-squared loss and fixed known targets. Quantized, clipped, noisy,
partial, cross-entropy, nonlinear, and unknown-target channels require separate
probe and fibre analyses. The constructed real-valued representative may lie
outside a discrete or bounded application domain; domain intersection remains a
required final identifiability check.
