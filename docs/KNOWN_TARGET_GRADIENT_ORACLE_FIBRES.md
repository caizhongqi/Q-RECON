# Complete Fibres of Known-Target Linear Gradient Oracles

## 1. Result and scope

This document characterizes the complete global collision classes of a full
parameter-query gradient oracle for biased linear regression with fixed known
targets and mean half-squared loss. Unlike a single-gradient collision, the
result concerns equality of the **entire gradient function** at every model
parameter value. It therefore applies to unlimited adaptive classical queries
and to coherent reversible access to any finite encoding of the same function.

Let

\[
X\in\mathbb R^{B\times d},\qquad
Y\in\mathbb R^{B\times c}
\]

be an input batch and a fixed public target matrix. The model and loss are

\[
f_{\Theta,b}(x)=\Theta x+b,
\]

\[
L_{X,Y}(\Theta,b)=\frac{1}{2B}
\|X\Theta^\top+\mathbf 1b^\top-Y\|_F^2.
\]

The oracle returns the full pair

\[
\mathcal G_{X,Y}(\Theta,b)
=
\bigl(\nabla_\Theta L_{X,Y},\nabla_b L_{X,Y}\bigr).
\]

The theorem is specific to this declared linear/MSE channel. It does not
implicitly cover cross-entropy, nonlinear networks, unknown targets, clipped or
noisy gradients, or discrete candidate domains.

## 2. Sufficient-statistic theorem

Define

\[
S_X=X^\top X,\qquad
m_X=X^\top\mathbf 1,\qquad
R_{Y,X}=Y^\top X,\qquad
t_Y=Y^\top\mathbf 1.
\]

### Theorem 1 — complete gradient-oracle statistics

For every \((\Theta,b)\),

\[
B\nabla_\Theta L_{X,Y}
=
\Theta S_X+b m_X^\top-R_{Y,X},
\]

\[
B\nabla_b L_{X,Y}
=
\Theta m_X+B b-t_Y.
\]

Consequently, for a fixed nonempty target matrix \(Y\), two batches \(X,X'\)
induce exactly the same full gradient oracle if and only if

\[
X^\top X=X'^\top X',
\]

\[
X^\top\mathbf 1=X'^\top\mathbf 1,
\]

\[
Y^\top X=Y^\top X'.
\]

### Proof

Expanding the residual gives

\[
\begin{aligned}
B\nabla_\Theta L
&=(X\Theta^\top+\mathbf1b^\top-Y)^\top X\\
&=\Theta X^\top X+b\mathbf1^\top X-Y^\top X,
\end{aligned}
\]

and

\[
B\nabla_b L
=(X\Theta^\top+\mathbf1b^\top-Y)^\top\mathbf1
=\Theta X^\top\mathbf1+B b-Y^\top\mathbf1.
\]

Equality of the three displayed statistics is sufficient by substitution. It is
also necessary: equality for every \(\Theta\) identifies the coefficient
\(X^\top X\), equality for every \(b\) identifies \(X^\top\mathbf1\), and the
constant term identifies \(Y^\top X\). \(\square\)

This theorem replaces an infinite family of parameter queries by three finite
matrices. More queries cannot reveal anything outside these statistics.

## 3. Complete orthogonal-orbit characterization

Let

\[
C=[\mathbf1,Y]\in\mathbb R^{B\times(c+1)}
\]

and define the target stabilizer

\[
\operatorname{Stab}(C)
=
\{Q\in O(B):QC=C\}.
\]

### Theorem 2 — exact fibre equals a target-stabilizer orbit

For fixed \(Y\),

\[
\mathcal G_{X,Y}=\mathcal G_{X',Y}
\quad\text{as functions of }(\Theta,b)
\]

if and only if there exists \(Q\in\operatorname{Stab}(C)\) such that

\[
X'=QX.
\]

Therefore the complete observation fibre is

\[
\mathcal F_Y(X)
=
\{QX:Q\in O(B),\;Q\mathbf1=\mathbf1,\;QY=Y\}.
\]

### Proof

If \(X'=QX\) and \(QC=C\), then

\[
X'^\top X'=X^\top Q^\top QX=X^\top X,
\]

\[
C^\top X'=C^\top QX=(Q^\top C)^\top X=C^\top X.
\]

The last equality contains both the sample-sum and target-cross statistics, so
Theorem 1 gives oracle equality.

Conversely, oracle equality gives equality of the Gram matrices

\[
[C,X]^\top[C,X]=[C,X']^\top[C,X'].
\]

Two real matrices with the same column Gram matrix define the same inner products
on their column spans. The induced isometry mapping the columns of \([C,X]\) to
the corresponding columns of \([C,X']\) extends to an orthogonal map on
\(\mathbb R^B\). Call this extension \(Q\). Then

\[
Q[C,X]=[C,X'],
\]

so \(QC=C\) and \(QX=X'\). \(\square\)

This is a complete characterization, not merely a sufficient collision
construction. It also shows that generic collisions are not record
permutations: the stabilizer contains continuous rotations whenever its free
subspace has dimension at least two.

## 4. Orbit dimension and identifiability regimes

Let

\[
r=\operatorname{rank}(C),\qquad m=B-r,
\]

and let \(U_\perp\) be an orthonormal basis of
\(\operatorname{span}(C)^\perp\). Define

\[
s=\operatorname{rank}(U_\perp^\top X).
\]

The stabilizer acts as \(O(m)\) on the target-orthogonal subspace. The subgroup
that fixes \(X\) acts as \(O(m-s)\) on the unused part of that subspace.

### Theorem 3 — orbit dimension

The continuous fibre dimension is

\[
\dim\mathcal F_Y(X)
=
\dim O(m)-\dim O(m-s)
=
\frac{s(2m-s-1)}2.
\]

The regimes are:

- \(m=0\) or \(s=0\): this stabilizer mechanism creates no alternative batch;
- \(m=1,s=1\): there is a discrete reflection ambiguity but no continuous
  orbit;
- \(m\ge2,s>0\): the fibre contains a positive-dimensional continuous family.

For generic targets of rank \(c\), \(r\le c+1\). Thus batches with
\(B>c+2\) normally leave at least a two-dimensional target-orthogonal subspace,
which makes continuous ambiguity possible whenever the inputs have a nonzero
component in that subspace.

## 5. Unlimited-query and coherent-query impossibility

### Corollary 4 — no query strategy separates orbit members

If \(X'=QX\) for a target-stabilizing orthogonal \(Q\), then every classical
query receives the same gradient response under \(X\) and \(X'\). Consequently,
any adaptive randomized classical algorithm has identical transcripts in the two
worlds.

The statement also survives coherent access. For any finite deterministic
encoding \(E\) of parameter queries and gradient values, the reversible oracle

\[
U_X|q\rangle|z\rangle
=
|q\rangle|z\oplus E(\mathcal G_{X,Y}(q))\rangle
\]

is exactly the same unitary as \(U_{X'}\). Replacing one hidden batch by the other
therefore leaves the final state of every quantum query algorithm unchanged,
regardless of the number of calls, adaptivity implemented through coherent
control, ancillas, or the final measurement.

This is stronger than saying Grover search does not help: the two experimental
worlds expose the same oracle.

## 6. Domain intersection

The orbit theorem is stated in the ambient real vector space. A constrained
candidate domain contains a collision only when the orbit intersects that domain
more than once.

For an open continuous domain, if every row of \(X\) lies in its interior, a
sufficiently small stabilizer rotation remains in the domain by continuity. For
box constraints this follows whenever the batch has positive distance from the
box boundary. Discrete, categorical, image-grid, token, and graph domains require
an explicit orbit-intersection or finite-fibre analysis; the continuous theorem
must not be transferred to them without that step.

## 7. Executable theorem witness

`qrecon.theory.known_target_collisions` implements:

- exact sufficient statistics and oracle evaluation;
- pointwise full-oracle equivalence checks;
- orthonormal bases of \(\operatorname{span}([1,Y])^\perp\);
- target-stabilizing rotations and reflections;
- the orbit-dimension certificate;
- constructive recovery of an orthogonal stabilizer map from any two batches
  with equal oracle statistics;
- numerical audits of orthogonality, fixed targets, statistics, parameter probes,
  and input displacement.

The tests verify direct and reduced gradient formulas, continuous rotations,
discrete reflections, the dimension formula, the complete converse construction,
and equality over multiple independent parameter probes.

## 8. Paper-level implication

This theorem closes one major ambiguity in the Q-RECON claim ledger. For known
fixed targets, the full biased-linear MSE gradient oracle is globally identifiable
only modulo the target-stabilizer orbit. Neither repeated parameter probing nor
coherent access repairs the loss of information.

A credible end-to-end quantum-advantage experiment must therefore do one of the
following:

1. choose a candidate domain whose intersection with every relevant orbit is a
   single declared equivalence class;
2. release additional information that breaks the stabilizer symmetry;
3. use a nonlinear or otherwise richer leakage channel and prove its global
   identifiability separately; or
4. explicitly target recovery of the orbit/equivalence class rather than the
   original ordered batch.

Any experiment that searches for a unique ordered batch while this fibre remains
nontrivial is information-theoretically mis-specified.
