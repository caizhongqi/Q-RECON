# Final-Head Identifiability in Shared-Head Modern Forecasters

## 1. Scope

This note formalizes what can and cannot be reconstructed when an attacker sees
only the exact gradients of the final biased linear forecasting head. It applies
to modern models such as PatchTST and iTransformer whenever the same final head is
shared over a set of effective examples. It does **not** claim that the same fibre
survives after all encoder-parameter gradients are also released.

Let a fixed encoder produce effective representations

\[
Z\in\mathbb R^{m\times d},
\]

and let the shared final head be

\[
\widehat Y=ZW^\top+\mathbf 1 b^\top,
\qquad
W\in\mathbb R^{h\times d},\quad b\in\mathbb R^h,
\]

with known targets \(Y\in\mathbb R^{m\times h}\) and mean squared loss over all
\(mh\) scalar outputs. For a batch of \(B\) multivariate examples with \(C\)
channels folded through a shared channel-independent head, the effective sample
count is

\[
m=BC.
\]

For a univariate model, \(C=1\) and therefore \(m=B\).

## 2. Sufficient statistics

The released head gradients are

\[
\nabla_W L
=\frac{2}{mh}(ZW^\top+\mathbf1b^\top-Y)^\top Z,
\]

\[
\nabla_b L
=\frac{2}{mh}(ZW^\top+\mathbf1b^\top-Y)^\top\mathbf1.
\]

For arbitrary queried head parameters \((W,b)\), the complete head-gradient
oracle depends on the private representation matrix only through

\[
S=Z^\top Z,
\qquad
u=Z^\top\mathbf1,
\qquad
R=Y^\top Z.
\]

This is the same known-target biased-linear gradient-oracle quotient implemented
in `qrecon.theory.known_target_collisions`.

## 3. Target-stabilizer fibre theorem

Define the target constraint matrix

\[
A=[\mathbf1,Y]\in\mathbb R^{m\times(h+1)}
\]

and its orthogonal complement dimension

\[
s=m-\operatorname{rank}(A).
\]

### Theorem 1 — exact indistinguishable orbit

For every orthogonal matrix \(Q\in O(m)\) satisfying

\[
QA=A,
\]

the transformed representation

\[
Z'=QZ
\]

produces exactly the same final-head gradient for every \((W,b)\).

### Proof

The stabilizer conditions imply

\[
Q\mathbf1=\mathbf1,
\qquad
QY=Y.
\]

Orthogonality then gives

\[
Z'^\top Z'=Z^\top Q^\top QZ=Z^\top Z,
\]

\[
Z'^\top\mathbf1=Z^\top Q^\top\mathbf1=Z^\top\mathbf1,
\]

and

\[
Y^\top Z'=Y^\top QZ=Y^\top Z.
\]

All sufficient statistics are unchanged, so the entire released head-gradient
oracle is unchanged. \(\square\)

The conclusion holds for classical, white-box and coherent access to this
released head-gradient oracle: identical classical functions induce identical
reversible oracle unitaries.

## 4. Effective-sample threshold

If \(s=0\), the target stabilizer contains only the identity on the sample-index
space and this particular collision mechanism is absent.

If \(s\ge1\) and the projection of \(Z\) onto the target-orthogonal subspace is
nonzero, reflection in that subspace gives a nontrivial discrete collision.

If \(s\ge2\) and the projected representation has nonzero rank, rotations in the
orthogonal complement give a continuous collision family. The executable orbit
report computes the exact projected rank and orbit dimension instead of assuming
genericity.

Since

\[
\operatorname{rank}[\mathbf1,Y]\le h+1,
\]

the following architecture-level sufficient conditions are immediate:

- \(m>h+1\) guarantees a nonzero target-orthogonal subspace;
- \(m>h+2\) guarantees that this subspace has dimension at least two.

For a shared multivariate forecasting head this becomes

\[
BC>h+1
\]

for a possible reflection fibre and

\[
BC>h+2
\]

for a possible continuous rotation fibre, subject to nonzero projected
representations.

These are sufficient dimension thresholds, not claims that every representation
has nonzero projection.

## 5. Consequences for PatchTST and iTransformer

### Univariate PatchTST

For batch size one, \(m=1\). The constraint vector \(\mathbf1\) already spans the
sample-index space, so this final-head orbit mechanism cannot create a collision.
This matches the revision-pinned GIFT-Eval/PatchTST experiment: all 20 audited
single-window points have zero orthogonal-complement dimension.

For horizon \(h=4\), a generic univariate batch first acquires a target-orthogonal
subspace once \(B\ge6\), and a two-dimensional complement once \(B\ge7\). The
batch-size sweep in `examples/gifteval_patchtst_final_head_orbit_batch_sweep.py`
checks the actual target ranks and representations rather than relying only on the
generic count.

### Multivariate iTransformer

With one seven-channel ETTm1 record, the shared head sees \(m=7\) effective
samples. For horizon four, \([\mathbf1,Y]\) has rank at most five, so the
orthogonal complement has dimension at least two. The revision-pinned ETTm1
experiment finds a one-dimensional continuous representation orbit on every one
of 20 audited records, with constructed final-head gradient discrepancies at
floating-point roundoff.

## 6. Information-theoretic recovery consequence

Let a prior assign positive mass or density to two non-equivalent reachable
representations in the same orbit. Any estimator based only on the released final
head gradients has exact-recovery success bounded by the posterior mass of the
most likely orbit member. No optimization method and no quantum query algorithm
can select the original member from information absent from the observation.

The theorem is representation-level. To claim raw-input non-identifiability, one
must additionally establish that at least two orbit members are reachable from
valid raw inputs through the fixed encoder, or exhibit a raw-input symmetry whose
encoded representations lie in the orbit.

## 7. Claim boundary

This result supports a strong partial-gradient conclusion:

> Exact final-head gradients of a shared modern forecasting head need not identify
> the effective hidden representations once the effective sample count exceeds the
> target-constraint rank.

It does not by itself establish:

- non-identifiability under complete encoder-gradient release;
- reachability of every transformed representation from valid time series;
- practical quantum advantage;
- a defense guarantee for arbitrary optimizers or multi-step updates.

Those distinctions are mandatory in the final paper.
