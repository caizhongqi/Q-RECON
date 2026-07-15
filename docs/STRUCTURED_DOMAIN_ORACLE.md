# Clean Structured-Domain Predicates

## 1. Motivation

A fair reconstruction comparison must use the same candidate prior and feasible
set. A classical solver may receive a product domain

\[
\mathcal D=D_1\times\cdots\times D_d,
\]

while an `n`-qubit quantum input register naturally contains every bit word in
`{0,1}^n`. Ignoring invalid words changes the marked fraction, success curve, and
state-preparation problem.

Q-RECON now supports two explicit choices:

1. search the full word space and coherently reject words outside `D`;
2. prepare a state supported only on `D`, with preparation cost reported
   separately.

The first choice is implemented by
`ReversibleProductDomainPredicateOracle` and can be composed with the fixed-point
MLP exact-output verifier.

## 2. Per-feature membership

Let feature `i` use a `b`-bit fixed-point word and let its allowed code set be
`D_i`. For one feature, define

\[
m_i(x_i)=\mathbf 1[x_i\in D_i].
\]

For every constant `a` in `D_i`, a clean constant-equality network toggles a
membership bit when `x_i=a`. Distinct constants are mutually exclusive on a
basis input, so XOR of these equality indicators equals their logical OR:

\[
\bigoplus_{a\in D_i}\mathbf 1[x_i=a]
=\mathbf 1[x_i\in D_i].
\]

No extra disjointness assumption is needed beyond equality to distinct words.
Duplicate caller values are removed during validation.

## 3. Product-domain predicate

After computing all feature membership bits, the oracle toggles its target iff
all are one:

\[
m_{\mathcal D}(x)=\bigwedge_{i=1}^{d}m_i(x_i).
\]

The conjunction is implemented by a clean equality-to-all-ones ladder. The
feature membership networks are then reversed.

### Theorem 1 — clean domain membership

For every input word `x`, target bit `z`, and zero work register,

\[
U_{\mathcal D}|x\rangle|z\rangle|0\rangle
=|x\rangle|z\oplus\mathbf 1[x\in\mathcal D]\rangle|0\rangle.
\]

#### Proof

Each constant equality toggles only the membership bit of its feature and
returns comparator work to zero. Because exactly zero or one declared constants
can equal a fixed feature word, the final feature bit is its domain-membership
indicator. The clean all-ones comparison toggles the target exactly when every
indicator is one and clears its conjunction work. Reversing the feature networks
returns all membership bits and shared equality work to zero. \(\square\)

The circuit is a basis permutation, so the result extends linearly to
superpositions.

## 4. Composition with exact MLP output

Let

\[
e_t(x)=\mathbf 1[F(x)=t]
\]

be the clean fixed-point MLP exact-output predicate. The restricted verifier is

\[
v_{\mathcal D,t}(x)=m_{\mathcal D}(x)\land e_t(x).
\]

`ReversibleDomainRestrictedMLPEqualityOracle` computes both clean predicate bits,
applies one Toffoli into the public target, then reverses both computations.
Thus

\[
U_{\mathcal D,t}|x\rangle|z\rangle|0\rangle
=|x\rangle|z\oplus v_{\mathcal D,t}(x)\rangle|0\rangle.
\]

The marked set is exactly

\[
\{x\in\mathcal D:F(x)=t\},
\]

which is the same solution set used by branch-and-bound and SMT.

## 5. Resource scaling

For one `b`-bit equality, the current ladder uses:

- one CNOT for `b=1`;
- one Toffoli for `b=2`;
- `2b-3` Toffoli gates and `b-2` reusable work bits for `b>=3`;
- two X gates for every zero bit of the compared constant.

If feature `i` has `|D_i|` allowed codes, its membership computation and cleanup
use twice the sum of those equality costs. Equality work is shared sequentially
across constants and features. Product conjunction uses at most `d-2` clean work
bits, also reused.

Consequently the explicit-domain compiler scales with

\[
O\!\left(\sum_i |D_i|b+d\right)
\]

logical equality-gate blocks and with `O(b+d)` shared domain work, rather than
with the full product size `prod_i |D_i|`. This is still expensive when a
per-feature domain is nearly the complete high-precision word set; an interval
comparator or structured state-preparation backend may then be preferable.

## 6. Full-register versus domain-state search

Adding a membership predicate does not change the Hilbert-space population:
standard uniform Hadamards still search all `2^n` bit words, with invalid words
unmarked. This is semantically correct but may reduce the marked fraction.

Preparing

\[
|\mathcal D\rangle=
\frac1{\sqrt{|\mathcal D|}}
\sum_{x\in\mathcal D}|x\rangle
\]

can restore a population of `|D|`, but its preparation and reflection costs must
be included. For arbitrary non-power-of-two and noncontiguous domains, these
operations are not free. Q-RECON therefore reports the domain predicate
separately and does not silently assume ideal structured state preparation.

## 7. Required validation

Small configurations must exhaustively verify:

- reference domain membership for every full-register word;
- both initial target bits;
- zero final work bits;
- inverse recovery of the initial basis state;
- phase sign;
- exact equality among branch-and-bound, SMT, and restricted-oracle solution
  sets;
- deterministic, nonnegative resource counts.

`tests/test_domain_oracle.py` applies these checks to noncontiguous signed
fixed-point domains.

## 8. Claim boundary

The domain compiler removes one candidate-set mismatch. It does not prove that
full-register search is efficient, nor that structured state preparation is
cheap. A final end-to-end claim must choose one search distribution, account for
its preparation and diffusion/reflection implementation, and use the same prior
for all classical baselines.
