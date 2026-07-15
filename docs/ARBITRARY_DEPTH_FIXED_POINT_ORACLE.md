# Arbitrary-Depth Fixed-Point MLP Oracle

## 1. Scope

This document specifies and proves the composition contract implemented by
`qrecon.oracles.fixed_point_deep_mlp`. It closes the previous two-layer-only gap
for fixed-point `Affine`/`ReLU` networks while retaining the same bit-exact
reference semantics, reachability checks, clean-ancilla requirement, and
construction-noncircularity rule.

The supported network is

\[
f = f_L\circ f_{L-1}\circ\cdots\circ f_1,
\]

where every `f_i` is a declared `QuantizedAffineLayer`, adjacent dimensions and
fixed-point formats match exactly, hidden layers may use `identity` or `relu`,
and the final layer uses `identity`. Rounding, signedness, fractional bits, and
overflow semantics are inherited from the public `QuantizedNetwork` evaluator.

The compiler does not enumerate the candidate domain or build a preimage table.
Its construction depends only on the layer parameters and declared word formats.

## 2. Layerwise reachability certificate

Let `R_i` be the interval range report obtained by propagating the original input
word range through layers `1,...,i`. The compiler records

- raw output bounds before the output-format overflow rule;
- encoded output bounds after the declared rule;
- the conjunction that all reachable raw values fit their declared formats.

When `require_no_overflow=True`, compilation is rejected unless every reachable
layer value is representable. This is deliberately weaker and more accurate than
requiring every bit pattern in an intermediate register to be semantically
reachable. The generated circuit remains a total reversible permutation, while
bit-exact equivalence is certified on every original input word.

## 3. Clean composition theorem

For layer `i`, assume a clean value oracle

\[
U_i|z_{i-1}\rangle|z_i\rangle|0^{a_i}\rangle
=
|z_{i-1}\rangle|z_i\oplus f_i(z_{i-1})\rangle|0^{a_i}\rangle.
\]

Allocate a retained register for every hidden activation, one public output
register, and one shared work register of width

\[
a_{\max}=\max_i a_i.
\]

The network circuit performs

1. `U_1,...,U_{L-1}` to compute hidden words;
2. `U_L` to XOR the final network value into the public output;
3. `U_{L-1}^{-1},...,U_1^{-1}` to clear every hidden word.

### Theorem 1 — bit-exact clean network oracle

For every valid input word `x`, output word `y`, and zero work state, the compiled
circuit satisfies

\[
U_f|x\rangle|y\rangle|0^A\rangle
=
|x\rangle|y\oplus f(x)\rangle|0^A\rangle,
\]

where

\[
A=\sum_{i=1}^{L-1}h_i+a_{\max}
\]

and `h_i` is the packed output width of hidden layer `i`.

#### Proof

After the forward prefix, induction on `i` gives that hidden register `i`
contains `f_i\circ\cdots\circ f_1(x)` and the shared arithmetic work is zero,
because each component oracle is clean. The final layer therefore XORs exactly
`f(x)` into the public output. Reversing hidden stages in descending order is
valid because all earlier inputs required by each inverse are still present.
Each inverse clears its corresponding hidden register and restores the shared
work to zero without touching the public output. The input is never modified.
Therefore the final state has the stated form. `square`

The implementation verifies this identity exhaustively for every input word on
small configurations and checks both forward and inverse permutations on
multiple nonzero output words.

## 4. Exact resource composition

Let `G_i` be the count of any self-inverse logical gate type in one synthesized
layer oracle and let `D_i` be its reported logical-depth upper bound. The composed
value oracle contains one copy of the final layer and a compute/uncompute pair
for every hidden layer. Hence, for X, CNOT, Toffoli, H, and Z counts independently,

\[
G_{\mathrm{net}}=G_L+2\sum_{i=1}^{L-1}G_i.
\]

A conservative sequential depth bound is

\[
D_{\mathrm{net}}\le D_L+2\sum_{i=1}^{L-1}D_i.
\]

The logical-qubit count is exactly

\[
Q=n_{\mathrm{in}}+n_{\mathrm{out}}
+\sum_{i=1}^{L-1}h_i+\max_i a_i.
\]

The key ancilla improvement is shared arithmetic work: clean per-layer work is
reused instead of summed. Hidden activation words cannot generally be reused in
this Bennett schedule because later layers and reverse cleanup require them.
`resource_breakdown()` exposes the one-copy resource record for every layer, the
multiplicity vector `(2,...,2,1)`, retained-hidden width, shared-work width, and
the synthesized total. Regression tests assert the gate-count identities rather
than merely checking that counts are positive.

## 5. Threshold and exact-observation phase oracles

For a one-output network, the threshold verifier computes the deep value, applies
a signed/unsigned integer comparison to a public code `tau`, and reverses the
value circuit:

\[
v_\tau(x)=\mathbf 1[f(x)\ge\tau].
\]

For any output dimension, the equality verifier uses a packed constant comparator:

\[
v_t(x)=\mathbf 1[f(x)=t].
\]

Both satisfy the clean XOR contract

\[
|x\rangle|b\rangle|0\rangle
\mapsto
|x\rangle|b\oplus v(x)\rangle|0\rangle,
\]

and therefore support phase kickback without leaving input-dependent garbage.
The verifier cost includes a complete forward and inverse deep-value computation;
it is not priced as a single classical network evaluation.

## 6. What this result does and does not establish

This compiler result establishes:

- arbitrary-depth bit-exact fixed-point composition within the declared layer
  semantics;
- clean hidden/work uncomputation;
- candidate-enumeration-free construction;
- exact layerwise gate multiplicities and shared-work qubit accounting;
- exhaustive small-width value, inverse, threshold, equality, and phase tests.

It does not establish practical quantum advantage. State/domain preparation,
diffusion, fault-tolerant synthesis, unknown-`K` search, repetitions,
measurement, and the strongest matched classical inversion algorithm remain in
the end-to-end comparison. The theorem removes a compiler-correctness gap; it
does not remove the cost or identifiability gates.
