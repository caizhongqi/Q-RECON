# Structure-Preserving Reversible Affine Oracle

## 1. Result and scope

This document specifies and proves the first polynomial-size model-aware
coherent compiler in Q-RECON. It lowers an integer-quantized affine model to an
exact reversible circuit built from X, CNOT, and Toffoli gates. It supports:

- multiple signed or unsigned input features;
- signed or unsigned integer affine outputs;
- negative weights through two's-complement modular coefficients;
- a clean multi-output value oracle;
- a clean signed affine-threshold predicate and phase oracle;
- interval certificates that rule out arithmetic overflow when mathematical
  integer semantics, rather than modular semantics, are claimed.

Fractional requantization, ReLU, general comparison, convolution, and nonlinear
loss evaluation remain outside this milestone.

## 2. Arithmetic ring and representation

Fix an accumulator width `m` and the ring

\[
R_m=\mathbb Z/2^m\mathbb Z.
\]

An `n`-bit input word is zero-extended when unsigned and sign-extended when
interpreted in two's complement. Let \(\bar x_i\in R_m\) denote the resulting
ring element. Every integer weight and bias is represented by its residue
\(\bar w_i,\bar b\in R_m\).

For one affine row, the compiled ring function is

\[
F(x)=\bar b+\sum_{i=1}^{d}\bar w_i\bar x_i\pmod {2^m}.
\]

If interval analysis proves that the corresponding mathematical integer result
lies in the declared signed or unsigned `m`-bit range for every valid input,
then the ring word is also the exact intended integer representation.

## 3. Reversible ripple-carry primitive

The compiler uses a fixed-width Cuccaro-style MAJ/UMA ripple-carry adder. On a
clean helper qubit it implements

\[
|a\rangle|b\rangle|0\rangle
\longmapsto
|a\rangle|a+b\bmod 2^m\rangle|0\rangle.
\]

For each bit, the forward carry sweep applies a majority block and the reverse
sweep applies an unmajority-and-add block. Exhaustive tests enumerate all pairs
of input words for widths one through four and verify the sum, preserved addend,
clean helper, and exact inverse.

Under the emitted gate sequence, one `m`-bit addition uses exactly

\[
N_{\mathrm{CCX}}^{\mathrm{add}}=2m,
\qquad
N_{\mathrm{CX}}^{\mathrm{add}}=4m.
\]

The implementation follows the MAJ/UMA construction introduced by Cuccaro et
al., while maintaining an independent local circuit IR and basis-state
executor.

## 4. Constant multiplication by shift-add

Write the modular coefficient as

\[
\bar w=\sum_{k=0}^{m-1}w_k2^k,
\qquad w_k\in\{0,1\}.
\]

For every set coefficient bit, the compiler:

1. copies the sign- or zero-extended input shifted by `k` into a clean `m`-bit
   scratch register;
2. adds the scratch register into the accumulator with the reversible adder;
3. reverses the copy so that the scratch register returns to zero.

The resulting accumulator change is

\[
\sum_{k:w_k=1}2^k\bar x=\bar w\bar x\pmod{2^m}.
\]

A constant bias is loaded into the same scratch register with X gates, added,
and unloaded.

## 5. Clean value-oracle theorem

### Theorem 1

For an affine map with rows

\[
F_j(x)=b_j+\sum_iw_{ji}x_i,
\]

let `C_j` be the emitted reversible shift-add circuit for row `j`. The complete
compute-copy-uncompute circuit implements

\[
U_F|x\rangle|y\rangle|0^a\rangle
=
|x\rangle|y\oplus F(x)\rangle|0^a\rangle,
\]

where every row is encoded modulo \(2^m\), all inputs are preserved, and all
accumulator, scratch, and carry-helper work qubits return to zero.

### Proof

Each shifted copy is a sequence of CNOTs from the immutable input register into
a clean scratch register. The ripple-carry primitive adds that scratch word into
the accumulator while preserving the scratch word and restoring its helper.
Reversing the copy returns scratch to zero. Summing all set coefficient bits and
the bias therefore leaves the accumulator in the word \(F_j(x)\), with all
other work registers clean.

The compiler then CNOT-copies the accumulator into the designated output word.
Every compute gate is X, CNOT, or Toffoli and is self-inverse, so replaying the
compute gate list in reverse returns the accumulator, scratch, and helper to
zero without changing the copied output. Rows are processed sequentially and
reuse the same clean work registers. Thus the final action is the stated clean
XOR oracle. \(\square\)

## 6. Clean threshold and phase oracle

For one signed affine row and public threshold \(\tau\), define

\[
D(x)=b-\tau+\sum_iw_ix_i.
\]

Assume interval analysis proves

\[
-2^{m-1}\le D(x)\le2^{m-1}-1
\]

for every candidate. The compiler computes `D(x)` into a signed accumulator.
Its most significant bit is one exactly when `D(x)<0`. Starting from a target
bit `z`, applying X followed by CNOT from the sign bit gives

\[
z\longmapsto z\oplus\mathbf 1[D(x)\ge0].
\]

The arithmetic is then reversed.

### Theorem 2

Under the certified no-overflow condition, the emitted predicate circuit obeys

\[
U_v|x\rangle|z\rangle|0^a\rangle
=
|x\rangle|z\oplus\mathbf 1[F(x)\ge\tau]\rangle|0^a\rangle.
\]

Preparing the target in \(|-\rangle\) therefore produces the phase oracle

\[
O_v|x\rangle=(-1)^{\mathbf 1[F(x)\ge\tau]}|x\rangle.
\]

The proof follows from Theorem 1, the exact two's-complement sign test, and
reverse uncomputation.

## 7. Exact and symbolic resource bounds

Let

\[
h_j=\sum_i\operatorname{wt}_m(w_{ji})
+\mathbf 1[b_j\not\equiv0\pmod {2^m}]
\]

be the number of ripple additions used to compute row `j`. Let
\(\ell_{jik}\le m\) be the number of copied source bits for set coefficient bit
`k` of weight `w_ji`.

For a clean value row, compute and uncompute together use exactly

\[
N_{\mathrm{CCX},j}=4mh_j.
\]

The exact CNOT count is

\[
N_{\mathrm{CX},j}
=m+2\left[
\sum_{i,k:w_{jik}=1}(4m+2\ell_{jik})
+4m\mathbf 1[b_j\not\equiv0]
\right],
\]

where the leading `m` is the output copy. Consequently,

\[
N_{\mathrm{CX},j}\le m+12mh_j.
\]

Bias loading and unloading contribute exactly

\[
N_{X,j}=4\operatorname{wt}_m(b_j)
\]

across compute and uncompute.

For a threshold predicate, replace `b` by `b-τ`, remove the `m` output-copy
CNOTs, and add one X plus one sign-bit CNOT. Thus

\[
N_{\mathrm{CCX}}=4mh,
\qquad
N_{\mathrm{CX}}\le1+12mh.
\]

Since \(h_j\le dm+1\), a `c`-row value oracle has

\[
N_{\mathrm{CCX}}=O(cdm^2),
\qquad
N_{\mathrm{CX}}=O(cdm^2),
\]

rather than the exponential worst-case term count of complete truth-table
synthesis.

The clean work register contains one `m`-bit accumulator, one `m`-bit scratch
word, and one helper:

\[
A_{\mathrm{work}}=2m+1.
\]

The complete value-oracle logical-qubit count is

\[
Q=dn+cm+2m+1,
\]

and the predicate count is

\[
Q=dn+1+2m+1.
\]

The implementation reports exact emitted gate counts and an ASAP logical depth
computed from wire dependencies; the symbolic formulas are tested on simple
closed-form instances.

## 8. Exhaustive validation strategy

Small instances are checked against three independent semantics:

1. direct mathematical integer affine evaluation;
2. the bit-exact `QuantizedNetwork` reference evaluator;
3. execution of the emitted X/CNOT/Toffoli circuit.

Tests enumerate all candidate words and verify arbitrary output XOR values,
work-register cleanup, inverse restoration, threshold bits, phase signs, and
Grover success. The structure-preserving compiler is also compared with the
minterm and ANF exact finite backends on domains small enough for exhaustive
truth tables.

## 9. Claim boundary

This result closes the polynomial coherent-compilation milestone for integer
affine and signed-threshold models. It does not yet close the full neural-network
milestone because deterministic fractional requantization, ReLU/comparison,
layer composition, and their precision contracts remain pending.

An end-to-end quantum-advantage statement additionally requires a fixed candidate
prior, a measured marked-set size, matched classical success probability,
fault-tolerant gate costs, state loading, compilation amortization, measurement,
and a strict positive break-even region. The executable cost model evaluates
those requirements but does not assume favorable hardware prices.

## References

- S. A. Cuccaro, T. G. Draper, S. A. Kutin, and D. P. Moulton, “A New Quantum
  Ripple-Carry Addition Circuit,” 2004, arXiv:quant-ph/0410184.
