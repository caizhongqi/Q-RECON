# Structure-Preserving Reversible MLP Oracle

## 1. Result and scope

Q-RECON now lowers a non-linear integer network of the form

\[
x \longmapsto W_2\,\operatorname{ReLU}(W_1x+b_1)+b_2
\longmapsto \mathbf 1[\cdot\ge\tau]
\]

to a clean gate-level predicate oracle containing only X, CNOT, and Toffoli
operations. The supported milestone is deliberately precise:

- two affine layers;
- a componentwise ReLU between them;
- integer/two's-complement words with no fractional requantization;
- a one-bit final threshold predicate;
- interval-certified no-overflow arithmetic;
- exact compute-copy-uncompute semantics.

This is the first Q-RECON compiler path whose non-linearity is implemented by a
reversible circuit rather than hidden inside a truth table or a variational prior.

## 2. Clean ReLU construction

Let \(z\) be a little-endian \(w\)-bit two's-complement word and let \(s\) be its
sign bit. For a clean output word \(r\), the compiler implements

\[
|z\rangle|r\rangle
\mapsto
|z\rangle|r\oplus\max(0,z)\rangle.
\]

It temporarily applies X to \(s\). Each non-sign input bit then controls a
Toffoli together with the inverted sign bit, targeting the corresponding output
bit. The sign bit is restored at the end; the output sign bit is untouched and
therefore remains zero. One ReLU compute uses

\[
2\text{ X}+(w-1)\text{ Toffoli}
\]

for \(w>1\), and no gates for the degenerate one-bit signed domain. Reversing the
same gate list clears the activation register exactly.

## 3. Network composition

Let \(U_1\) be the clean first-affine value oracle and \(U_2\) the clean final
affine-threshold predicate oracle. The network compiler performs:

1. apply \(U_1\) to materialize all hidden preactivations;
2. compute every ReLU output into a clean activation register;
3. apply \(U_2\) to XOR the final predicate into the target qubit;
4. reverse all ReLU gates;
5. apply \(U_1^{-1}\) to clear hidden preactivations and arithmetic work.

### Theorem — clean MLP predicate correctness

For every valid input word \(x\), target bit \(y\), and clean work register,

\[
U_{\mathrm{MLP}}|x\rangle|y\rangle|0^a\rangle
=
|x\rangle|y\oplus v(x)\rangle|0^a\rangle,
\]

where

\[
v(x)=\mathbf 1\!\left[
W_2\operatorname{ReLU}(W_1x+b_1)+b_2\ge\tau
\right].
\]

### Proof

Clean correctness of \(U_1\) leaves the input unchanged, writes the exact
first-layer words, and returns its own arithmetic work to zero. The ReLU block
copies exactly the non-negative hidden values without changing the preactivation
registers. Clean correctness of \(U_2\) toggles only the target by the declared
threshold predicate and clears its work. Reversing the ReLU block clears every
activation because the preactivations are unchanged. Finally \(U_1^{-1}\)
clears all first-layer outputs and work. Thus only the input and toggled target
remain. Every component is a reversible basis permutation, so linearity extends
the identity to arbitrary superpositions. \(\square\)

## 4. Exact resource composition

Let \((X_1,C_1,T_1)\) be the X/CNOT/Toffoli counts for one clean first-affine
value-oracle call and \((X_2,C_2,T_2)\) those for the final predicate. For \(h\)
hidden neurons of width \(w>1\), the complete clean MLP oracle has exact counts

\[
X=2X_1+X_2+4h,
\]

\[
C=2C_1+C_2,
\]

\[
T=2T_1+T_2+2h(w-1).
\]

The factors of two account for Bennett uncomputation of the first affine layer
and the ReLU outputs. Under the repository's conservative exact-Toffoli charge,

\[
T\text{-count}\le7T,
\qquad
T\text{-depth}\le3T.
\]

If the two affine subcompilers require \(a_1\) and \(a_2\) clean work qubits, the
network uses

\[
n_{\mathrm{in}}+1+2hw+a_1+a_2
\]

logical qubits: input, predicate target, hidden preactivation and activation
words, and the two reusable arithmetic work regions.

Because the affine subcircuits use constant shift-add multiplication and
ripple-carry addition, this construction is polynomial in layer dimensions and
word widths. It is structurally different from the exponential minterm and ANF
truth-table baselines.

## 5. Verification obligations implemented

The test suite exhaustively checks small configurations for:

- two's-complement ReLU correctness for every input and output word;
- preservation of the ReLU input and exact inverse cleanup;
- equality with the public `QuantizedNetwork` evaluator on every candidate;
- XOR behavior for both initial target states;
- zero hidden and arithmetic work after every oracle call;
- inverse restoration of the complete basis state;
- phase signs used by Grover simulation;
- exact gate-count decomposition formulas;
- rejection of unproved overflow and unsupported fractional scaling.

The Grover test computes marked candidates from the reference evaluator but
applies phases through the compiled gate circuit. Any disagreement between
reference semantics and the netlist therefore changes the measured search curve
and fails regression.

## 6. Claim boundary and next compiler theorem

This milestone proves clean coherent access for a finite-word, two-layer integer
MLP. It does not yet establish practical fault-tolerant advantage. The next
compiler result must add deterministic fixed-point requantization and support
multiple hidden layers while reducing ancilla and T-depth through liveness-aware
register reuse. The end-to-end paper claim still requires matched classical
costs, state-preparation cost, oracle reuse assumptions, approximate arithmetic
error, and experiments over a nontrivial reconstruction candidate space.
