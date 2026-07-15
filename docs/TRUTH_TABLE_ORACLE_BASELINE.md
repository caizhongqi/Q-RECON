# Exact Truth-Table Coherent Oracle Baseline

## 1. Scope

This artifact is the first complete coherent-oracle milestone for Q-RECON. It
turns a finite, bit-exact quantized model into a clean value oracle, derives a
one-bit verifier and phase oracle, exhaustively checks basis-state correctness,
and runs an exact Grover state-vector simulation on the compiled predicate.

It is intentionally a **correctness-first exponential baseline**. It does not
claim that truth-table synthesis is asymptotically efficient or that it preserves
an end-to-end quantum advantage. Its role is to fix semantics, establish a
machine-checkable oracle contract, and provide a conservative resource ceiling
against which arithmetic compilers can be compared.

## 2. Bit-exact model semantics

An input component is represented by a fixed-point word with format

\[
(b, f, s, o),
\]

where `b` is the word width, `f` the number of fractional bits, `s` the signedness,
and `o` the overflow policy. A signed code \(q\) represents
\(q2^{-f}\). Signed words use two's complement. Requantization uses nearest
rounding with exact half ties broken away from zero. Overflow is either rejected
or explicitly saturated; wraparound is not implicit.

For an affine layer,

\[
a_j = \sum_i x_i w_{ji} + b_j,
\]

all products are accumulated as unbounded Python integers at the exact product
scale before deterministic requantization. ReLU is applied after requantization.
A range report propagates interval bounds through every layer. Compilation in
the default no-overflow mode is rejected unless all propagated bounds fit their
declared output words.

The public reference function is therefore a total bit function

\[
f_b:\{0,1\}^{n}\rightarrow\{0,1\}^{m}
\]

only after its arithmetic and overflow contract has been fixed.

## 3. Compiler construction

For every input word \(u\) and output bit \(j\) with
\([f_b(u)]_j=1\), the compiler emits a mixed-polarity minterm-controlled X gate
that targets output bit \(j\) and is enabled exactly when the input register is
\(u\). Let the resulting circuit be \(U_f\).

### Theorem 1 — clean value-oracle correctness

For every input word \(x\), output word \(y\), and clean decomposition-ancilla
register,

\[
U_f|x\rangle|y\rangle|0^a\rangle
=
|x\rangle|y\oplus f_b(x)\rangle|0^a\rangle.
\]

### Proof

The compiler never targets the input register. Among all input minterms, exactly
one matches the computational-basis input \(x\). For that minterm, output bit
\(j\) is toggled exactly when \([f_b(x)]_j=1\); all other minterms are disabled.
Consequently the output register is XORed with \(f_b(x)\). Every abstract
multi-controlled X is self-inverse. Under the declared clean-ancilla
decomposition, its temporary conjunction bits are uncomputed before the gate
returns. Therefore all work qubits finish in zero. Linearity extends the
basis-state action to arbitrary superpositions. \(\square\)

Because the map preserves \(x\) and XORs a fixed word into \(y\), it is a
permutation of basis states and its inverse is itself.

## 4. Verifier and phase oracle

Given a public target word \(t\), metric \(d\), and threshold \(\tau\), define

\[
v(x)=\mathbf 1[d(f_b(x),t)\le \tau].
\]

The implementation supports exact mismatch, Hamming distance, and absolute word
distance. Applying the clean one-bit verifier to a target qubit in
\(|-\rangle\) gives

\[
|x\rangle|-\rangle\mapsto(-1)^{v(x)}|x\rangle|-\rangle,
\]

so the predicate is a valid phase oracle. The number of marked inputs is
measured from the complete table and is never assumed to be one.

## 5. Exact resource upper bound

Let

\[
S=\sum_{x\in\{0,1\}^n}\operatorname{wt}(f_b(x))
\]

be the total number of set output bits in the truth table. The baseline emits
exactly \(S\) mixed-polarity \(n\)-controlled X gates.

Under the documented naive synthesis:

- every zero-valued control is implemented by an X before and after its minterm;
- a one-controlled X is one CNOT;
- for \(n\ge 2\), an \(n\)-controlled X uses \(n-2\) clean ancillas and
  \(2n-3\) Toffoli gates;
- one exact Toffoli is conservatively charged 7 T gates and T-depth 3.

Therefore, for \(n\ge2\),

\[
N_{\mathrm{Toffoli}}=S(2n-3),
\qquad
A_{\mathrm{peak}}=n-2,
\]

and

\[
N_T\le 7S(2n-3),
\qquad
D_T\le3S(2n-3).
\]

The negative-control X count is reported exactly for the emitted minterms. These
are transparent upper bounds for this implementation, not optimal synthesis
claims. Since \(S\le m2^n\), the construction is exponential in the worst case.

## 6. Finite identifiability certificate

Complete enumeration partitions the candidate words into output fibres. The
artifact reports:

- population and number of distinct observations;
- injectivity;
- number of candidates belonging to non-singleton fibres;
- largest fibre and fibre-size histogram;
- uniform-prior Bayes exact-reconstruction ceiling;
- corresponding conditional min-entropy.

This is a global finite-space collision analysis, unlike the existing Jacobian
rank diagnostic, which is only local and differential.

## 7. End-to-end logical search check

The compiled one-bit verifier is applied in an exact state-vector Grover
simulation. Starting from the uniform state, every iteration performs the
compiled predicate phase flip followed by the standard diffusion operator. If
there are \(K\) marked inputs among \(N\), the measured success is regression-
tested against

\[
\sin^2((2r+1)\arcsin\sqrt{K/N}).
\]

A machine-readable logical resource report includes state preparation, verifier
calls, diffusion operations, logical qubits, Toffoli count, T-count, and T-depth
under the same synthesis assumptions.

## 8. Exhaustive acceptance tests

For every supported small model configuration, tests enumerate all input words
and verify:

1. reference evaluation and oracle output agree bit for bit;
2. arbitrary output words are XORed correctly;
3. applying the oracle twice restores the complete basis state;
4. all ancillas are clean;
5. the basis action is a permutation;
6. verifier marks agree with the public metric and threshold;
7. phase signs agree with the verifier table;
8. finite collision statistics are exact;
9. Grover simulation agrees with the closed-form success curve;
10. unsafe no-overflow models are rejected before compilation.

## 9. Publication claim boundary

This milestone establishes a real coherent-oracle implementation in the finite
truth-table model. It supports a compiler-correctness theorem and an exact
resource theorem for the emitted baseline. It does **not** establish a useful
asymptotic or practical advantage because its synthesis cost is exponential.

The next publishable systems result must replace minterm synthesis with a
structure-preserving reversible arithmetic compiler for affine layers,
requantization, ReLU/comparison, and uncomputation. That compiler must be checked
against this truth-table oracle on exhaustive small instances and must exhibit a
strictly better symbolic and measured resource curve before an end-to-end
advantage region can be claimed.
