# Algebraic-Normal-Form Oracle Optimization

## 1. Purpose

The minterm compiler in `TRUTH_TABLE_ORACLE_BASELINE.md` is an exact correctness
baseline, but every set truth-table output bit becomes a fully controlled gate.
This document defines a second exact backend based on algebraic normal form
(ANF), also called the positive-polarity Reed–Muller form. It often removes
negative controls and can reduce the non-Clifford cost dramatically for
low-degree Boolean structure.

The backend remains finite and truth-table derived. Its worst-case size is still
exponential, so it is an optimization and a cross-checking backend rather than a
replacement for the pending structure-preserving arithmetic compiler.

## 2. Unique Boolean polynomial

Every Boolean output bit

\[
f_j:\{0,1\}^n\rightarrow\{0,1\}
\]

has a unique multilinear polynomial over \(\mathbb F_2\):

\[
f_j(x)=\bigoplus_{S\subseteq[n]}a_{j,S}
\prod_{i\in S}x_i.
\]

The coefficient vector is obtained from the truth table by the Boolean Möbius
transform. The implementation applies the in-place recurrence

\[
a[m]\leftarrow a[m]\oplus a[m\setminus\{i\}]
\]

for each variable \(i\) and every mask \(m\) containing that variable.

## 3. Clean oracle construction

For each nonzero coefficient \(a_{j,S}=1\), emit one positive-control
multi-controlled X targeting output bit \(j\), with controls on the input bits in
\(S\). The empty monomial emits an unconditional X and a degree-one monomial
emits a CNOT.

### Theorem 1 — ANF oracle correctness

Let \(U_{\mathrm{ANF}}\) be the emitted gate sequence. Then for every input and
output word,

\[
U_{\mathrm{ANF}}|x\rangle|y\rangle|0^a\rangle
=|x\rangle|y\oplus f(x)\rangle|0^a\rangle.
\]

### Proof

For a fixed output bit \(j\), a monomial gate toggles the target exactly when all
variables in its support are one. XORing the target over all nonzero ANF
coefficients therefore evaluates the unique polynomial for \(f_j(x)\). Repeating
this independently for every output bit produces \(f(x)\). Input bits are only
controls. Every gate is self-inverse, and the declared multi-control
decomposition uncomputes its clean temporary conjunctions. Hence the complete
map is a clean XOR oracle. \(\square\)

The implementation independently evaluates the emitted monomial gates and checks
the result against every reference truth-table row; it does not use the table as
the execution path.

## 4. Resource theorem

Let \(d_t\) be the degree of emitted monomial \(t\). Under the same exact
clean-ancilla multi-control convention as the minterm baseline:

- \(d_t=0\): one X;
- \(d_t=1\): one CNOT;
- \(d_t\ge2\): \(2d_t-3\) Toffolis and \(d_t-2\) reusable clean ancillas.

Therefore

\[
N_{\mathrm{Toffoli}}
=\sum_{t:d_t\ge2}(2d_t-3),
\]

\[
A_{\mathrm{peak}}=\max_t\max\{0,d_t-2\},
\]

and, under the declared exact Toffoli accounting,

\[
N_T\le7N_{\mathrm{Toffoli}},
\qquad
D_T\le3N_{\mathrm{Toffoli}}.
\]

No negative-control X gates are required.

For affine Boolean functions, all monomials have degree at most one. The ANF
backend therefore uses only X and CNOT gates and has zero Toffoli/T cost under
this accounting. The regression suite verifies this property on parity
predicates.

## 5. Deterministic backend selection

`compare_exact_syntheses` compiles the same reference function with both the
mixed-polarity minterm backend and the ANF backend. Selection is lexicographic in:

1. T-count upper bound;
2. Toffoli count;
3. number of controlled-X terms;
4. logical depth upper bound.

Because both candidates are independently exact, selecting the smaller resource
key cannot change the function or marked set. The decision and both complete
resource records remain visible; the losing backend is not discarded from the
report.

## 6. Cross-backend validation

Tests verify that:

- Möbius coefficients reproduce every output bit of multibit functions;
- ANF gate execution equals the reference table for every input;
- arbitrary output words are XORed correctly;
- the gate map is self-inverse and a basis-state permutation;
- ANF and minterm predicates mark the same candidates;
- either backend drives the same Grover success curve;
- parity uses only degree-one monomials and eliminates Toffolis.

Using two mathematically different exact synthesis paths is also a defense
against compiler-test circularity: their gate lists and resource formulas differ,
while their basis-state semantics must agree.

## 7. Claim boundary

ANF can yield exponential savings relative to naive minterms on structured
Boolean functions, but its own worst case still contains exponentially many
monomials. No polynomial end-to-end model-oracle claim follows from this backend.

The next compiler must lower affine multiply-accumulate, deterministic
requantization, ReLU/comparison, and uncomputation directly from the quantized
model IR. Minterm and ANF circuits then serve as independent exhaustive oracles
for small-instance equivalence checking and as transparent upper-bound baselines.
