# Clean Aggregate-Gradient Reconstruction Oracle

## 1. Leakage model

For an ordered batch of \(B\) integer linear-regression records
\((x_j,t_j)\), a released model

\[
z_j=w^	op x_j+b
\]

and squared loss, define the residual

\[
r_j=z_j-t_j.
\]

The released sum gradient is

\[
G_w=\sum_{j=1}^{B}r_jx_j,\qquad
G_b=\sum_{j=1}^{B}r_j.
\]

Q-RECON now implements this map as both a finite reference channel and a clean,
structure-preserving X/CNOT/Toffoli value oracle. The candidate may contain only
the inputs when labels are public, or both inputs and labels when labels are
private. The current implementation treats the batch as an ordered candidate;
all publication results must additionally report recovery modulo batch
permutation when that is the application-appropriate equivalence relation.

## 2. Reversible construction

Let \(U_{\mathrm{rec}}\) be the clean single-record gradient value oracle. The batch
compiler allocates one reusable record-gradient register and one reusable
arithmetic work region. For record \(j\), it:

1. loads the public target into a clean temporary word, or reads the private target
   directly from the candidate register;
2. applies \(U_{\mathrm{rec}}\) to compute \((r_jx_j,r_j)\);
3. adds each component into a persistent aggregate register with a clean
   ripple-carry adder;
4. applies \(U_{\mathrm{rec}}^{-1}\) and unloads any public target word.

After all records, the aggregate is copied into the designated output register.
Every record-accumulation sequence is then reversed in reverse record order,
clearing the aggregate and all temporary registers.

### Theorem 1 — clean aggregate-gradient value oracle

For every valid batch candidate \(D\), output word \(y\), and clean work register,

\[
U_G|Dangle|yangle|0^aangle
=
|Dangle|y\oplus G(D)angle|0^aangle,
\]

where \(G(D)\) is the packed exact sum gradient.

### Proof

Clean correctness of \(U_{\mathrm{rec}}\) gives the exact record-gradient word
without modifying the candidate and with its internal work reset. Each clean
adder updates only the aggregate by the corresponding record component and
returns its carry helper to zero. Reversing \(U_{\mathrm{rec}}\) clears the
record-gradient register. Induction over records therefore leaves the aggregate
at the exact sum and all other reusable work clean. Copying preserves the sum in
the output. The reverse record sequences subtract each contribution and clear the
aggregate, public-target, record-gradient, and arithmetic registers. Only the
candidate and XORed output remain. Linearity extends the basis-state identity to
superpositions. \(\square\)

## 3. Shared-work resource theorem

Let the packed gradient have \(k=d+1\) components of width \(q\), and let one
single-record value oracle use gate counts \((X_r,C_r,T_r)\) and \(a_r\) clean work
qubits. Let one \(q\)-bit clean adder use \((X_a,C_a,T_a)\). Ignoring public-target
constant loads for the moment, one record accumulation sequence has

\[
(2X_r+kX_a,\;2C_r+kC_a,\;2T_r+kT_a).
\]

The complete clean value oracle applies every sequence once to build the sum and
once in reverse after the output copy. Hence

\[
X=4B X_r+2BkX_a,
\]

\[
C=4B C_r+2BkC_a+kq,
\]

\[
T=4B T_r+2BkT_a.
\]

The final \(kq\) CNOTs copy the aggregate. If labels are public, loading and
unloading target \(t_j\) contributes four times its encoded Hamming weight in X
gates across forward computation and reverse cleanup.

The arithmetic work is reused across all records. With candidate width
\(n_D\), the value-oracle logical-qubit count is

\[
Q=n_D+3kq+a_r+b_{\mathrm{public}},
\]

where the three gradient-sized regions are output, aggregate, and reusable
record-gradient storage; \(b_{\mathrm{public}}\) is one reusable public-target word
or zero for private targets. The work cost depends on one record, not \(B a_r\).

The equality/phase verifier computes this value oracle, compares all \(kq\) bits
with the released gradient, and reverses the value oracle. Its machine-readable
report includes the additional value and equality-tree storage.

## 4. Identifiability regimes

### Private targets

When both inputs and regression targets are private, aggregate gradients are
many-to-one in general. The continuous batch-mixing theorem in
`BATCH_GRADIENT_NONIDENTIFIABILITY.md` constructs entire collision families. The
finite two-record benchmark also contains nontrivial collisions after quotienting
out record permutation; they are not merely the symmetry obtained by swapping
records. Consequently an exact-original-batch claim must be bounded by the
appropriate fibre or target-equivalence Bayes ceiling.

### Public targets

Public labels remove degrees of freedom and can make a restricted finite candidate
space identifiable. The executable benchmark with

- \(B=2\), one feature per record;
- two-bit signed inputs \(\{-2,-1,0,1\}\);
- \(w=1,b=0\);
- public ordered targets \((0,1)\);

has 16 candidates and 16 distinct aggregate-gradient observations. This is an
exhaustive finite-domain injectivity certificate, not a general theorem for all
public-label batches.

For a selected candidate in that benchmark, the compiled full-word equality
oracle has exactly one marked input. Its phase netlist drives Grover simulation
and is checked against the standard closed-form success curve.

## 5. Strong classical baseline obligation

Finite injectivity does not imply quantum advantage. Public labels and low-degree
moment equations may admit specialized algebraic, dynamic-programming,
meet-in-the-middle, lattice, or mixed-integer solvers that dominate unstructured
search. Before any advantage claim, the same candidate prior and success target
must be evaluated against:

- exhaustive and branch-and-bound search;
- algebraic elimination or direct moment inversion where available;
- meet-in-the-middle over batch partitions;
- integer programming or SAT/SMT encodings;
- gradient-matching optimization with multiple restarts;
- learned/generative priors when the candidate space is structured.

The compiled Grover path establishes coherent access and an auditable resource
cost. It does not by itself establish the best classical query or time lower
bound.

## 6. Executable evidence

The regression suite checks:

- exact agreement between the arithmetic netlist and the finite reference channel
  for every public-target candidate;
- arbitrary output XOR, inverse restoration, and zero ancillas;
- global injectivity of the 16-candidate public-target benchmark;
- clean unique equality marking and phase signs;
- the Grover success curve from the compiled phase oracle;
- private-target collision fibres over all 256 small candidates;
- one-record arithmetic-work reuse across the complete batch;
- range-contract rejection when aggregate words are too narrow;
- exact logical-qubit, Toffoli, T-count, and T-depth reporting.

## 7. Publication boundary

This milestone closes the first aggregate-training-leakage compiler loop and
provides both a positive finite identifiability example and a negative private-
target collision regime. It still does not justify an end-to-end quantum
advantage claim. The next required result is a nontrivial identifiable structured
batch family for which the strongest specialized classical solver has a proven or
measured scaling gap after the full fault-tolerant oracle, state preparation,
measurement, and amortization costs are included.
