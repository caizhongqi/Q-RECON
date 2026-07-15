# Explicit Empirical Tables: Certified No-Advantage Regions

## 1. Motivation

A candidate table containing `N` records of `w` bits is not automatically a free
coherent oracle. If the table is supplied as an explicit classical description
and the quantum pipeline compiles an exact lookup unitary, the compiler must read
every description bit in the worst case. Treating this setup as free QRAM would
change the access model rather than optimize the same pipeline.

This document converts that observation into an executable end-to-end negative
certificate. The certificate is intentionally one-sided: it can prove that a
strict quantum cost advantage is impossible under declared lower and upper
bounds. Failure to certify no advantage does not prove advantage.

## 2. Compiler bit-probe theorem

Let

\[
T\in\{0,1\}^{N\times w}
\]

be an arbitrary explicit table, and let an exact compiler output a circuit for

\[
U_T|i\rangle|y\rangle=|i\rangle|y\oplus T_i\rangle.
\]

### Theorem 1 — exact explicit-table compilation requires `Nw` bit probes

In the worst case, every exact compiler must inspect all `Nw` input-description
bits.

### Proof

Suppose one table bit is never inspected. Two tables that differ only in that bit
produce the same compiler transcript and hence the same compiled artifact. Their
lookup unitaries differ on the corresponding address and output bit, so one of
the two artifacts is incorrect. Therefore no exact compiler can omit any bit in
the worst case. `□`

The result is independent of Grover search and independent of circuit synthesis.
It is a setup lower bound caused by the chosen explicit-data access model.

## 3. One-workload no-advantage certificate

Choose one common cost unit in which one explicit-table bit probe costs one unit.
Let:

- `Q_setup_extra_lower` be any additional quantum setup lower bound;
- `Q_variable_lower` be a lower bound per reconstruction instance after setup;
- `C_setup_upper` be an upper bound for complete matched classical setup;
- `C_variable_upper` be an upper bound per complete matched classical instance;
- `M` be the number of reconstruction instances sharing setup.

Then

\[
C_Q(M)\ge Nw+Q_{setup,extra}^{L}+M Q_{var}^{L},
\]

while

\[
C_C(M)\le C_{setup}^{U}+M C_{var}^{U}.
\]

### Theorem 2 — strict no-advantage certificate

If

\[
Nw+Q_{setup,extra}^{L}+M Q_{var}^{L}
\ge C_{setup}^{U}+M C_{var}^{U},
\]

then strict quantum cost advantage is impossible for that workload under the
declared common unit and access model.

### Proof

Every admissible quantum implementation costs at least the left-hand side, while
at least one complete matched classical implementation costs no more than the
right-hand side. The inequality therefore implies `C_Q(M) >= C_C(M)` for the best
quantum implementation relative to the exhibited classical upper bound. `□`

This is stronger than comparing two measured point estimates: the comparison is
between a quantum lower bound and a classical upper bound.

## 4. Exact workload region

Define

\[
A=Nw+Q_{setup,extra}^{L}-C_{setup}^{U},
\qquad
B=Q_{var}^{L}-C_{var}^{U}.
\]

The no-advantage workloads are exactly the positive integers satisfying

\[
A+MB\ge0.
\]

The resulting region is always one of:

- empty;
- a bounded prefix `{1, ..., M_max}` when setup dominates initially but classical
  per-instance cost is larger;
- an unbounded suffix `{M_min, M_min+1, ...}` when quantum per-instance lower cost
  is no better than the classical upper bound;
- all positive workloads.

A bounded prefix makes the amortization obligation explicit: a paper may discuss
possible advantage only beyond the first workload not ruled out by loading.

## 5. Executable implementation

`qrecon.theory.data_loading_boundary` provides:

- `ExplicitTableNoAdvantageCertificate`;
- `PositiveIntegerWorkloadRegion`;
- `certify_explicit_table_no_advantage`;
- `certified_explicit_table_no_advantage_region`.

Tests cover one-shot certificates, bounded setup-amortization prefixes, unbounded
regions, late-onset regions, empty regions, exact boundary points, and invalid
cost declarations.

## 6. Access-model boundary

The theorem applies when the candidate data arrive as an explicit classical
table that must be compiled exactly. It must not be silently transferred to:

- a physical QRAM assumed to be populated before accounting starts;
- a succinct generator whose description is asymptotically smaller than `Nw`;
- an algorithmic dataset with a public reversible generation procedure;
- a pre-existing coherent service supplied as part of the threat model.

Those are legitimate but different access assumptions. Their construction,
population, maintenance, error, and invocation costs must be stated separately.

## 7. Paper consequence

For a real empirical candidate set, Q-RECON must report both:

1. the search cost after coherent access exists; and
2. the cost or explicit external assumption by which coherent access is obtained.

If the executable certificate covers every workload in the experimental range,
the correct result is a no-advantage boundary, not an omitted setup term. A
rigorous negative phase diagram of this form can be a central contribution when
paired with the identifiability and compiler theorems.
