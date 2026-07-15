# Robust End-to-End Costing for Unknown Marked Count

## 1. Objective

A reconstruction experiment often does not know the number `K` of candidates
consistent with the released observation. A cost report must therefore avoid two
optimistic substitutions:

1. selecting the known-`K` Grover iteration count after inspecting the fibre;
2. comparing quantum search only with random classical sampling when a
   structure-aware solver exists.

The executable module `qrecon.oracles.unknown_k_costing` prices a fixed BBHT
schedule and compares it with an externally measured or modeled **specialized
classical solver** in one declared cost unit.

## 2. Per-round quantum cost

For a reached BBHT round with window `m`, the implementation samples

\[
j\sim\operatorname{Uniform}\{0,\ldots,m-1\}.
\]

Let `G(j)` be the fully priced logical execution returned by
`estimate_grover_resources`, including state-preparation gates internal to the
logical run, `j` phase-oracle invocations, `j` diffusion steps, and the declared
measurement price. Define the additional costs

- `L`: external candidate/state loading;
- `R`: measured-word readout and decoding;
- `V`: post-measurement candidate verification.

The conditional mean cost of the round is

\[
C_m=L+R+V+\frac1m\sum_{j=0}^{m-1}G(j).
\]

If `f_{r-1}(K)` is the probability of reaching round `r`, expected variable cost
for marked count `K` is

\[
C_Q(K)=\sum_r f_{r-1}(K)C_{m_r}.
\]

The implementation also averages the decomposed T count separately. Verification
is never absorbed into the phase-oracle count: every reached measurement round
pays `V` once.

## 3. Robust marked-count envelope

Let a uniform finite BBHT certificate cover

\[
K\in\{K_{\min},\ldots,N\}
\]

at target success `eta`. Q-RECON reports

\[
\overline C_Q=\max_{K_{\min}\le K\le N}C_Q(K).
\]

This is a worst-over-`K` **expected** variable cost. It prevents an experiment
from selecting a favorable hidden fibre size after results are known. The report
records the `K` attaining this maximum and the largest expected T count, while
the schedule itself remains independent of `K`.

A separate deterministic worst-case bound is available from the schedule's
maximum phase and verification calls. Papers must state whether they compare
expected, high-probability, or hard worst-case cost; these quantities are not
interchangeable.

## 4. Strong classical opponent

`SpecializedClassicalSolverCosts` has

\[
C_C(M)=S_C+M V_C,
\]

where `S_C` includes reusable preprocessing and `V_C` is the per-instance cost of
the strongest matched solver. Depending on the final task, this solver may be:

- a closed-form analytic decoder;
- algebraic elimination;
- branch-and-bound with sound interval pruning;
- meet-in-the-middle or dynamic programming;
- SAT/SMT or mixed-integer programming;
- multi-start continuous gradient inversion;
- an optimized hybrid of these methods.

Random sampling is not an adequate opponent when the verifier exposes exploitable
structure.

## 5. End-to-end comparison

With quantum compilation cost `S_Q`, amortized workload `M`, and robust variable
cost `Cbar_Q`,

\[
C_Q(M)=S_Q+M\overline C_Q.
\]

Strict advantage requires

\[
S_Q+M\overline C_Q<S_C+MV_C.
\]

When `V_C>\overline C_Q`, the smallest positive workload is the first integer
strictly exceeding

\[
\frac{S_Q-S_C}{V_C-\overline C_Q}.
\]

If the quantum variable cost is no smaller and setup is no better, no amount of
amortization creates an advantage. The executable report returns this threshold
or `None`.

## 6. Unit discipline

All supplied prices must use one common unit. Valid examples include:

- measured wall-clock time under a declared hardware and implementation stack;
- fault-tolerant logical cycles after mapping both methods to a declared compute
  model;
- monetary or energy cost under an explicit conversion;
- an abstract operation unit with independently justified conversion factors and
  sensitivity analysis.

Invalid comparisons include:

- quantum T gates versus classical seconds;
- coherent oracle calls versus Python node visits;
- query counts with compiler/state-preparation/readout omitted;
- classical online time with reusable preprocessing omitted;
- a classical solver evaluated at a different success target or candidate prior.

The costing API intentionally accepts externally supplied classical cost rather
than pretending that one universal operation conversion is scientifically
settled.

## 7. Theorem — robust cost implication

Suppose a BBHT certificate guarantees success at least `eta` for every
`K` in the declared range. Suppose the quantum cost evaluator upper-bounds the
expected cost for each such `K`, and let `Cbar_Q` be their maximum. If

\[
S_Q+M\overline C_Q<S_C+MV_C,
\]

then the modeled quantum pipeline has lower expected total cost than the declared
classical pipeline for every allowed marked count, at quantum success at least
`eta`.

### Proof

For every allowed `K`, certificate validity gives the success guarantee and the
definition of the maximum gives `C_Q(K)<=Cbar_Q`. Multiplying by `M`, adding the
shared quantum setup cost, and applying the strict inequality yields lower total
modeled cost for each `K`. The conclusion is conditional on the declared cost
model and on the classical solver being the strongest matched baseline.
\(\square\)

## 8. What this does not prove

The module makes the comparison auditable; it cannot manufacture an advantage.
A publication claim still requires:

- measured or defensible cost conversions with uncertainty ranges;
- the strongest applicable classical solver suite;
- matched preprocessing rights, success criterion, prior, and target equivalence;
- real compiler resources for the final verifier;
- state preparation for the actual structured candidate distribution;
- robustness of a nonempty advantage region across the predeclared sensitivity
  envelope.

If the robust region is empty, the resulting no-advantage boundary is itself a
valid and potentially important research conclusion.
