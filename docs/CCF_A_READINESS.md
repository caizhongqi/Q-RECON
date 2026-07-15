# Q-RECON CCF-A Readiness and Claim Gates

## 1. Purpose

This document is an internal acceptance contract for turning Q-RECON from a
research prototype into a submission-grade top-tier paper. It does not predict
or guarantee conference acceptance. No paper draft may claim more than the
strongest theorem, compiler artifact, experiment, and classical comparison that
all hold simultaneously.

The project separates five layers:

1. **information layer** — whether the released observation identifies the
   private training object at all;
2. **access layer** — whether the attacker receives classical samples, released
   internals, or a coherent reversible oracle;
3. **query layer** — how many ideal verifier calls are required, including the
   unknown-marked-count protocol;
4. **implementation layer** — how the verifier, candidate domain, state
   preparation, diffusion, precision, measurement, and cleanup are implemented;
5. **end-to-end layer** — whether the full quantum pipeline beats the strongest
   matched classical pipeline at the same success criterion and in one cost unit.

A result at one layer never implies a result at a later layer.

## 2. Submission thesis candidates

### Thesis A — identifiability limits of training-gradient reconstruction

Core claim:

> Exact or approximate optimization cannot recover information absent from the
> released gradient channel; for declared models and loss functions, Q-RECON
> gives explicit collision families, Bayes ceilings, and sharp
> identifiable/non-identifiable regimes.

Required evidence:

- formal observation channel, candidate prior, and target equivalence;
- explicit collision family or injectivity theorem;
- Bayes-optimal exact/equivalence-class success;
- finite and continuous counterexamples;
- experiments separating optimization failure from information failure.

Current state: **strong foundation**. Aggregate private-label linear regression
has an explicit continuous collision family. Full exact single-record biased
linear gradients have a proved nonzero-residual analytic decoder and a
zero-residual non-identifiable fibre. For fixed known targets, the complete
biased-linear MSE gradient-oracle fibre is characterized by the target-stabilizer
orbit, with an exact orbit-dimension formula, identifiable quotient, optimal
packed query plan, and matching physical lower bound. The final nonlinear or
partial-leakage task still needs its own collision/injectivity result.

### Thesis B — clean structure-preserving compilation of reconstruction oracles

Core claim:

> A declared quantized model/leakage verifier and finite structured domain can be
> compiled without global candidate enumeration into a clean reversible
> value/equality/phase oracle, with exact bit semantics, complete uncomputation,
> symbolic resource bounds, and exhaustive small-width verification.

Required evidence:

- public classical reference semantics;
- candidate-enumeration-free arithmetic compiler;
- explicit structured-domain predicate or priced domain-state preparation;
- compute-copy-uncompute correctness and clean-ancilla theorems;
- exact gate/qubit/depth accounting;
- cross-check against independent truth-table/ANF backends;
- scaling beyond enumerative backends.

Current state: **substantial and increasingly complete within declared scope**.
Supported families include integer Affine, integer ReLU MLP/deep MLP, exact
single/batch linear gradients, fixed-point requantization, fixed-point
Affine-ReLU-Affine value/threshold/exact-output equality, and clean noncontiguous
product-domain membership. Unsupported semantics are rejected rather than
silently approximated. Arbitrary-depth fixed-point composition, optimized state
preparation, and larger-width synthesis remain open.

### Thesis C — end-to-end quantum advantage or a sharp no-advantage boundary

Core claim:

> At matched reconstruction success, target equivalence, candidate prior, and
> preprocessing rights, a structure-preserving coherent reconstruction pipeline
> has lower total cost than the strongest specialized classical solver in a
> robust nonempty regime—or the project proves why no such regime exists.

Required evidence:

- a real identifiable leakage task not dominated by an analytic decoder;
- a clean non-enumerative oracle and explicit domain contract;
- a marked-count-independent search protocol or a priced counting procedure;
- state preparation, inverse calls, diffusion, fault-tolerant synthesis,
  verification, measurement, and decoding costs;
- strongest analytic, algebraic, branch-and-bound, SAT/SMT, MIP,
  meet-in-the-middle, and optimized continuous baselines where applicable;
- a strict break-even region with uncertainty analysis;
- scaling trends showing the region is not a one-point artifact.

Current state: **not established**. Q-RECON must not claim Thesis C yet.

## 3. Readiness matrix

| Area | Submission gate | Current status | Blocking work |
|---|---|---|---|
| problem definition | prior, observation, domain, target equivalence fixed before experiments | green | keep all declarations machine-readable |
| information theory | exact/noisy Bayes optimum and data-processing limits | green | extend to the final continuous/noisy task |
| local identifiability | Jacobian rank used only as a local sufficient certificate | green | calibrated perturbation and stability experiments |
| global identifiability | collision theorem or exhaustive finite fibre certificate | green for several linear/finite tasks | prove or certify the final nonlinear leakage task |
| access assumptions | C/W/Q access distinguished explicitly | green | justify construction and attacker rights in final task |
| finite oracle correctness | value/verifier/phase semantics, inverse, and phase tests | green | retain as an independent regression backend |
| non-circular construction | structural compiler does not enumerate global candidates | green for structural backends | run construction audit for every final configuration |
| integer arithmetic compiler | Affine, equality, ReLU MLP, deep MLP | green within declared semantics | optimize depth/ancilla and validate larger widths |
| fixed-point compiler | requantization, Affine, two-layer ReLU MLP exact equality | yellow-green | arbitrary-depth composition, broader overflow/precision study |
| structured candidate domain | clean finite product-domain membership and composition | yellow-green | interval/grammar priors and priced structured state preparation |
| training-leakage compiler | exact single and ordered-batch linear gradients | green within linear scope | nonlinear, partial, aggregated, quantized, and noisy leakages |
| known-`K` query model | exact standard Grover curve | green as reference only | never use as deployed cost unless `K` is public or priced |
| unknown-`K` quantum search | finite BBHT schedule and all-positive-`K` success certificate | yellow-green | zero-solution handling, approximate/noisy execution, hardware study |
| classical baselines | analytic, branch-and-bound, additive MITM, exact Z3 SMT | yellow-green | MIP/algebraic/optimized continuous suite on the final task |
| end-to-end cost | setup-aware known/unknown-`K` equations and robust `K` envelope | yellow-green | instantiate one common measured unit and uncertainty ranges |
| nonempty advantage region | strict `C_Q < C_C` at matched success | red | final empirical/theoretical contribution |
| real data | existing GIFT-Eval/Community Forensics gradient inversion | yellow | connect coherent verifier to realistic structured leakage priors |
| statistical quality | deterministic bootstrap/Wilson intervals, balanced-seed checks, failure accounting, scaling fits, environment manifests | yellow-green | pinned runners, repeated within-seed timing, paired ratios, real-data matrix |
| reproducibility | package, examples, multi-version CI, solver/quantum jobs | green | freeze environments and archive final artifacts |
| external validity | multiple models, modalities, defenses, access conditions | red-yellow | broaden only after the primary thesis is fixed |

## 4. Hard rejection rules for internal review

A draft is not submission-ready if any rule below is violated.

### R1 — collision blindness

Reconstruction quality is reported without identifying whether the observation
fibre contains multiple non-equivalent candidates.

### R2 — local/global conflation

Full-column Jacobian rank at one sample is presented as global uniqueness, or
rank deficiency is presented as a general proof of non-identifiability.

### R3 — classical API equals quantum oracle

A normal prediction or gradient API is treated as coherent superposition access
at no construction cost.

### R4 — enumerative oracle circularity

The compiler evaluates every global candidate, stores a table/index, then omits
that setup or the equivalent classical preimage lookup from cost comparison.

### R5 — query count equals runtime

Classical queries and quantum calls are compared without one cost unit and
without compilation, inverse, state preparation, diffusion, precision,
verification, measurement, and decoding.

### R6 — weak classical opponent

Quantum search is compared only with random/exhaustive search despite applicable
analytic, algebraic, branch-and-bound, MITM, SAT/SMT, MIP, or gradient solvers.

### R7 — objective substitution

Low verifier loss or output consistency is called recovery of the original
sample without exact/equivalence-class evaluation.

### R8 — hidden arithmetic mismatch

Overflow, saturation, rounding, signedness, activation, or quantization differs
between the reference evaluator, classical solver, and coherent circuit.

### R9 — single-point break-even

An advantage relies on one selected parameter point, omits uncertainty, or
vanishes under reasonable cost conversions.

### R10 — trivial identifiable task

The selected task has a cheap analytic decoder, while Grover is compared with an
intentionally weaker method.

### R11 — hidden known-`K` oracle

The deployed quantum plan chooses iterations from the true marked count without
showing that `K` is public or including counting/certification cost.

### R12 — candidate-domain mismatch

The classical solver searches a structured subset while quantum success is
priced on a different population or an unpriced ideal domain state.

### R13 — incomplete solver called exact

A timeout, solution limit, or `unknown` SMT result is reported as a complete
fibre certificate.

### R14 — pseudoreplication or hidden failures

The same `(configuration, seed)` is counted as multiple independent samples,
solver timeouts are removed from the denominator, skipped checks are called
successes, or only favorable configurations are retained.

### R15 — shared-runner timing claim

GitHub Actions wall-clock measurements are presented as hardware-comparable
performance without pinned runners, warmups, within-instance repetitions,
affinity controls, and a declared common cost conversion.

## 5. Minimum experiment package

The final paper should have one primary task and at most two supporting tasks.
The artifact must generate the following automatically.

### 5.1 Information report

- candidate population, prior, and structured domain;
- target equivalence and leakage type;
- exact/estimated fibre distribution;
- Bayes exact and equivalence-class ceilings;
- local singular spectrum and calibrated noise stability where applicable.

### 5.2 Compiler report

- reference-semantics hash and construction audit;
- logical qubits and peak clean ancillas;
- X, CNOT, Toffoli, T-count, T-depth, and logical depth;
- forward, inverse, domain, phase, and verification calls;
- rounding/overflow/precision contract;
- exhaustive or property-based correctness coverage.

### 5.3 Search report

- known-`K` reference curve and deployed unknown-`K` protocol;
- existence/zero-solution assumption;
- target and certified minimum success;
- expected and worst-case phase and verification calls;
- ideal/approximate/noisy success curves;
- extrapolation assumptions separated from measurements.

### 5.4 Classical report

- analytic decoder when available;
- optimized brute force and branch-and-bound;
- meet-in-the-middle/dynamic programming for additive structure;
- complete SAT/SMT or MIP formulation for discrete networks;
- multi-start gradient inversion for continuous priors;
- preprocessing, time, memory, completion status, and scaling.

### 5.5 Statistical report

- predeclared configurations and balanced independent seed sets;
- no duplicate `(configuration, seed)` observations;
- Wilson intervals for success, completion, and exact-agreement rates;
- deterministic bootstrap intervals for scalar means and medians;
- raw per-instance results, failures, timeouts, and skipped checks;
- at least three population scales, preferably five or more;
- paired comparisons on the same generated instances;
- pinned hardware and repeated within-seed measurements for runtime claims;
- machine-readable environment and package manifest;
- exploratory and confirmatory seeds kept separate.

### 5.6 End-to-end report

For every method report

\[
C=C_{setup}+M C_{variable}
\]

in one declared unit, with sensitivity sweeps over:

- amortized instances `M`;
- logical/T-state cost and error-correction assumptions;
- state/domain preparation and reflection cost;
- oracle precision and operational error;
- population, fibre size, and unknown-`K` stopping rule;
- classical memory and timeout budgets.

An advantage region is accepted internally only if it remains nonempty under a
predeclared robustness envelope and against the best completed classical solver.

## 6. Recommended paper shape

A coherent submission should contain four tightly connected contributions.

1. **Identifiability theorem.** A new collision or injectivity result for the
   final training-leakage setting.
2. **Compiler theorem.** A clean non-enumerative reversible compiler for the same
   verifier and structured domain, with exact arithmetic and resources.
3. **Classical boundary.** A theorem or strongest specialized solver suite that
   prevents an artificial quantum comparison.
4. **End-to-end phase diagram.** Information limit, classical cost, quantum cost,
   precision, and uncertainty on the same axes.

Everything else should support one of these four claims.

## 7. Current verdict

Q-RECON has moved well beyond an early CCF-C-style prototype. It now has original
identifiability and query-optimality results, executable information bounds,
multiple clean reversible compiler families, structured-domain composition,
construction-circularity audits, known/unknown-`K` search models, robust cost
envelopes, analytic/MITM/branch-and-bound/Z3 classical boundaries, balanced-seed
statistical reports, and extensive automated verification.

It is still not honest to label the repository itself “CCF-A achieved.” The
remaining gap is concentrated:

- select one nontrivial identifiable real leakage task;
- extend the compiler and strongest solver suite to that exact task;
- run pinned-hardware, repeated-timing, multi-scale real-data experiments;
- calibrate one common end-to-end cost model with uncertainty;
- demonstrate a robust nonempty advantage region, or prove a sharp no-advantage
  boundary if the region is empty.

A rigorous negative result can be top-tier: proving that identifiability,
structured classical inversion, domain preparation, oracle construction,
statistical uncertainty, or fault-tolerant cost eliminates an apparent quantum
advantage is a valid central contribution.
