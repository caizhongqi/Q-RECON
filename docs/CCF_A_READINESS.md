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

Current state: **strong foundation with one real-candidate negative boundary**.
Aggregate private-label linear regression has an explicit continuous collision
family. Full exact single-record biased linear gradients have a proved
nonzero-residual analytic decoder and a zero-residual non-identifiable fibre. For
fixed known targets, the complete biased-linear MSE gradient-oracle fibre is
characterized by the target-stabilizer orbit, with an exact orbit-dimension
formula, identifiable quotient, optimal packed query plan, and matching physical
lower bound. The versioned two-record empirical benchmark adds exact quantized
pair fibres and the following dichotomy: if the target fibre has size greater
than one, original-pair recovery is information-limited; if it has size one,
complete vector two-sum already matches the ideal Grover verifier-call exponent.
A broader nonlinear or partial-leakage theorem remains open.

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
single/batch linear gradients, fixed-point requantization, arbitrary-depth
fixed-point Affine/ReLU value, threshold and exact-output equality oracles, and
clean noncontiguous product-domain membership. The arbitrary-depth compiler has
layerwise reachability certificates, retained hidden-register cleanup, one
shared arithmetic-work register, and exact layer multiplicities
`(2, ..., 2, 1)`. Unsupported semantics are rejected rather than silently
approximated. Optimized state preparation, broader overflow/precision studies,
and larger-width synthesis remain open.

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

Current state: **a sharp no-query-exponent boundary is established for additive
batch size two; practical end-to-end advantage is not established**. Versioned
GIFT-Eval and Community Forensics calibration manifests, selected-source hashes,
quantization/fibre reports, complete vector two-sum, an exact coherent predicate
reference, staged unknown-`K` certificates, and explicit loading floors now share
one report schema. For a target fibre of size `K>1`, exact original-pair recovery
has conditional uniform-prior ceiling `1/K`. For `K=1`, vector two-sum is expected
`O(N)` time and memory while Grover over `choose(N,2)` pairs uses `Theta(N)`
verifier calls. The truth-table predicate is retained only as an enumerative
semantic reference and is explicitly rejected as a scalable empirical-data
oracle. A positive result still requires a non-additive identifiable task with a
non-enumerative coherent access construction, or the negative theorem must be
generalized substantially.

## 3. Readiness matrix

| Area | Submission gate | Current status | Blocking work |
|---|---|---|---|
| problem definition | prior, observation, domain, target equivalence fixed before experiments | green | keep all declarations machine-readable |
| information theory | exact/noisy Bayes optimum and data-processing limits | green | extend to the final continuous/noisy task |
| local identifiability | Jacobian rank used only as a local sufficient certificate | green | calibrated perturbation and stability experiments |
| global identifiability | collision theorem or exhaustive finite fibre certificate | green for linear/finite and real batch-two calibration tasks | prove or certify the final nonlinear leakage task |
| access assumptions | C/W/Q access distinguished explicitly | green | fix explicit compilation, physical QRAM, or succinct-generator rights for the final task |
| finite oracle correctness | value/verifier/phase semantics, inverse, and phase tests | green | retain as an independent regression backend |
| non-circular construction | structural compiler does not enumerate global candidates | green for structural backends | the empirical pair predicate remains reference-only; build non-enumerative access for any positive claim |
| integer arithmetic compiler | Affine, equality, ReLU MLP, deep MLP | green within declared semantics | optimize depth/ancilla and validate larger widths |
| fixed-point compiler | requantization and arbitrary-depth Affine/ReLU value/threshold/equality | green within declared semantics | broader overflow/precision study and larger-width synthesis |
| structured candidate domain | clean finite product-domain membership and composition | yellow-green | interval/grammar priors and priced structured state preparation |
| empirical candidate access | explicit-table `Nw` compiler probe lower bound, typical circuit lower bound, minterm upper bound, no-advantage workload certificate | green for explicit compilation | instantiate physical QRAM or succinct-generator alternatives separately when claimed |
| versioned real candidates | immutable revision, canonical manifest, selected tensor SHA256, failed-precision preservation | green at calibration protocol level | run and lock publication hashes for every final real-data artifact |
| training-leakage compiler | exact single and ordered-batch linear gradients | green within linear scope | nonlinear, partial, aggregated, quantized, and noisy leakages |
| real batch-two fibre | exact unordered fibres, Bayes ceilings, complete vector two-sum, coherent reference agreement | green | execute pinned real manifests and archive confirmatory reports |
| known-`K` query model | exact standard Grover curve | green as reference only | never use as deployed cost unless `K` is public or priced |
| unknown-`K` quantum search | finite BBHT schedules, all-positive-`K` success certificates, staged execution, and one-sided zero-solution decision | green in exact-verifier finite semantics | approximate/noisy execution and hardware study |
| classical baselines | analytic, branch-and-bound, additive MITM/two-sum, exact Z3 SMT | green for current additive/two-layer scopes | MIP/algebraic/optimized continuous suite on the final nonlinear task |
| end-to-end cost | setup-aware known/unknown-`K` equations, robust `K` envelope, explicit-table no-advantage regions | yellow-green | instantiate one common measured unit and uncertainty ranges |
| nonempty advantage region | strict `C_Q < C_C` at matched success | red | final positive empirical/theoretical contribution |
| no-advantage boundary | information/query dichotomy for real-candidate additive batch size two | yellow-green | run pinned real manifests, add multi-scale statistics, generalize beyond one leakage family |
| statistical quality | deterministic bootstrap/Wilson intervals, balanced-seed checks, failure accounting, scaling fits, environment manifests | yellow-green | pinned runners, repeated within-seed timing, paired ratios, real-data matrix |
| reproducibility | package, examples, multi-version CI, solver/quantum jobs | green | freeze final environments and archive publication artifacts |
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
The real batch-two truth-table predicate is therefore reference-only.

### R5 — query count equals runtime

Classical queries and quantum calls are compared without one cost unit and
without compilation, inverse, state preparation, diffusion, precision,
verification, measurement, and decoding.

### R6 — weak classical opponent

Quantum search is compared only with random/exhaustive search despite applicable
analytic, algebraic, branch-and-bound, MITM/two-sum, SAT/SMT, MIP, or gradient
solvers. In particular, enumerating all record pairs is not an admissible
classical baseline for additive batch size two.

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

The selected task has a cheap analytic decoder or vector two-sum inversion, while
Grover is compared with an intentionally weaker method.

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
successes, failed precisions disappear, or only favorable configurations are
retained.

### R15 — shared-runner timing claim

GitHub Actions wall-clock measurements are presented as hardware-comparable
performance without pinned runners, warmups, within-instance repetitions,
affinity controls, and a declared common cost conversion.

### R16 — unversioned real candidate artifact

A real-data result omits an immutable dataset revision, the canonical manifest
hash, selected feature/target tensor hash, or the exact quantized table hash.

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
- vector two-sum or meet-in-the-middle for additive structure;
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
A no-advantage boundary is accepted only if it compares a quantum lower bound or
complete cost model against a valid classical upper bound on the same task.

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
envelopes, analytic/MITM/two-sum/branch-and-bound/Z3 classical boundaries,
versioned real-candidate manifests, balanced-seed statistical reports, and
extensive automated verification.

The new real-candidate batch-two path supplies a defensible negative core:

- for nonunique aggregate-gradient fibres, exact original-pair reconstruction is
  information-limited;
- for unique fibres, complete vector two-sum already has expected linear time and
  matches the ideal Grover verifier-call exponent;
- explicit empirical-table loading and the enumerative predicate construction
  cannot be treated as free.

It is still not honest to label the repository itself “CCF-A achieved.” The
remaining gap is now more focused:

- execute the revision-pinned GIFT-Eval and Community Forensics calibration
  manifests, lock selected-source hashes, and archive confirmatory reports;
- run pinned-hardware, repeated-timing, multi-scale experiments;
- either generalize the no-advantage theorem to broader nonlinear/partial/noisy
  leakage settings, or identify a non-additive globally identifiable task with a
  non-enumerative coherent access construction;
- calibrate one common end-to-end cost model with uncertainty;
- demonstrate a robust nonempty advantage region, or a substantially generalized
  sharp no-advantage boundary.

A rigorous negative result can be top-tier: proving that identifiability,
structure-aware classical inversion, data loading, oracle construction,
statistical uncertainty, or fault-tolerant cost eliminates an apparent quantum
advantage is a valid central contribution.
