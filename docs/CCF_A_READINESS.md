# Q-RECON CCF-A Readiness and Claim Gates

## 1. Purpose

This document is an internal acceptance contract for turning Q-RECON from a
research prototype into a submission-grade top-tier paper. It does not predict
or guarantee conference acceptance. Its purpose is stricter: no paper draft may
claim more than the strongest theorem, compiler artifact, experiment, and
classical comparison that all hold simultaneously.

The project separates five layers:

1. **information layer** — whether the released observation identifies the
   private training object at all;
2. **access layer** — whether the attacker receives classical samples, released
   internals, or a coherent reversible oracle;
3. **query layer** — how many ideal verifier calls are required;
4. **implementation layer** — how the verifier is compiled and what gates,
   qubits, precision, state preparation, measurement, and cleanup it costs;
5. **end-to-end layer** — whether the full quantum pipeline beats the strongest
   matched classical pipeline at the same success criterion.

A result at one layer never implies a result at a later layer.

## 2. Submission thesis candidates

Q-RECON currently supports three defensible thesis directions.

### Thesis A — identifiability limits of training-gradient reconstruction

Core claim:

> Exact or approximate optimization cannot recover information absent from the
> released gradient channel; for declared linear models and loss functions,
> Q-RECON gives explicit collision families, Bayes ceilings, and sharp
> identifiable/non-identifiable regimes.

Required evidence:

- formal observation channel and target equivalence;
- explicit collisions or injectivity theorem;
- Bayes-optimal exact/equivalence-class success;
- finite and continuous counterexamples;
- experiments that separate optimization failure from information failure.

Current state: **strong foundation**. Aggregate private-label linear regression
has an explicit continuous collision family. Full exact single-record biased
linear gradients have a proved nonzero-residual analytic decoder and a
zero-residual non-identifiable fibre.

### Thesis B — clean structure-preserving compilation of training-leakage oracles

Core claim:

> A declared quantized model/leakage verifier can be compiled without candidate
> enumeration into a clean reversible value/equality/phase oracle, with exact
> bit semantics, complete uncomputation, symbolic resource bounds, and exhaustive
> small-width verification.

Required evidence:

- public classical reference semantics;
- candidate-enumeration-free compiler;
- compute-copy-uncompute correctness theorem;
- clean ancilla theorem;
- exact gate/qubit/depth accounting;
- cross-check against truth-table/ANF on small domains;
- scaling beyond the enumerative backends.

Current state: **substantial but scoped**. Supported families include integer
Affine, integer ReLU MLP/deep MLP, exact single/batch linear gradients,
overflow-free fixed-point downscaling, fixed-point identity Affine, and a
two-layer fixed-point Affine-ReLU-Affine value/threshold oracle. Unsupported
semantics are rejected rather than approximated silently.

### Thesis C — end-to-end quantum advantage for reconstruction

Core claim:

> At matched reconstruction success and matched preprocessing rights, a
> structure-preserving coherent reconstruction algorithm has lower total cost
> than the strongest specialized classical solver in a nonempty measured regime.

Required evidence:

- a real identifiable leakage task not dominated by an analytic decoder;
- a clean non-enumerative oracle;
- quantum query plan and approximate-oracle success bound;
- state preparation, inverse calls, diffusion, fault-tolerant synthesis,
  measurement, and decoding costs;
- strongest classical algebraic, combinatorial, SAT/SMT, MIP, meet-in-the-middle,
  and optimized attack baselines where applicable;
- a strict break-even region with uncertainty analysis;
- scaling trends showing the region is not a one-point artifact.

Current state: **not established**. Q-RECON must not claim Thesis C yet.

## 3. Readiness matrix

| Area | Submission gate | Current status | Blocking work |
|---|---|---|---|
| problem definition | candidate prior, observation channel, target equivalence fixed before experiments | green | keep per-experiment declarations machine-readable |
| information theory | exact/noisy Bayes optimum and data-processing limits | green | extend to continuous/noisy regimes used by the final task |
| local identifiability | Jacobian rank interpreted only as a local sufficient certificate | green | add calibrated noise/stability experiments |
| global identifiability | collision theorem or exhaustive finite fibre certificate | green for several linear tasks | prove or certify the final nonlinear leakage task |
| access assumptions | C/W/Q access distinguished explicitly | green | final paper must justify Q-Access construction |
| finite oracle correctness | value/verifier/phase semantics and inverse tests | green | retain as independent regression backend |
| non-circular construction | compiler does not enumerate candidates or expose preimage table | green for structural backends | include audit in every reported result |
| integer arithmetic compiler | affine, equality, ReLU MLP, deep MLP | green within declared semantics | optimize ancilla/depth and validate larger widths |
| fixed-point compiler | downscaling, identity Affine, two-layer ReLU MLP | yellow-green | arbitrary-depth composition, upscaling/saturation policy, precision study |
| training-leakage compiler | exact single and ordered-batch linear gradients | green within declared linear scope | nonlinear/partial/noisy leakage compilers |
| quantum algorithm | exact small-space Grover and resource planner | green as a logical baseline | robust unknown-K search and approximate/noisy execution |
| classical baselines | analytic single-record and additive MITM batch baselines | yellow | SAT/SMT/MIP/algebraic/optimized inversion suite on final task |
| end-to-end cost | matched setup/query/fault-tolerant cost equations | yellow | instantiate with measured compiler and hardware assumptions |
| nonempty advantage region | strict `C_Q < C_C` at matched success | red | final empirical/theoretical contribution |
| real data | existing gradient inversion on GIFT-Eval/Community Forensics | yellow | connect the coherent verifier to realistic structured candidate priors |
| statistical quality | seeds, confidence intervals, failure rates, scaling | red-yellow | complete benchmark matrix |
| reproducibility | package, examples, CI, machine-readable reports | green | freeze environments and archive final artifacts |
| external validity | multiple models, modalities, defenses, access conditions | red-yellow | broader evaluation after final thesis is selected |

## 4. Hard rejection rules for internal review

A draft is not submission-ready if any of the following occurs.

### R1 — collision blindness

The paper reports reconstruction quality without first identifying whether the
observation fibre contains multiple non-equivalent candidates.

### R2 — local/global conflation

A full-column Jacobian rank at one sample is presented as global uniqueness, or
rank deficiency is presented as a general proof of non-identifiability.

### R3 — classical API equals quantum oracle

A normal prediction or gradient API is treated as if it supplied coherent
superposition queries at no construction cost.

### R4 — enumerative oracle circularity

The compiler evaluates every candidate, stores the truth table or ANF source
table, and then excludes that setup or classical preimage index from the cost
comparison.

### R5 — query count equals runtime

Classical queries and quantum oracle calls are compared without a common cost
unit and without compilation, inverse, state preparation, diffusion, precision,
measurement, and decoding.

### R6 — weak classical opponent

The quantum method is compared only with random or exhaustive search when the
observation admits analytic, algebraic, dynamic-programming, meet-in-the-middle,
SAT/SMT, MIP, or gradient-based structure-aware solvers.

### R7 — toy objective substitution

A low verifier loss or matching model output is reported as recovery of the
original training sample without exact/equivalence-class evaluation.

### R8 — hidden unsupported arithmetic

Overflow, saturation, rounding, signedness, activation, or quantization semantics
are left implicit or differ between the classical reference and quantum circuit.

### R9 — single-point break-even

An advantage claim relies on one manually selected parameter point, omits
uncertainty, or disappears under a reasonable alternative cost conversion.

### R10 — trivial identifiable task

The chosen task has a linear-time analytic decoder, so Grover search is presented
as an advantage over an intentionally weaker classical baseline.

## 5. Minimum experiment package for a top-tier submission

The final paper should provide one primary task and at most two supporting tasks.
For the primary task, the artifact must generate the following automatically.

### 5.1 Information report

- candidate population and prior;
- target equivalence;
- observed leakage type;
- exact/estimated number and distribution of fibres;
- Bayes exact and equivalence-class ceilings;
- local singular spectrum and noise stability where applicable.

### 5.2 Compiler report

- reference-semantics hash;
- construction-audit result;
- logical qubits and peak clean ancillas;
- X, CNOT, Toffoli, T-count, T-depth, and logical depth;
- forward, inverse, and phase-oracle calls;
- rounding/overflow/precision contract;
- exhaustive or property-based correctness coverage.

### 5.3 Search report

- marked count or unknown-K protocol;
- target success probability;
- ideal and approximate success curves;
- oracle-error accumulation bound;
- measured logical simulation for tractable instances;
- extrapolation assumptions separated from measurements.

### 5.4 Classical report

- analytic decoder when available;
- optimized brute force;
- meet-in-the-middle/dynamic programming for additive structure;
- SAT/SMT or MIP formulation for discrete networks;
- multi-start gradient inversion for continuous priors;
- preprocessing and reusable-index cost;
- time, memory, success, and failure-rate scaling.

### 5.5 End-to-end report

For every method, report

\[
C=C_{setup}+M(C_{fixed}+qC_{query})
\]

in a declared common unit, with sensitivity sweeps over:

- number of amortized instances `M`;
- T-state or logical-gate cost;
- error-correction cycle assumptions;
- state-preparation cost;
- oracle precision;
- candidate population and marked fraction;
- classical memory budget.

An advantage region is accepted internally only if it remains nonempty under a
predeclared robustness envelope.

## 6. Recommended final paper shape

A coherent submission should contain four tightly connected contributions rather
than a catalogue of unrelated modules.

1. **Identifiability theorem.** A new collision or injectivity result for the
   final training-leakage setting.
2. **Compiler theorem.** A clean non-enumerative reversible compiler for the same
   verifier, including arithmetic semantics and resource bounds.
3. **Classical boundary theorem/baseline.** An analytic lower/upper boundary or a
   strongest specialized solver that prevents artificial quantum comparisons.
4. **End-to-end phase diagram.** A measured region showing information limit,
   classical cost, quantum cost, and compiler precision on the same axes.

Everything else should support one of these four claims.

## 7. Current verdict

Q-RECON has moved beyond an early CCF-C-style prototype: it now has original
identifiability results, executable information bounds, multiple clean reversible
compiler families, construction-circularity audits, resource models, classical
MITM boundaries, and extensive automated verification.

It is not yet honest to label the repository itself “CCF-A achieved.” The
remaining gap is concentrated rather than diffuse:

- choose a nontrivial identifiable final leakage task;
- complete the strongest matched classical solver suite;
- connect the fixed-point/nonlinear compiler to that task;
- produce statistically sound real-data scaling;
- demonstrate a robust nonempty end-to-end advantage region, or publish a sharp
  no-advantage boundary if the region is empty.

A negative result can still be top-tier when it is rigorous: proving that oracle
construction, identifiability, or specialized classical inversion eliminates an
apparent quantum advantage is a valid central contribution.
