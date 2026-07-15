# Statistical Benchmark and Evidence Protocol

## 1. Purpose and claim boundary

This protocol turns the Q-RECON fixed-point benchmark matrix into a reproducible
statistical artifact. It addresses a common failure mode in systems and security
papers: reporting one favorable random instance, one wall-clock timing, or one
solver outcome as if it established a stable scaling law.

The protocol is a **necessary evidence gate**, not a conference-acceptance
predictor. Passing it does not establish end-to-end quantum advantage. In
particular, wall-clock timings produced by shared GitHub runners remain
reproducibility diagnostics unless the runner, warmup policy, repetition count,
affinity, and resource-to-cost conversion are fixed in advance.

The executable implementation is
`qrecon.benchmarks.summarize_fixed_point_mlp_benchmark_matrix`.

## 2. Experimental unit

The independent experimental unit is

\[
(\text{configuration},\text{seed}).
\]

A configuration fixes at least:

- input dimension and candidate domain;
- hidden width and output dimension;
- input, weight, bias, hidden, and output formats;
- rounding, overflow, activation, and equality semantics;
- target-success requirement;
- classical solver limits and timeouts;
- oracle verification and enumeration limits.

A seed deterministically generates the model parameters, private record, and
corresponding exact output. Re-running the same `(configuration, seed)` does not
create a second independent sample. The statistical aggregator therefore rejects
duplicate seeds within a configuration to prevent accidental pseudoreplication.

For matched comparisons, branch-and-bound, SMT, coherent-verifier evaluation,
and search accounting must operate on the **same generated instance**.

## 3. Balanced design

The preferred matrix uses the same seed set for every configuration. A balanced
design avoids conflating scale effects with a changing instance distribution.
The report records whether the matrix is balanced and does not pass its internal
quality gate when seed sets differ.

The default internal minimum is:

- at least 10 independent seeds per configuration;
- at least 3 distinct candidate-population scales;
- zero classical/coherent solution-set mismatches;
- no incomplete or mismatching SMT runs among the runs for which SMT is claimed.

For a final paper, the recommended target is at least 20 seeds and 5 or more
scales whenever computationally feasible. A power or precision analysis should
justify smaller matrices.

The lightweight CI example intentionally uses smaller thresholds only to verify
that the reporting pipeline executes. Its output is labelled `ci-smoke` and must
not be cited as publication-level statistical evidence.

## 4. Outcome classes

### 4.1 Bernoulli outcomes

The following are Bernoulli outcomes and are reported with Wilson score
intervals:

- whether the private record is uniquely identifiable on the declared domain;
- whether branch-and-bound and the coherent oracle produce identical solution
  sets;
- whether exhaustive basis-permutation verification succeeds when attempted;
- whether an SMT run terminates with a complete solution set;
- whether a completed SMT solution set equals the branch-and-bound solution set.

A rate of `1.0` with a finite sample does not imply certainty. The Wilson lower
bound is therefore reported even when all observed runs succeed.

### 4.2 Scalar outcomes

The following are summarized by count, mean, sample standard deviation, median,
minimum, maximum, and deterministic percentile-bootstrap intervals for both mean
and median:

- solution/fibre size;
- branch-and-bound leaf fraction;
- branch-and-bound time;
- oracle-construction time;
- SMT time when enabled;
- numeric oracle resources, including logical qubits, ancillas, Toffoli count,
  T-count, depth, and related fields;
- certified expected BBHT phase-oracle calls when available.

The bootstrap seed is recorded in the report. Metric-specific bootstrap streams
are derived by hashing the base seed and metric label, so adding another metric
does not silently change existing confidence intervals.

### 4.3 Scaling outcomes

For positive timing summaries at at least two distinct population scales, the
report fits

\[
\log y = \alpha + \beta\log N.
\]

It records the slope `beta`, intercept, coefficient of determination, and a
normal-approximation slope interval when at least three points exist. This fit is
**descriptive**. It is not a complexity proof and should be compared with the
symbolic algorithmic and compiler bounds.

A defensible paper should show raw points and uncertainty bands rather than only
the fitted exponent.

## 5. Failure accounting

Failures must remain visible in the denominator.

- A classical/coherent solution-set mismatch is a correctness failure.
- An SMT `unknown` or timeout is a solver-completion failure, not an unsatisfiable
  result.
- A skipped exhaustive basis test is reported as skipped rather than successful.
- A missing optional SMT run is not counted as an SMT success or failure; it is
  reported through the attempted-run count.
- A benchmark exception must produce a machine-readable failed-instance record in
  publication runs rather than silently dropping the instance.

The current in-process runner raises on unexpected exceptions. The final cluster
runner should wrap each unit of work and persist status, exception type, traceback
hash, elapsed time, and retry count.

## 6. Timing protocol

GitHub Actions timings are useful for detecting regressions, not for comparing
classical and quantum hardware. Publication timing requires a pinned environment:

1. dedicated runner type, CPU model, memory, operating system, and power mode;
2. fixed package lockfile and solver version;
3. disabled or recorded turbo/frequency scaling where possible;
4. process affinity and thread-count controls;
5. predeclared warmup count;
6. repeated measurements within each generated instance;
7. a robust within-instance statistic, normally the median;
8. paired comparisons on the same instance;
9. explicit timeout and memory limits;
10. reporting of censored runs and solver `unknown` outcomes.

The current benchmark result contains one elapsed measurement per operation and
seed. Consequently, its confidence intervals quantify **between-instance**
variation plus runner noise. Before a wall-clock claim, add repeated within-seed
measurements and use a hierarchical or paired bootstrap.

## 7. Quantum/classical comparison discipline

Statistical significance does not repair an unmatched comparison. Every reported
quantum/classical contrast must also satisfy the semantic and cost gates:

- same candidate prior and target equivalence;
- same exact fixed-point model and output predicate;
- same accepted solution set;
- same target success probability;
- candidate preparation, inverse calls, diffusion, verification, measurement,
  decoding, and compilation included;
- classical preprocessing and reusable indexes included under the same
  amortization policy;
- solver timeouts and memory budgets declared;
- exact and approximate-oracle errors separated.

A quantum query reduction against exhaustive search is not meaningful when a
structure-aware algebraic, SMT, MIP, meet-in-the-middle, or analytic decoder is
available.

## 8. Multiple comparisons and model selection

The primary task, primary metric, seed set, configurations, and cost conversion
should be fixed before inspecting the final results. Exploratory sweeps may guide
engineering, but the final confirmatory matrix must be rerun from a frozen
manifest.

When many task/model/baseline combinations are tested, report the complete matrix
and control the family of confirmatory claims. Confidence intervals should not be
selectively shown only for favorable cells.

Hyperparameters for a solver or attack must be selected without access to the
held-out confirmatory seeds. Tuning and evaluation seed sets should be disjoint.

## 9. Environment manifest

Every statistical report includes:

- Python version and implementation;
- platform, machine, processor, and CPU count;
- versions of Q-RECON, NumPy, PyTorch, and Z3 when installed;
- GitHub commit and run identifiers when available;
- Python hash seed and executable path;
- an explicit timing-scope warning.

The final artifact should additionally archive the lockfile, container digest,
runner specification, raw per-run JSON, and scripts used to build every table and
figure.

## 10. Internal quality gate

The machine-readable gate currently checks:

1. balanced seed sets across configurations;
2. minimum independent seeds per configuration;
3. minimum number of candidate-population scales;
4. exact branch-and-bound/coherent solution-set agreement on every instance;
5. completion and agreement of every attempted SMT run.

The gate deliberately does **not** certify:

- practical quantum advantage;
- sufficient external validity;
- adequate power for a particular effect size;
- stable hardware timing;
- correctness of an unsupported arithmetic semantic;
- novelty or conference acceptance.

Passing is necessary before promoting a benchmark matrix into a central paper
result, but the CCF-A readiness gates remain stricter.

## 11. Executable reports

A lightweight CI report is generated by:

```bash
python examples/fixed_point_statistical_report.py
```

The optional exact SMT cross-check is:

```bash
python examples/fixed_point_statistical_report.py --z3
```

A larger predeclared seed/bootstrap matrix is:

```bash
python examples/fixed_point_statistical_report.py --publication --z3
```

The publication flag is still only a reference profile. Final experiments should
be driven by a versioned manifest rather than editing the example script.

## 12. Required next extension

The next statistical milestone is a manifest-driven cluster runner that adds:

- repeated within-seed timings;
- timeout and memory censoring;
- paired runtime/cost ratios with paired bootstrap intervals;
- real-data candidate priors and held-out seeds;
- figure-ready long-form records;
- sensitivity envelopes over fault-tolerant and classical hardware assumptions.

Only after those records are connected to a nontrivial globally identifiable
leakage task can the project evaluate an honest end-to-end advantage region.
