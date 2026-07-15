# Manifest-Driven Repeated Benchmarks

## 1. Purpose

The fixed-point benchmark matrix originally produced one elapsed measurement for
one generated instance. That is sufficient for correctness regression but not for
a publication-level runtime or scaling claim. This module adds a versioned,
hashable experiment contract and records warmups, repeated measurements, errors,
and within-seed timing distributions before aggregating across independent
seeds.

The implementation is in `qrecon.benchmarks.manifest`.

This artifact does **not** make shared GitHub runner timings comparable to
fault-tolerant quantum hardware. It establishes the data discipline required
before a pinned-hardware study.

## 2. Canonical manifest

`FixedPointMLPBenchmarkManifest` fixes:

- the complete ordered configuration list;
- the independent seed set;
- warmup runs and measured repeats per seed;
- whether the exact Z3 baseline is enabled;
- Z3 timeout;
- target search success;
- BBHT growth, attempts per stage, and optional stage limit;
- a schema version.

The manifest is encoded as sorted compact JSON and identified by

\[
H=\operatorname{SHA256}(\operatorname{canonical\_json}(M)).
\]

Every run record carries `H`. Changing one seed, arithmetic configuration,
solver limit, or search parameter changes the identifier. This prevents a table
from silently mixing results produced under different experiment contracts.

The JSON loader rejects unsupported schema versions and converts serialized
candidate domains back to immutable tuples.

## 3. Experimental hierarchy

The execution hierarchy is:

\[
\text{configuration}
\rightarrow
\text{seed}
\rightarrow
\text{warmup/measurement repeat}.
\]

A seed determines the model, private candidate, and exact observation. Repeated
measurements of the same seed quantify timing variation; they are not independent
problem instances. Consequently, repeated runs are first collapsed to one
within-seed statistic. The current default is the median.

Only after this collapse are confidence intervals computed across seeds. This
avoids treating repeated timings from one generated instance as independent
samples.

## 4. Raw run record

Every attempted run produces a `BenchmarkRunRecord` containing:

- manifest SHA256;
- configuration index and seed;
- phase (`warmup` or `measurement`);
- repeat index;
- status (`success` or `error`);
- total wall-clock duration;
- the complete benchmark result for successful measured runs;
- exception type, message, and deterministic exception hash for failures.

Warmup results are intentionally not retained as measured observations. Warmup
failures remain visible and are counted separately.

Unexpected exceptions are recorded rather than silently removing an unfavorable
instance. `continue_on_error=False` remains available for debugging workflows
that should stop at the first failure.

## 5. Semantic consistency across repeats

Repeated executions of the same `(configuration, seed)` must describe the same
mathematical instance and produce the same non-timing outputs. Before timing
collapse, Q-RECON hashes a normalized result with elapsed-time fields removed.

A mismatch means the benchmark is nondeterministic, the environment changed
semantic behavior, or a result field incorrectly depends on runtime state. Such a
cell is rejected rather than averaged.

This check covers, among other fields:

- generated private input and target codes;
- fibre/solution count;
- identifiability flag;
- branch-and-bound report;
- coherent-oracle resources and marked set agreement;
- BBHT certificate;
- Z3 completion and solution set.

## 6. Complete and incomplete cells

A complete cell has exactly `repeats_per_seed` successful measured runs. By
default, hierarchical aggregation requires every cell to be complete.

The raw `ManifestExecution` is still serializable when one or more cells fail. It
reports:

- expected and observed measured runs;
- measured successes and failures;
- warmup failures;
- raw error records.

For exploratory fault-tolerance analysis, callers may collapse cells with at
least one successful repeat by setting `require_complete_cells=False`. Such a
report must preserve the missing-run counts and must not be presented as the
confirmatory matrix.

## 7. Hierarchical report

`ManifestStatisticalReport` contains:

- manifest identifier;
- expected, observed, successful, and failed measurement counts;
- Wilson interval for run completion;
- complete and incomplete cell counts;
- semantic-consistency failures;
- scalar summaries of per-cell median branch-and-bound, oracle-construction, and
  total times;
- the existing multi-seed matrix summary calculated from one collapsed result per
  cell.

This is a two-level descriptive analysis:

1. median across repeats within each generated instance;
2. bootstrap/Wilson uncertainty across independent seeds.

A final runtime comparison should extend this with paired classical/quantum cost
ratios and a hierarchical or cluster bootstrap that preserves configuration and
seed structure.

## 8. Reproducibility contract

The complete machine-readable artifact should archive:

- manifest JSON and SHA256;
- every raw run record;
- collapsed per-cell results;
- statistical summary;
- environment manifest;
- Git commit, container digest, runner specification, and dependency lockfile.

The manifest identifies intended inputs, not the execution environment. Two runs
with the same manifest but different environment hashes should be analyzed as
separate execution blocks.

## 9. CI and reference profiles

The lightweight smoke report is:

```bash
python examples/fixed_point_manifest_report.py
```

The exact SMT variant is:

```bash
python examples/fixed_point_manifest_report.py --z3
```

The larger reference profile is:

```bash
python examples/fixed_point_manifest_report.py --publication --z3
```

The CI profile uses two seeds, three candidate scales, one warmup, and two
measured repeats to validate the pipeline. It is explicitly labelled
`ci-smoke` and is not publication-level evidence.

The publication-oriented reference profile uses 20 seeds, two warmups, seven
measured repeats, and 5,000 bootstrap samples. It is still only a template: a
final study must load a frozen manifest file, pin hardware, define timeouts and
memory limits, and justify statistical power for its primary effect.

## 10. Remaining systems work

The next execution milestone is a resumable manifest runner with:

- one-record-per-file or append-safe long-form storage;
- sharding by `(configuration, seed)`;
- process isolation and hard memory/time limits;
- retry policy that does not overwrite failed attempts;
- container and hardware fingerprints;
- paired classical/quantum cost components;
- real-data candidate priors;
- figure-ready exports and sensitivity envelopes.

These additions are required before using runtime measurements to support an
end-to-end advantage or no-advantage claim.
