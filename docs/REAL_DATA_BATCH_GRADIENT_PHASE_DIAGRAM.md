# Real-Data Two-Record Gradient Phase Diagram

## 1. Purpose

This benchmark is the first Q-RECON path that joins all of the following in one
machine-readable experiment:

1. a versioned empirical candidate source;
2. deterministic feature selection and target extraction;
3. declared fixed-point quantization and overflow semantics;
4. exact aggregate-gradient observation fibres;
5. a complete structure-aware classical solver;
6. an exact coherent predicate reference and unknown-`K` search certificate;
7. explicit candidate-loading and amortization boundaries; and
8. a theorem-level advantage or no-advantage verdict.

The primary result is deliberately a boundary rather than a presupposed speedup.
For an unordered batch of two records, additive leakage admits a complete vector
two-sum solver. This changes the correct classical comparison fundamentally.

## 2. Versioned candidate contract

A manifest fixes before execution:

- dataset name, split, revision, and candidate count;
- context/horizon or image size;
- feature-selection rule and target coordinate;
- the private target pair of candidate indices;
- all quantization precisions and overflow policies;
- linear victim weights, bias, and aggregate-gradient width;
- unknown-`K` search parameters;
- exact-enumeration and basis-verification limits;
- coherent data-access contract; and
- workloads sharing one setup.

Remote publication manifests must pin an immutable dataset revision and the
SHA256 of the selected feature/target tensor. A calibration manifest may pin the
dataset revision while leaving the selected-source hash empty; its first run
produces the hash, after which publication mode can be enabled.

The manifest itself has a canonical JSON representation and SHA256. Changing a
precision, feature coordinate, target pair, BBHT parameter, access assumption, or
source pin changes the manifest identity.

## 3. Quantized record semantics

Let the selected candidate records be

\[
(x_i,t_i),\qquad i\in\{0,\ldots,N-1\},
\]

where \(x_i\in\mathbb R^d\) and \(t_i\in\mathbb R\). For each declared fixed-point
format, Q-RECON applies deterministic nearest rounding with half ties away from
zero and either rejects or explicitly saturates overflow.

The audit reports:

- original and quantized unique-record counts;
- quantization-induced collisions;
- exact-index Bayes ceilings;
- saturation count;
- mean-square and maximum absolute distortion;
- the exact quantized table hash; and
- explicit coherent-loading lower and upper bounds.

The classical solver, gradient evaluator, and coherent predicate reference all
consume the same integer codes. No method receives the pre-quantized tensor when
another receives only the quantized one.

## 4. Additive two-record gradient observation

Fix integer weights \(w\in\mathbb Z^d\) and bias \(b\in\mathbb Z\). For one
quantized record define

\[
r_i=b+w^\top x_i-t_i
\]

and the full linear-regression gradient contribution

\[
h_i=(r_i x_i,\;r_i)\in\mathbb Z^{d+1}.
\]

For an unordered batch \(\{i,j\}\), \(i<j\), the released aggregate observation is

\[
G(i,j)=h_i+h_j.
\]

All contribution and pair-sum components must fit the declared signed gradient
word. An insufficient word width is a failed precision point, not a wrapped
observation.

The exact observation fibre of a value \(g\) is

\[
F_g=\{(i,j):i<j,\;G(i,j)=g\}.
\]

For a uniform prior over the \(\binom N2\) unordered batches, the global Bayes
exact-index ceiling is

\[
P^*_{\mathrm{guess}}
=\frac{|G(\binom{[N]}2)|}{\binom N2}.
\]

Conditioned on the private pair's observation, if its fibre size is \(K\), the
best possible exact-index success is

\[
P^*_{\mathrm{guess}}(\{I,J\}\mid G(I,J)=g)=\frac1K.
\]

This ceiling applies to classical and quantum post-processing of the same
observation.

## 5. Complete vector two-sum baseline

Build a hash table mapping each contribution vector \(h_i\) to every candidate
index having that vector. For each left index \(i\), query the complement

\[
g-h_i.
\]

Emit every matching \(j>i\). The implementation returns the complete unordered
fibre and rechecks it against exhaustive pair enumeration on every benchmark
instance.

### Theorem 1 — exactness

The hash algorithm returns exactly all unordered pairs satisfying
\(h_i+h_j=g\).

### Proof

Every emitted pair is obtained from a bucket keyed by \(g-h_i\), so its sum is
\(g\). Conversely, every satisfying pair has \(h_j=g-h_i\); the complete hash
index contains \(j\), and the `i<j` rule emits it exactly once. \(\square\)

### Complexity

Index construction and complement lookup use expected

\[
O(N+K)
\]

time and

\[
O(N)
\]

memory, where \(K\) is the output fibre size. The executable report records the
number of indexed contributions, complement lookups, hash buckets, and emitted
solutions.

## 6. Information/query dichotomy

The pair population is

\[
P=\binom N2=\Theta(N^2).
\]

Ideal unstructured Grover search with \(K\) marked pairs has verifier-call scale

\[
\Theta\!\left(\sqrt{P/K}\right).
\]

### Theorem 2 — two-record boundary

For exact original-pair reconstruction under a uniform prior, exactly one of the
following holds.

1. **\(K>1\): information-limited.** The private pair is not identifiable from
   the released observation; conditional exact-index success is at most \(1/K\).
2. **\(K=1\): no query-exponent separation.** Complete classical vector two-sum
   uses expected \(O(N)\) time, while Grover over \(P=\Theta(N^2)\) pairs uses
   \(\Theta(N)\) verifier calls.

### Consequence

A two-record additive-gradient experiment cannot establish a new asymptotic
query exponent advantage for exact original-pair recovery. When it is
identifiable, the strongest basic classical solver already matches the Grover
exponent. When it is not identifiable, optimization cannot repair the missing
information.

This does not say all constants, memory tradeoffs, or other batch sizes are
identical. It says that a claimed quadratic improvement over an \(O(N^2)\)
classical pair scan is invalid because that scan is not the correct baseline.

## 7. Coherent predicate reference

For exact finite verification, lexicographically rank all unordered pairs and
construct a padded one-bit truth-table predicate that marks precisely the target
fibre. The reference verifies:

- the marked ranks equal the exhaustive fibre;
- forward and inverse action restore all ancillas;
- both initial target bits behave correctly;
- phase signs match the truth table; and
- the staged BBHT schedule reaches its declared success over every positive
  marked count in the padded population.

This predicate is explicitly labeled **enumerative and circular as a scalable
implementation**: building it consumes the fibre that the search is supposed to
find. It is retained only as an exact semantic and search-control reference. Its
gate count must not be presented as a scalable empirical-data oracle.

A future positive advantage claim would require either:

- a non-enumerative coherent record lookup plus reversible gradient/equality
  computation;
- a physically specified QRAM; or
- a succinct reversible generator for the candidate records.

## 8. Data-loading boundaries

Under explicit compilation, the quantized record table contains

\[
N(d+1)b
\]

description bits for `b` bits per feature/target value. An exact compiler must
inspect every description bit in the worst case. The report therefore includes:

- one-shot explicit-table input-processing boundary;
- amortized table-bit probe floors for every declared reuse count;
- typical bounded-arity circuit-counting lower bound;
- literal minterm resources when small enough; and
- a separate direct pair-predicate table boundary.

The direct pair predicate is not a substitute for a record-loading construction.
It is reported precisely to expose the cost and circularity of compiling a
precomputed fibre.

## 9. Precision phase diagram

Each declared precision is one independent point. The runner preserves all
failures, including:

- candidate quantization overflow;
- gradient-word overflow;
- exact pair population exceeding the manifest limit;
- BBHT certification limit; and
- source-hash mismatch.

For every successful point it reports:

- information ceilings and fibre histogram;
- target conditional Bayes ceiling;
- quantization distortion and saturation;
- two-sum completion and exact fibre agreement;
- coherent reference truth-table hash and resources;
- unknown-`K` success/query certificate;
- candidate-loading floors; and
- the theorem-selected verdict.

Failed precision points remain in the denominator and carry an exception hash.
Publication mode requires the versioned source hash and every semantic cross-
check to pass.

## 10. Dataset templates

The repository includes revision-pinned calibration templates for:

- GIFT-Eval forecasting windows; and
- Community Forensics Small image candidates.

The templates deliberately start with `publication_mode=false` because the
selected-source SHA256 must be produced by the first execution in the target
environment. The paper artifact must then copy that hash into the manifest,
enable publication mode, and archive the complete report.

An offline synthetic template and tests exercise the same pipeline in CI without
claiming external validity.

## 11. Theorem-to-code mapping

| Result or object | Executable implementation |
|---|---|
| versioned manifest and source hash | `RealBatchGradientManifest`, `load_real_candidate_set` |
| quantization/loading audit | `audit_quantized_candidate_tensor`, `empirical_candidate_loading_report` |
| complete vector two-sum | `solve_vector_two_sum` |
| Theorem 2 | `batch_two_recovery_dichotomy` |
| exact fibres and Bayes ceilings | `run_real_batch_gradient_phase_diagram` |
| coherent predicate reference | `TruthTableOracle` generated by the phase-diagram runner |
| staged unknown-`K` certificate | `certify_staged_bbht_uniform_success` |
| explicit loading boundary | `one_shot_explicit_table_boundary`, `amortized_explicit_table_probe_floor` |

## 12. Paper interpretation

For this primary two-record task, the scientifically defensible result is likely
a sharp no-advantage boundary:

> Exact recovery is either blocked by an observation fibre or, on uniquely
> identifiable instances, already solved in linear expected time by vector
> two-sum, matching the ideal Grover verifier-call exponent before coherent data
> loading and fault-tolerant costs are charged.

That negative result is useful because it rules out a broad class of superficial
“Grover over all record pairs” claims on real training data. It also identifies
where a genuinely different positive result would have to come from: a leakage
map without exploitable additive decomposition, a structured domain with a
succinct coherent generator, or a provable time-memory separation against the
strongest classical algorithms.
