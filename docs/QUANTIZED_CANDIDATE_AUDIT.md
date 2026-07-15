# Quantized Candidate Auditing: Information Loss Before Coherent Search

## 1. Motivation

A coherent reconstruction oracle ultimately consumes finite bit strings, while
real candidate records usually begin as floating-point tensors. The conversion

\[
X\longmapsto Q_b(X)
\]

is part of the observation and candidate model, not an implementation detail.
Rounding, saturation, and finite precision can merge distinct real records before
any classical or quantum search begins. A correct end-to-end evaluation must
therefore audit two separate effects:

1. **information loss:** how many originally distinct candidates become the same
   quantized word;
2. **loading cost:** how the resulting finite candidate table is supplied
   coherently.

The executable entry point is
`qrecon.benchmarks.tensor_candidates.audit_quantized_candidate_tensor`.

## 2. Declared fixed-point semantics

For `f` fractional bits, the scale is

\[
s=2^f.
\]

Each real value is converted to the integer code

\[
q(x)=\operatorname{RoundAway}(sx),
\]

where nearest-integer ties are resolved away from zero. For example,

\[
q(0.5)=1,
\qquad
q(-0.5)=-1
\]

when `f=0`.

For a signed `b`-bit word, valid codes are

\[
-2^{b-1}\le q\le 2^{b-1}-1.
\]

For an unsigned word, they are

\[
0\le q\le 2^b-1.
\]

Overflow behavior must be declared:

- `raise`: reject the candidate set if any rounded code is outside the range;
- `saturate`: clamp to the nearest endpoint and report every saturated value.

Modular wraparound is never used silently. The reference evaluator, classical
solver, and reversible circuit must share exactly the same semantics.

## 3. Quantization as deterministic post-processing

Let the private candidate index be `I`, let the original tensor be

\[
X=T_I,
\]

and let

\[
Z=Q_b(X)
\]

be its quantized representation. Then

\[
I\longrightarrow X\longrightarrow Z
\]

is a Markov chain with deterministic second map.

### Theorem 1 — quantization cannot improve exact-index recovery

For any prior on candidate indices,

\[
P^*_{\mathrm{guess}}(I\mid Z)
\le
P^*_{\mathrm{guess}}(I\mid X).
\]

### Proof

`Z` is obtained solely by deterministic post-processing of `X`. The Bayes
guessing probability obeys the data-processing inequality already proved in
`THEORY_FOUNDATIONS.md`. Therefore post-processing cannot increase the optimum.
\(\square\)

This statement holds for both classical and quantum post-processing of the same
released quantized word. A quantum optimizer cannot recreate distinctions erased
by quantization.

## 4. Uniform-prior exact formula

Suppose `N` candidate indices are uniformly likely. Partition the indices by
identical original tensors and by identical quantized tensors. Let

\[
U_X=|\{T_i:i\in[N]\}|,
\qquad
U_Z=|\{Q_b(T_i):i\in[N]\}|.
\]

Because quantization cannot split an original equality class,

\[
U_Z\le U_X\le N.
\]

### Corollary 1 — exact-index Bayes ceilings

Under the uniform prior,

\[
P^*_{\mathrm{guess}}(I\mid X)=\frac{U_X}{N},
\]

and

\[
P^*_{\mathrm{guess}}(I\mid Z)=\frac{U_Z}{N}.
\]

The number of quantization-induced mergers is reported as

\[
U_X-U_Z.
\]

This is an exact information ceiling, not an attack success estimate. If two
non-equivalent records quantize to the same word, no verifier that sees only that
word can identify which record was original.

## 5. Distortion and saturation metrics

The audit reconstructs each code as

\[
\widehat x=q/s
\]

and reports

\[
\operatorname{MSE}
=rac1D\sum_j(\widehat x_j-x_j)^2
\]

and

\[
\|\widehat X-X\|_\infty.
\]

It also reports the number of values whose rounded code exceeded the declared
range. These metrics answer different questions:

- small MSE does not imply that candidate identities remain distinct;
- no collisions does not imply low numerical distortion;
- saturation can create large fibres even when ordinary rounding would not;
- perceptual similarity does not replace exact/equivalence-class evaluation.

Both fibre statistics and distortion must be included.

## 6. Source and representation identity

The report hashes:

- the complete tensor shape;
- the NumPy dtype descriptor;
- the tensor bytes in candidate order;
- a versioned schema identifier.

Changing row order, content, shape, or dtype changes the source SHA256. This
prevents candidate tables produced by different preprocessing pipelines from
being silently combined.

The quantized loading report separately hashes the exact integer-code table and
records its word width and candidate count. The source hash and quantized-table
hash identify two distinct artifacts and should both be archived.

## 7. Integration with coherent loading

After quantization, the integer tensor is passed to the explicit empirical-table
loading audit. The combined report contains:

- original and quantized unique-candidate counts;
- exact-index Bayes ceilings before and after quantization;
- signedness, precision, scale, rounding, and overflow policy;
- saturation and distortion metrics;
- quantized candidate-table hash;
- explicit and deduplicated table-description sizes;
- compiler input-reading lower bound;
- typical bounded-arity circuit-counting lower bound;
- literal minterm resources when the table is small enough;
- a skip reason when literal synthesis would be misleadingly large.

Thus an experiment cannot claim a real-data Grover speedup while omitting either
quantization collisions or coherent table-loading setup.

## 8. Target equivalence

Exact index equality is deliberately strict. In some applications, multiple
records are acceptable under a declared relation, such as:

- permutation of a training batch;
- graph isomorphism;
- tokenization equivalence;
- application-defined tolerance or semantic class.

The relation must be fixed before inspecting reconstruction results. Quantized
word equality is not automatically the target relation. If distinct original
records collide under quantization but belong to the same declared target class,
report both exact-index and equivalence-class Bayes ceilings. If they are not
equivalent, the collision is an irreducible failure of the chosen representation.

## 9. Real-data protocol

For each real candidate source:

1. archive the raw/preprocessed tensor and source SHA256;
2. declare candidate-axis semantics and target equivalence;
3. sweep a predeclared set of bit widths and fractional precisions;
4. report overflow failures separately from saturation runs;
5. report original and quantized fibre distributions;
6. report MSE, maximum absolute error, and saturation count;
7. feed each quantized table to the loading audit;
8. use the same fixed-point semantics in branch-and-bound, SMT/MIP, and the
   coherent compiler;
9. compare end-to-end cost only at matched accepted-solution sets and success;
10. include precision as an axis in the final advantage or no-advantage phase
    diagram.

A precision selected after observing favorable collisions or gate counts is a
form of model selection and must not be treated as confirmatory evidence.

## 10. Internal rejection rules

A result is not submission-ready if any of the following occurs:

- floating-point records are cast to integers without a declared rounding rule;
- overflow wraps silently;
- saturation is used but saturation count is omitted;
- only distortion is reported and candidate collisions are ignored;
- the classical solver sees the original tensor while the quantum verifier sees
  the quantized tensor, or vice versa;
- quantized duplicates are counted as separately identifiable marked items;
- precision is tuned on confirmatory seeds;
- loading costs are omitted after quantization.

## 11. Theorem-to-code mapping

| Object or result | Executable implementation |
|---|---|
| deterministic fixed-point conversion | `audit_quantized_candidate_tensor` |
| source artifact hash | `QuantizedCandidateAudit.source_sha256` |
| uniform exact-index ceilings | `exact_index_bayes_success_*` properties |
| quantization-induced collisions | `quantization_induced_collision_count` |
| distortion and saturation | `quantization_mse`, `maximum_absolute_error`, `saturation_count` |
| coherent loading audit | nested `EmpiricalCandidateLoadingReport` |

Focused tests cover induced collisions, positive and negative half ties,
raise-versus-saturate behavior, dtype-sensitive hashing, nonfinite inputs, and
integer-width limits.

## 12. Next milestone

The next integration step is to run this audit on versioned GIFT-Eval and
Community Forensics candidate manifests, connect the selected precision to the
fixed-point model/leakage compiler, and generate a joint phase diagram over:

- information ceiling;
- candidate population and fibre sizes;
- classical solver completion and cost;
- coherent loading and verifier resources;
- unknown-`K` search cost;
- fault-tolerant assumptions;
- quantization precision and saturation policy.

That diagram may reveal a robust advantage region or a rigorous no-advantage
boundary. Either outcome is stronger than hiding the quantization step.
