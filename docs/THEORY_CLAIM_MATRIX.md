# Q-RECON Theory Claim Matrix

The complete access-model, information, query, and cost arguments live in
[`THEORY_FOUNDATIONS.md`](THEORY_FOUNDATIONS.md), the coherent compiler contract
in [`COHERENT_ORACLE_SPEC.md`](COHERENT_ORACLE_SPEC.md), and the empirical
protocol in [`THEORY_EVALUATION_PROTOCOL.md`](THEORY_EVALUATION_PROTOCOL.md).

| Claim | Mathematical status | Executable evidence | Publication status |
|---|---|---|---|
| deterministic observation fibres impose an exact Bayes ceiling | proved for finite candidate spaces | `bayes_reconstruction_success` | ready as a foundational lemma, not a novelty claim |
| recovery modulo a declared equivalence relation has a class-posterior Bayes optimum | proved for deterministic and noisy channels | `bayes_equivalence_reconstruction_success`, `channel_bayes_equivalence_reconstruction_success` | ready as a formal evaluation rule |
| stochastic post-processing cannot increase exact guessing probability | proved by data processing | `postprocess_channel` plus tests | ready as a foundational lemma |
| full-column gradient Jacobian rank certifies local injectivity | standard inverse-function argument | `gradient_jacobian_report.full_column_rank` | ready only as a local certificate |
| rank deficiency proves non-identifiability | false in general | counterexample documented | must not be claimed |
| identical released quantum states cannot be distinguished | standard Helstrom result | `binary_helstrom_success` | ready as an information bound |
| a classical prediction API supplies coherent queries | false by access-model definition | access models documented | must not be claimed |
| standard Grover search gives a black-box quadratic query reduction | standard result under clean Q-Access | exact success/query helpers and tests | usable only with explicit oracle assumptions |
| the present VQC prior gives quantum query advantage | not established | no supporting oracle experiment | must not be claimed |
| a compiled neural verifier preserves query advantage end to end | open | compiler specification only | requires correctness, resource, precision, and cost theorems |
| batch-one biased first-layer Linear gradients reveal the raw input | proved under explicit leakage assumptions | analytic implementation and real-data verification | usable only with the stated assumptions |

## Target-equivalence Bayes theorem

Let `c(x)` denote the declared acceptable target class of candidate `x`. For a
finite prior `pi(x)` and noisy observation channel `W(y|x)`, the best possible
success probability for returning any candidate in the correct target class is

\[
P^*_{\mathrm{class}}(X\mid Y)
=\sum_y\max_a\sum_{x:c(x)=a}\pi(x)W(y\mid x).
\]

For each observed `y`, any decision rule must choose one class `a`. Its joint
correct mass is the inner class sum, so choosing the largest class posterior is
optimal independently for every `y`; summing over observations proves the
formula. The deterministic result follows by setting
`W(y|x)=1[g(x)=y]`.

This theorem is the correct formal target for datasets with permutation,
isomorphism, tokenization, or application-defined equivalences. Exact-candidate
success and class success must be reported separately.

## Acceptance gate for an end-to-end advantage claim

All conditions below must hold simultaneously:

1. the target equivalence relation and candidate prior are fixed before evaluation;
2. the coherent verifier is bit-level equivalent to a public reference evaluator;
3. all work qubits are uncomputed and phase marking is correct;
4. the number of marked candidates and non-equivalent collisions is reported;
5. oracle precision yields a non-vacuous hybrid success lower bound;
6. compiler, state preparation, inverse calls, shots, measurement, and readout are counted;
7. the classical baseline uses the same verifier, target success, and cost unit;
8. the measured parameter region satisfies the strict break-even inequality.
