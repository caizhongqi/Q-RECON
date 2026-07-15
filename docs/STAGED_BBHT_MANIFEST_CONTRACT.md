# Staged BBHT Manifest Contract

## 1. Purpose

A reproducible search experiment must execute the search parameters recorded in
its manifest. Merely hashing `target_success`, `growth_factor`,
`attempts_per_stage`, or `max_stages` while the runner silently uses different
defaults invalidates both the artifact identity and every derived cost figure.

Q-RECON therefore binds `FixedPointMLPBenchmarkManifest` directly to the staged
unknown-marked-count certificate used by each benchmark result.

## 2. Public schedule

For population `N`, growth `lambda`, attempts per stage `A`, and stage limit `S`,
define the public stage windows

\[
m_s=\left\lceil\min\{\lambda^{s-1},\sqrt N\}\right\rceil,
\qquad s=1,\ldots,S,
\]

with `1 < lambda < 4/3`. Stage `s` repeats the same randomized Grover round `A`
times. The online window sequence is therefore

\[
(\underbrace{m_1,\ldots,m_1}_{A},
 \underbrace{m_2,\ldots,m_2}_{A},\ldots).
\]

No window depends on the hidden marked count `K`.

## 3. Exact finite certificate

Let

\[
p_m(K)=\frac1m\sum_{j=0}^{m-1}
\sin^2((2j+1)\arcsin\sqrt{K/N})
\]

be one-round success. After `t` complete stages, the exact failure probability is

\[
F_t(K)=\prod_{s=1}^{t}(1-p_{m_s}(K))^A.
\]

For a declared positive-count promise `K >= K_min`, the compiler returns the
first complete-stage prefix satisfying

\[
\min_{K=K_{\min},\ldots,N}(1-F_t(K))\ge\eta,
\]

where `eta` is the manifest target success.

### Theorem — manifest-level uniform guarantee

If `certify_staged_bbht_uniform_success` returns a certificate for a manifest,
then the exact online schedule encoded in every successful benchmark result has
success at least `eta` for every allowed positive integer marked count, without
using the actual `K` to choose a window.

#### Proof

The schedule is determined solely by the manifest tuple
`(N, lambda, A, S, K_min, eta)`. For each allowed `K`, multiplying the exact
conditional failure factors gives `F_t(K)`. The certifier exhaustively evaluates
every integer `K` in the declared finite range and returns only after their
minimum success reaches `eta`. Therefore the single returned schedule satisfies
the bound simultaneously. `square`

## 4. Query accounting

For every reached attempt with window `m`, expected phase-oracle calls are
`(m-1)/2` and one measured-candidate verification is charged. Repeating a stage
is not treated as a free confidence adjustment. The resulting certificate
records:

- exact window sequence;
- total rounds;
- worst marked count for success;
- certified minimum success;
- maximum expected phase-oracle calls;
- maximum expected verification queries;
- finite worst-case total oracle calls.

The zero-solution decision layer is specified separately in
`UNKNOWN_K_ZERO_SOLUTION_DECISION.md`.

## 5. Executable binding

The manifest runner forwards, without renaming or omission:

- `target_success`;
- `bbht_growth_factor`;
- `bbht_attempts_per_stage`;
- `bbht_max_stages`;
- Z3 enablement and timeout.

`run_fixed_point_mlp_benchmark` passes these values to
`certify_staged_bbht_uniform_success` and serializes the effective values inside
`bbht_certificate`. A regression test asserts the complete forwarded keyword
mapping, preventing a future artifact from hashing parameters that are not
actually executed.

## 6. Failure semantics

If the stage limit cannot certify the requested success, the run is recorded as
an error by the manifest executor. CI-smoke reports may aggregate cells with at
least one successful repeat while retaining failure counts and exception hashes.
Publication mode requires complete cells. Neither mode may silently substitute a
longer schedule, lower target, favorable hidden `K`, or known-`K` iteration count.

## 7. Claim boundary

This certificate proves an exact finite query-model success contract. It does not
prove low physical cost, verifier correctness, state-preparation efficiency, or
quantum advantage over a specialized classical solver. Those costs remain in the
end-to-end phase diagram.
