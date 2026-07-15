# Unknown-K Search with a Certified Zero-Solution Decision

## 1. Problem

A reconstruction verifier may have an unknown marked count

\[
K\in\{0,1,\ldots,N\}.
\]

A positive-only search guarantee is incomplete when the feasible set can be
empty. Selecting the Grover iteration count from the hidden `K` is also invalid.
Q-RECON therefore uses a public marked-count-independent BBHT schedule and an
exact final verifier.

The online decision rule is:

1. execute the fixed randomized-iteration BBHT schedule;
2. after every measurement, verify the measured candidate exactly;
3. return `present` immediately after an accepted verification;
4. return `empty` only when the complete schedule terminates without acceptance.

## 2. One-sided decision theorem

Assume the final verification query has no false acceptance. Let a public BBHT
schedule detect a verified candidate with probability at least `p` for every
promised positive marked count

\[
K\in\{K_{\min},K_{\min}+1,\ldots,N\}.
\]

Then the decision rule above satisfies:

- for `K=0`, `Pr[empty]=1`;
- for every promised positive `K`, `Pr[present]>=p`;
- false-present probability is zero;
- false-empty probability is at most `1-p`.

### Proof

When `K=0`, no basis candidate satisfies the verifier, so exact post-measurement
verification can never accept. The schedule therefore terminates with `empty`
with probability one. When `K>0`, the rule returns `present` exactly on the event
that the BBHT schedule samples and verifies a marked candidate. By the uniform
positive-`K` certificate this event has probability at least `p`; the remaining
probability is the false-empty event. `square`

For `K_min=1`, this is a promise-free bounded-error existence decision over all
`K` in `[0,N]`. For `K_min>1`, marked counts `1,...,K_min-1` are deliberately
reported as outside the certificate rather than silently included.

## 3. Zero-case cost

The zero case reaches every round because no verification can terminate early.
For a schedule with integer windows `m_r`, its exact expected costs are

\[
E[Q_{phase}\mid K=0]=\sum_r \frac{m_r-1}{2},
\]

\[
E[Q_{verify}\mid K=0]=R,
\]

where `R` is the number of rounds. These are not hidden worst-case costs: the
implementation records them explicitly in
`BBHTExistenceDecisionCertificate.zero_case_expected_*`.

For positive `K`, early successful rounds reduce expected calls; those costs are
computed by the existing exact finite BBHT evaluator.

## 4. Executable certificate

`qrecon.theory.unknown_k_decision` provides:

- `certify_bbht_existence_decision`;
- `evaluate_bbht_existence_decision`;
- `BBHTExistenceDecisionCertificate`;
- `BBHTExistenceDecisionEvaluation`.

The certificate wraps the exhaustive all-positive-`K` schedule proof and adds the
exact zero-case decision and cost semantics. Tests certify every `K` for a small
finite population, the one-sided error properties, zero-case expected calls, and
the explicit promise gap when `K_min>1`.

## 5. Claim boundary

The zero false-positive statement assumes an exact final verifier. An approximate
or noisy verifier needs separate false-acceptance accounting and repeated or
fault-tolerant verification. The result also does not make the absence decision
free: its full no-early-stop cost must enter the end-to-end envelope. Finally,
this is a query-model decision guarantee, not evidence of practical quantum
advantage over specialized classical inversion.
