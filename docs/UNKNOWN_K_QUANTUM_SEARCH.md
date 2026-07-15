# Unknown-Marked-Count Quantum Search

## 1. Why the known-`K` assumption is unsafe

Standard fixed-iteration Grover search chooses an iteration count from the marked
fraction `K/N`. In reconstruction, however, `K` is the size of an observation
fibre or threshold-feasible set. Knowing it may itself require solving a hard
counting problem or enumerating the candidate space. Pricing a known-`K` plan
without accounting for this information can therefore make an end-to-end
comparison circular.

Q-RECON now keeps two search contracts separate:

- `optimal_standard_grover_iterations(N, K)` is an ideal known-`K` reference;
- the BBHT schedule in `qrecon.theory.unknown_k` never receives the actual `K`
  during execution.

The known-`K` curve remains useful as a lower-level oracle-model baseline, but it
must not be reported as the deployed reconstruction protocol unless `K` is
public or a counting procedure and its cost are included.

## 2. Randomized-iteration schedule

Let the search population be `N` and let the unknown positive marked count be
`K`. In round `r`, the algorithm has an integer window `m_r`, chooses

\[
j_r\sim \operatorname{Uniform}\{0,\ldots,m_r-1\},
\]

performs `j_r` Grover iterations, measures a candidate, and verifies it. The
window starts at one and grows geometrically,

\[
M_{r+1}=\min\{\lambda M_r,\sqrt N\},
\qquad 1<\lambda<4/3,
\]

with the executable integer window `m_r=ceil(M_r)`. The complete schedule depends
only on `N`, `lambda`, and a public stopping rule. It is independent of `K`.

Every measured candidate consumes a verification query. Q-RECON reports these
verification queries separately from phase-oracle calls so they cannot disappear
from the resource total.

## 3. Exact one-round success

Set

\[
\theta_K=\arcsin\sqrt{K/N}.
\]

Conditioned on reaching a round with window `m`, exact success is

\[
p_m(K)=\frac1m\sum_{j=0}^{m-1}
\sin^2((2j+1)\theta_K).
\]

For `0<K<N`, this has the closed form

\[
p_m(K)=\frac12-
\frac{\sin(4m\theta_K)}{4m\sin(2\theta_K)}.
\]

The endpoint values are `p_m(0)=0` and `p_m(N)=1`. The implementation
`randomized_grover_round_success` evaluates this expression and the test suite
cross-checks it against the explicit finite sum.

## 4. Exact finite schedule evaluation

For windows `m_1,...,m_R`, let

\[
f_r(K)=\prod_{s=1}^{r}(1-p_{m_s}(K))
\]

be failure probability after round `r`, with `f_0(K)=1`. Then

\[
P_{\mathrm{succ}}(K)=1-f_R(K).
\]

The probability of reaching round `r` is `f_{r-1}(K)`. Since the mean randomly
chosen Grover iteration count is `(m_r-1)/2`, expected phase-oracle calls are

\[
\mathbb E[Q_{\mathrm{phase}}\mid K]
=\sum_{r=1}^{R}f_{r-1}(K)\frac{m_r-1}{2},
\]

and expected verification queries are

\[
\mathbb E[Q_{\mathrm{verify}}\mid K]
=\sum_{r=1}^{R}f_{r-1}(K).
\]

`evaluate_bbht_schedule` returns every round's reach probability, conditional
success, success mass, cumulative success, and expected phase calls, together
with aggregate phase and verification counts.

## 5. Uniform finite certificate

### Theorem 1 — finite unknown-`K` success certificate

Fix a finite population `N`, a public lower bound `K_min>=1`, a target success
`0<eta<1`, and a schedule prefix `S`. If exhaustive evaluation gives

\[
\min_{K\in\{K_{\min},\ldots,N\}}
P_{\mathrm{succ}}(S,K)\ge\eta,
\]

then executing `S` reaches success at least `eta` for every allowed unknown
marked count, without using the actual `K` to choose any window.

#### Proof

For each integer `K` in the declared range, the formula in Section 4 is the exact
probability of the randomized algorithm because it conditions on every reached
round and averages over every allowed iteration choice. The schedule is identical
for all `K`. Taking the finite minimum proves the simultaneous guarantee.
\(\square\)

`certify_bbht_uniform_success` builds geometric prefixes and returns the first
one satisfying this exact finite minimum. It also records:

- the marked count attaining the smallest certified success;
- maximum expected phase-oracle calls over the allowed `K` range;
- maximum expected verification queries;
- worst-case finite phase and verification call counts.

The certification computation is an offline audit over `K`, not information
provided to the online search. A guard prevents accidental use on populations
too large for exact enumeration; larger studies must use a proved analytic bound
or a separately validated interval argument.

## 6. Relation to the standard BBHT result

The geometric randomized-iteration method and its expected
`O(sqrt(N/K))` query scaling are standard quantum-search results. Q-RECON does
not claim them as a new theorem. The project-specific contribution must be the
same-task integration:

1. a non-enumerative clean reconstruction verifier;
2. an identifiable or explicitly collision-audited target;
3. an unknown-`K` protocol or a justified public `K`;
4. compiler, diffusion, verification, state preparation, and measurement costs;
5. comparison against the strongest structure-aware classical solver.

## 7. Zero-solution and collision cases

A positive marked-count lower bound is essential. If `K=0`, no search algorithm
can return a valid candidate, and a finite sequence of failed measurements does
not by itself certify emptiness. Experiments must either prove existence, use a
separate emptiness/counting procedure, or report the zero-solution failure mode.

If `K>1`, amplification returns a member of the marked set. It does not determine
which member generated the private observation. Original-record success,
equivalence-class success, fibre size, and Bayes ceilings remain separate from
search success.

## 8. Cost-accounting requirements

An end-to-end report using the unknown-`K` schedule must include at least:

- every phase-oracle invocation inside randomized Grover iterations;
- one measured-candidate verification per reached round;
- diffusion cost for every Grover iteration;
- repeated state preparation and readout for every round;
- inverse/uncomputation calls already embedded in the clean oracle;
- stopping rule and maximum round count;
- any cost used to establish `K_min`, existence, or a tighter marked-count range.

The exact finite certificate removes an unrealistic known-`K` assumption. It does
not by itself establish an end-to-end quantum advantage.

## 9. Primary reference

M. Boyer, G. Brassard, P. Høyer, and A. Tapp, “Tight Bounds on Quantum
Searching,” *Fortschritte der Physik*, 46(4–5), 1998.
