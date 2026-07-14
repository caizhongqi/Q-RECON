# Q-RECON Theory Foundations

## 0. Scope and claim discipline

This document fixes the mathematical object studied by Q-RECON before any claim
of quantum advantage is made. The central rule is:

> Information limits, query limits, and implementation costs are separate
> statements. A result in one layer does not imply a result in another.

The current executable theory package covers finite candidate spaces, exact
Bayes reconstruction bounds, local differential identifiability, ideal
unstructured search, approximate-oracle degradation, and a transparent
end-to-end cost inequality. It does **not** yet prove that a realistic compiled
neural-network oracle yields end-to-end quantum advantage.

The intended paper contributions should therefore distinguish three classes of
results:

1. **Foundational results reused and specialized here:** Bayes decision rules,
   the inverse-function theorem, Helstrom discrimination, and Grover/BBBV query
   bounds.
2. **Q-RECON-specific results to prove:** collision structure of concrete
   training/leakage maps, correctness and complexity of the coherent model
   compiler, and break-even regimes after all costs are instantiated.
3. **Empirical claims:** measured reconstruction quality, resource counts,
   robustness to noise, and comparisons with classical attacks.

## 1. Reconstruction experiment

Let \(\mathcal X\) be a candidate space. An element \(X\in\mathcal X\) may be a
single training record, a minibatch, a complete dataset, or an equivalence class
of datasets. Let

\[
X\sim \pi
\]

be the attacker prior. A training-and-release mechanism induces an observation
channel

\[
W(y\mid x)=\Pr[Y=y\mid X=x].
\]

This channel can include training randomness, gradient noise, quantization,
partial parameter release, randomized defenses, and measurement noise. A
reconstructor is a decision rule \(R:\mathcal Y\to\mathcal X\), and its exact
success is

\[
P_{\mathrm{succ}}(R)=\Pr[R(Y)=X].
\]

For structured data, exact equality may be the wrong target. Fix a target
relation \(\approx\), such as equality up to training-set permutation, graph
isomorphism, tokenization equivalence, or an application-defined semantic
relation. The event then becomes

\[
R(Y)\approx X.
\]

All exact-identifiability claims must state the target relation explicitly.

### 1.1 Observation equivalence

Two candidates are observation-equivalent when they induce the same released
information:

\[
x\equiv_W x'
\quad\Longleftrightarrow\quad
W(\cdot\mid x)=W(\cdot\mid x').
\]

A necessary condition for exact recovery modulo \(\approx\) is

\[
x\equiv_W x'\;\Longrightarrow\;x\approx x'.
\]

If this implication fails, no classical or quantum post-processing of the same
observation can identify which non-equivalent candidate generated it.

For a deterministic observation map \(g:\mathcal X\to\mathcal Y\), define the
fibre

\[
F_y=\{x\in\mathcal X:g(x)=y\}.
\]

The fibres are the exact collision classes of the leakage.

## 2. Exact information-theoretic reconstruction bounds

### Theorem 1 — deterministic fibre optimum

Let \(\mathcal X\) be finite, let \(X\sim\pi\), and let \(Y=g(X)\) be
deterministic. The optimal exact-reconstruction probability is

\[
P^*_{\mathrm{guess}}(X\mid Y)
=\sum_{y\in g(\mathcal X)}\max_{x\in F_y}\pi(x).
\]

#### Proof

For a fixed observation \(y\), any estimator must output one candidate
\(R(y)\in F_y\). Its contribution to total success is \(\pi(R(y))\), which is
maximized by choosing a highest-prior element of the fibre. Summing the
independent optimum over all fibres gives the expression. \(\square\)

### Corollary 1 — uniform prior

For \(|\mathcal X|=N\) and a uniform prior,

\[
P^*_{\mathrm{guess}}(X\mid g(X))
=\frac{|g(\mathcal X)|}{N}.
\]

Thus a many-to-one observation map has an exact, attack-independent ceiling.
Optimization quality cannot exceed it.

### Theorem 2 — noisy-channel optimum

For a finite observation channel \(W(y\mid x)\),

\[
P^*_{\mathrm{guess}}(X\mid Y)
=\sum_y\max_x \pi(x)W(y\mid x).
\]

#### Proof

For each observed \(y\), the joint mass assigned to a correct decision \(x\) is
\(\pi(x)W(y\mid x)\). Maximizing this quantity independently for every \(y\)
and summing gives the Bayes decision rule and its exact value. \(\square\)

The corresponding classical conditional min-entropy is

\[
H_{\min}(X\mid Y)=-\log_2 P^*_{\mathrm{guess}}(X\mid Y).
\]

### Theorem 3 — data processing for reconstruction

Suppose \(X\to Y\to Z\) is a Markov chain, where \(Z\) is obtained only by
stochastic post-processing of \(Y\). Then

\[
P^*_{\mathrm{guess}}(X\mid Z)
\le P^*_{\mathrm{guess}}(X\mid Y).
\]

#### Proof

Let \(Q(z\mid y)\) be the post-processing channel. Then

\[
\begin{aligned}
P^*_{\mathrm{guess}}(X\mid Z)
&=\sum_z\max_x\pi(x)\sum_y W(y\mid x)Q(z\mid y)\\
&\le\sum_z\sum_y Q(z\mid y)\max_x\pi(x)W(y\mid x)\\
&=\sum_y\max_x\pi(x)W(y\mid x).
\end{aligned}
\]

The last equality uses \(\sum_zQ(z\mid y)=1\). \(\square\)

This theorem rules out a common overclaim: a more sophisticated decoder cannot
create information absent from the released transcript. It may approach the
Bayes optimum more closely, but it cannot move the optimum itself.

### Theorem 4 — all-pairs pure-privacy ceiling

Assume a uniform prior over \(N\) candidates and the strong condition

\[
W(y\mid x)\le e^\varepsilon W(y\mid x')
\quad\text{for every }x,x',y.
\]

Then

\[
P^*_{\mathrm{guess}}(X\mid Y)
\le \frac{e^\varepsilon}{e^\varepsilon+N-1}.
\]

#### Proof

For an observation \(y\), choose \(x^*\) maximizing \(W(y\mid x)\). The
all-pairs condition gives

\[
W(y\mid x)\ge e^{-\varepsilon}W(y\mid x^*)
\]

for every other candidate. Hence the largest posterior mass is at most

\[
\frac{W(y\mid x^*)}
{W(y\mid x^*)+(N-1)e^{-\varepsilon}W(y\mid x^*)}
=\frac{e^\varepsilon}{e^\varepsilon+N-1}.
\]

Averaging over \(y\) preserves the bound. \(\square\)

**Scope warning.** This is an all-pairs condition. Ordinary dataset differential
privacy is usually defined only for neighbouring datasets. Applying this bound
to a sparse adjacency graph without an additional group-privacy argument would
be invalid.

## 3. Local versus global identifiability

Let a differentiable leakage map be

\[
g:\mathbb R^d\to\mathbb R^m,
\]

for example

\[
g(x)=\operatorname{vec}(\nabla_\theta L(f_\theta(x),t)).
\]

### Theorem 5 — full-column-rank local certificate

If \(g\) is continuously differentiable near \(x_0\) and

\[
\operatorname{rank}J_g(x_0)=d,
\]

then there is a neighbourhood \(U\) of \(x_0\) on which \(g\) is injective.

#### Proof

Full column rank implies that some \(d\times d\) row minor of \(J_g(x_0)\) is
nonsingular. Let \(P\) select those output coordinates and define
\(h=P\circ g:\mathbb R^d\to\mathbb R^d\). The Jacobian \(J_h(x_0)\) is
nonsingular. The inverse-function theorem gives a neighbourhood on which \(h\)
is one-to-one. If \(g(x)=g(x')\) in that neighbourhood, then
\(h(x)=h(x')\), so \(x=x'\). \(\square\)

This theorem justifies the existing full-gradient Jacobian rank calculation as
a **sufficient local certificate**.

It does not establish global injectivity. In particular:

- distant candidates may still collide;
- rank deficiency is not a proof of non-identifiability (for example,
  \(x\mapsto x^3\) is injective although its derivative vanishes at zero);
- a poor smallest singular value implies local instability under observation
  noise even when the rank is full;
- a local certificate for one sample says nothing about aggregated batches.

Global identifiability must be established by collision analysis, structural
arguments, exhaustive enumeration on finite spaces, or a global inverse result.

## 4. Classical, white-box, and coherent access

Q-RECON uses three non-interchangeable access models.

### C-Access: classical channel access

A query is a classical input and produces a classical sample from a channel:

\[
x\longmapsto Y\sim W(\cdot\mid x).
\]

The interface measures or otherwise classicalizes the query boundary. A quantum
computer may optimize which classical query to issue next, but it does not
receive a coherent oracle call for free.

### W-Access: released internals

The attacker receives parameters, per-sample gradients, aggregated gradients,
updates, activations, optimizer states, or combinations thereof. W-Access is
represented by a deterministic or noisy observation channel and is therefore
subject to Sections 1–3.

### Q-Access: coherent reversible access

A quantized function \(f_b\) is supplied as a unitary

\[
U_{f_b}|x\rangle|y\rangle
=|x\rangle|y\oplus f_b(x)\rangle.
\]

This interface preserves superposition across candidate inputs. Q-Access is a
strictly stronger assumption than a normal prediction API and must be justified
as a separate threat model.

The paper must not infer Q-Access from C-Access. It must either posit Q-Access
explicitly, or account for the cost and feasibility of constructing the
coherent oracle from available model information.

## 5. Candidate verification and query complexity

Let \(\mathcal C\) contain \(N\) structured candidate datasets or latent codes.
A verifier marks

\[
v(x)=\mathbf 1[E(x)\le\tau],
\]

and suppose exactly \(K\) candidates are marked.

### Proposition 6 — exact classical sampling success

If a classical algorithm queries \(q\) uniformly chosen distinct candidates,
then

\[
P_{\mathrm C}(q)
=1-\frac{\binom{N-K}{q}}{\binom{N}{q}}
\]

for \(q\le N-K\), and the success is one for \(q>N-K\). The expected position
of the first marked candidate in a random permutation is

\[
\mathbb E[Q_{\mathrm C}]=\frac{N+1}{K+1}.
\]

### Standard quantum search result

Let

\[
\theta=\arcsin\sqrt{K/N}.
\]

Ideal standard Grover search has success after \(r\) oracle iterations

\[
P_{\mathrm Q}(r)=\sin^2((2r+1)\theta),
\]

with a first optimum near

\[
r^*=\frac{\pi}{4\theta}-\frac12.
\]

For constant bounded error in the black-box model, unstructured classical search
requires \(\Theta(N/K)\) queries while coherent quantum search requires
\(\Theta(\sqrt{N/K})\) oracle calls. The lower and upper bounds are standard
oracle-model results; they are not by themselves a new Q-RECON contribution.

The Q-RECON contribution must instead show that:

1. the reconstruction objective can be evaluated by a correct coherent
   verifier;
2. the candidate set and number of acceptable solutions are defined without
   leaking the answer into the prior;
3. the reversible verifier cost does not erase the query reduction;
4. an output candidate is evaluated against the original-sample/equivalence
   target, not merely against a low objective value.

## 6. Approximate coherent oracles

A compiled oracle will generally be quantized or approximate. Define its
per-query operational error by

\[
\delta=\sup_{\rho_{AR}}
\frac12\left\|
(\mathcal O\otimes I_R)(\rho)-
(\widetilde{\mathcal O}\otimes I_R)(\rho)
\right\|_1.
\]

The reference system \(R\) is included so that the bound remains valid for
entangled algorithm states.

### Theorem 7 — hybrid degradation bound

If an ideal algorithm makes \(q\) oracle calls and succeeds with probability
\(p\), replacing every call by an implementation with operational error at most
\(\delta\) yields success at least

\[
\max\{0,p-q\delta\}.
\]

#### Proof sketch

Replace oracle calls one at a time. Contractivity of trace distance under the
intervening channels bounds the change caused by each replacement by \(\delta\).
The triangle inequality gives final-state distance at most \(q\delta\). The
probability of any measurement event differs by no more than that distance.
\(\square\)

This creates an explicit precision obligation: a quadratic query reduction is
not meaningful if the compiled oracle error grows so quickly that
\(q\delta\) is order one.

## 7. End-to-end cost model

Query complexity is not wall-clock, gate, energy, or monetary complexity. Choose
one common cost unit and define

\[
C_{\mathrm C}(M)=S_{\mathrm C}+M(F_{\mathrm C}+q_{\mathrm C}c_{\mathrm C}),
\]

\[
C_{\mathrm Q}(M)=S_{\mathrm Q}+M(F_{\mathrm Q}+q_{\mathrm Q}c_{\mathrm Q}),
\]

where:

- \(M\) is the number of reconstruction instances sharing setup;
- \(S\) is one-time setup or compilation cost;
- \(F\) is fixed per-instance encoding, measurement, and post-processing cost;
- \(q\) is the number of verifier/oracle queries;
- \(c\) is the cost per query in the same unit.

Define variable per-instance costs

\[
V_{\mathrm C}=F_{\mathrm C}+q_{\mathrm C}c_{\mathrm C},
\qquad
V_{\mathrm Q}=F_{\mathrm Q}+q_{\mathrm Q}c_{\mathrm Q}.
\]

### Proposition 8 — break-even condition

Strict quantum cost advantage holds exactly when

\[
S_{\mathrm Q}-S_{\mathrm C}
<M(V_{\mathrm C}-V_{\mathrm Q}).
\]

If \(V_{\mathrm Q}<V_{\mathrm C}\), the minimum amortized workload must satisfy

\[
M>
\frac{S_{\mathrm Q}-S_{\mathrm C}}
{V_{\mathrm C}-V_{\mathrm Q}}.
\]

If both \(S_{\mathrm Q}\ge S_{\mathrm C}\) and
\(V_{\mathrm Q}\ge V_{\mathrm C}\), no workload can produce an advantage under
this model.

A resource table must therefore report at least:

- data/candidate state preparation;
- model-oracle compilation;
- logical qubits and reusable/peak ancillas;
- Toffoli or non-Clifford count;
- fault-tolerant T-count and T-depth under a stated synthesis precision;
- full circuit depth;
- oracle calls, including inverse/uncomputation calls;
- repetitions and shots;
- measurement and classical decoding;
- classical baseline cost measured under the same task and success target.

Mixing incomparable units—for example, comparing quantum oracle calls with
classical seconds—is not a valid break-even claim.

## 8. Binary quantum-state information bound

If two candidate datasets produce quantum states \(\rho_0\) and \(\rho_1\) with
priors \(p_0\) and \(p_1\), the optimal binary discrimination success is

\[
P^*_{\mathrm{Hel}}=
\frac12\left(1+\|p_0\rho_0-p_1\rho_1\|_1\right).
\]

For equal priors this becomes

\[
P^*_{\mathrm{Hel}}=
\frac12+\frac14\|\rho_0-\rho_1\|_1.
\]

If \(\rho_0=\rho_1\), no measurement—including a collective quantum
measurement—can distinguish the candidates beyond choosing the more likely
prior. Quantum computation does not repair an observation collision.

## 9. Theorem-to-code mapping

| Mathematical object | Executable implementation |
|---|---|
| deterministic fibres | `qrecon.theory.observation_fibres` |
| Theorem 1 | `bayes_reconstruction_success` |
| Corollary 1 | `uniform_fibre_success` |
| Theorem 2 | `channel_bayes_reconstruction_success` |
| Theorem 3 | `postprocess_channel` plus Bayes-success comparison |
| conditional min-entropy | `conditional_min_entropy_bits` |
| Theorem 4 | `all_pairs_epsilon_private_uniform_bound` |
| Helstrom binary optimum | `binary_helstrom_success` |
| Proposition 6 | `classical_success_without_replacement` |
| ideal Grover curve | `grover_success` and query helpers |
| Proposition 8 | `AlgorithmCost` and break-even helpers |
| Theorem 7 | `oracle_error_success_lower_bound` |

These implementations are intended for exact small-space verification,
regression tests, and experiment reports. They are not substitutes for formal
proofs.

## 10. Required Q-RECON-specific theorems

The following results remain the actual research frontier for this repository.
They should be promoted to paper theorems only after complete proofs and matching
implementations exist.

### T-A: training-map collision characterization

For a specified model, loss, optimizer, leakage interface, batch size, and target
equivalence, characterize when

\[
\mathcal O(A(D_1),D_1)=\mathcal O(A(D_2),D_2).
\]

A useful theorem must go beyond the already-known batch-one biased-linear
identity and cover a nontrivial model or aggregation regime.

### T-B: coherent compiler correctness

For every supported quantized model \(f_b\) and every valid input word \(x\), the
compiler must produce a clean reversible circuit satisfying

\[
U_{f_b}|x\rangle|0\rangle|0\rangle
=|x\rangle|0\rangle|f_b(x)\rangle.
\]

The theorem must state arithmetic, overflow, activation, and approximation
semantics.

### T-C: compiler resource bound

Derive logical-qubit, depth, and non-Clifford complexity as functions of layer
width, weight precision, activation precision, and architecture. Empirical gate
counts should validate, not replace, the symbolic bound.

### T-D: end-to-end advantage region

Combine the search-query bound, compiler resource bound, oracle precision bound,
and classical baseline cost to identify a nonempty parameter region in which

\[
C_{\mathrm Q}<C_{\mathrm C}
\]

at the same reconstruction success and target equivalence.

## 11. Claim ledger

| Claim | Current status |
|---|---|
| batch-one raw input can be recovered from a directly connected biased linear layer under full gradient leakage | implemented and tested under explicit assumptions |
| full-column gradient Jacobian rank certifies local injectivity | mathematically justified; numerical implementation exists |
| deterministic/noisy observation collisions impose Bayes ceilings | proved here and executable |
| a normal classical API supplies coherent quantum access | **false; must not be claimed** |
| the current VQC prior gives a query-complexity advantage | **not established** |
| standard Grover search gives a black-box quadratic query reduction under Q-Access | standard result; executable ideal curve added |
| a compiled neural verifier preserves that advantage end to end | pending compiler, correctness proof, and cost instantiation |

## 12. Primary references

- C. W. Helstrom, *Quantum Detection and Estimation Theory*, Academic Press,
  1976.
- L. K. Grover, “A Fast Quantum Mechanical Algorithm for Database Search,”
  STOC, 1996.
- C. H. Bennett, E. Bernstein, G. Brassard, and U. Vazirani, “Strengths and
  Weaknesses of Quantum Computing,” *SIAM Journal on Computing*, 26(5), 1997.
- C. Dwork, F. McSherry, K. Nissim, and A. Smith, “Calibrating Noise to
  Sensitivity in Private Data Analysis,” TCC, 2006.
